use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::module::Module;
use burn::nn::loss::BinaryCrossEntropyLossConfig;
use burn::nn::{LayerNorm, LayerNormConfig};
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::activation::relu;
use burn::tensor::activation::sigmoid;
use burn::tensor::{Shape, Tensor, backend::Backend};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::sync::mpsc::sync_channel;
use std::thread;
use std::time::{Duration, Instant};

// ==========================================
// 1. CẤU HÌNH HỆ THỐNG "THỊT ĐÈ NGƯỜI"
// ==========================================
const MAX_VARS: usize = 300; // N
const MAX_CLAUSES: usize = 2400; // M tối đa (8 * 300)
const BATCH_SIZE: usize = 32;
const HIDDEN_DIM: usize = 128;
const MP_STEPS: usize = 3; // Số vòng Message Passing (Lan truyền tín hiệu)

// ==========================================
// 2. DATA STRUCTURES
// ==========================================
pub struct SatBatchData {
    // Ma trận liên thuộc [BATCH, MAX_CLAUSES, MAX_VARS]
    // 1.0 = Khẳng định, -1.0 = Phủ định, 0.0 = Không tham gia
    pub incidence_matrix: Vec<f32>,
    // Nhãn [BATCH, MAX_VARS]
    pub targets: Vec<f32>,
}

// ==========================================
// 3. GENERATOR THREAD (CHẠY TRÊN CPU)
// ==========================================
fn run_data_worker(tx: std::sync::mpsc::SyncSender<SatBatchData>) {
    let mut rng = SmallRng::from_entropy(); // Dùng SmallRng cho tốc độ xé gió

    loop {
        let mut batch_incidence = vec![0.0; BATCH_SIZE * MAX_CLAUSES * MAX_VARS];
        let mut batch_targets = vec![0.0; BATCH_SIZE * MAX_VARS];

        for b in 0..BATCH_SIZE {
            let n = rng.gen_range(100..=MAX_VARS);
            let m = rng.gen_range(n..=std::cmp::min(8 * n, MAX_CLAUSES));

            // Planted Solution
            let solution: Vec<bool> = (0..n).map(|_| rng.gen_bool(0.5)).collect();

            // Ghi Target
            for i in 0..n {
                batch_targets[b * MAX_VARS + i] = if solution[i] { 1.0 } else { 0.0 };
            }

            // Gen Ngoặc (Clauses)
            for clause_idx in 0..m {
                let k = rng.gen_range(3..=15);
                let mut is_satisfied = false;
                let mut clause_vars = Vec::with_capacity(k);
                let mut clause_signs = Vec::with_capacity(k);

                for _ in 0..k {
                    let v = rng.gen_range(0..n);
                    let sign = rng.gen_bool(0.5);
                    clause_vars.push(v);
                    clause_signs.push(sign);
                    if solution[v] == sign {
                        is_satisfied = true;
                    }
                }

                // Cứu ngoặc nếu sai toàn tập
                if !is_satisfied {
                    let lucky = rng.gen_range(0..k);
                    clause_signs[lucky] = !clause_signs[lucky];
                }

                // Điền vào Incidence Matrix
                for idx in 0..k {
                    let v = clause_vars[idx];
                    let sign_val = if clause_signs[idx] { 1.0 } else { -1.0 };
                    let flat_idx = b * (MAX_CLAUSES * MAX_VARS) + clause_idx * MAX_VARS + v;
                    batch_incidence[flat_idx] = sign_val;
                }
            }
        }

        // Đẩy batch vào channel. Nếu GPU đang bận thì CPU sẽ block (chờ).
        if tx
            .send(SatBatchData {
                incidence_matrix: batch_incidence,
                targets: batch_targets,
            })
            .is_err()
        {
            break; // Main thread đã chết -> Thoát worker
        }
    }
}

// ==========================================
// 4. KIẾN TRÚC BURN MODEL V2 (GNN CHUẨN MỰC)
// ==========================================
#[derive(Module, Debug)]
pub struct BipartiteSatModel<B: Backend> {
    // Các lớp chiếu (Projection) để phân tách Khẳng định/Phủ định
    msg_var_to_pos: Linear<B>,
    msg_var_to_neg: Linear<B>,
    msg_clause_to_pos: Linear<B>,
    msg_clause_to_neg: Linear<B>,

    // Mạng MLP xử lý thông tin tại Node
    var_mlp: Linear<B>,
    clause_mlp: Linear<B>,

    // Bộ chuẩn hóa (Tránh bệnh "đồng hóa" vector)
    var_norm: LayerNorm<B>,
    clause_norm: LayerNorm<B>,

    output_layer: Linear<B>,
}

impl<B: Backend> BipartiteSatModel<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            msg_var_to_pos: LinearConfig::new(HIDDEN_DIM, HIDDEN_DIM).init(device),
            msg_var_to_neg: LinearConfig::new(HIDDEN_DIM, HIDDEN_DIM).init(device),
            msg_clause_to_pos: LinearConfig::new(HIDDEN_DIM, HIDDEN_DIM).init(device),
            msg_clause_to_neg: LinearConfig::new(HIDDEN_DIM, HIDDEN_DIM).init(device),

            var_mlp: LinearConfig::new(HIDDEN_DIM, HIDDEN_DIM).init(device),
            clause_mlp: LinearConfig::new(HIDDEN_DIM, HIDDEN_DIM).init(device),

            var_norm: LayerNormConfig::new(HIDDEN_DIM).init(device),
            clause_norm: LayerNormConfig::new(HIDDEN_DIM).init(device),

            output_layer: LinearConfig::new(HIDDEN_DIM, 1).init(device),
        }
    }

    pub fn forward(
        &self,
        incidence: Tensor<B, 3>, // [Batch, M, N] chứa 1.0, -1.0, 0.0
    ) -> Tensor<B, 2> {
        let batch_size = incidence.dims()[0];

        // 1. Khởi tạo TRÍ NHỚ (Memory) cho cả Biến và Ngoặc
        let mut var_emb =
            Tensor::<B, 3>::zeros([batch_size, MAX_VARS, HIDDEN_DIM], &incidence.device());
        let mut clause_emb =
            Tensor::<B, 3>::zeros([batch_size, MAX_CLAUSES, HIDDEN_DIM], &incidence.device());

        // 2. Tách ma trận Kép: Nhận diện chính xác 100% Khẳng định và Phủ định
        let pos_mask = relu(incidence.clone()); // Chỉ giữ +1.0
        let neg_mask = relu(incidence.clone().neg()); // Ép -1.0 thành +1.0 (relu của số dương)

        let pos_mask_t = pos_mask.clone().transpose(); // [Batch, N, M]
        let neg_mask_t = neg_mask.clone().transpose(); // [Batch, N, M]

        // 3. Vòng lặp Lan truyền Đồ thị (Message Passing)
        for _ in 0..MP_STEPS {
            // ==========================================
            // BƯỚC A: BIẾN truyền tín hiệu đến NGOẶC
            // ==========================================
            // Biến "nói chuyện" theo 2 giọng khác nhau cho Khẳng định và Phủ định
            let var_pos_msg = self.msg_var_to_pos.forward(var_emb.clone());
            let var_neg_msg = self.msg_var_to_neg.forward(var_emb.clone());

            // Ngoặc tổng hợp tín hiệu (Chỉ cộng những chỗ có mặt)
            let clause_signals =
                pos_mask.clone().matmul(var_pos_msg) + neg_mask.clone().matmul(var_neg_msg);

            // Cập nhật trí nhớ của Ngoặc (Dùng Residual Connection `+ clause_emb`)
            let clause_update = relu(self.clause_mlp.forward(clause_signals));
            clause_emb = self.clause_norm.forward(clause_emb + clause_update);

            // ==========================================
            // BƯỚC B: NGOẶC phản hồi lại BIẾN
            // ==========================================
            // Ngoặc cũng phản hồi lại theo 2 giọng
            let clause_pos_msg = self.msg_clause_to_pos.forward(clause_emb.clone());
            let clause_neg_msg = self.msg_clause_to_neg.forward(clause_emb.clone());

            // Biến tổng hợp phản hồi
            let var_signals = pos_mask_t.clone().matmul(clause_pos_msg)
                + neg_mask_t.clone().matmul(clause_neg_msg);

            // Cập nhật trí nhớ của Biến
            let var_update = relu(self.var_mlp.forward(var_signals));
            var_emb = self.var_norm.forward(var_emb + var_update);
        }

        // 4. Xuất Phổ Logits
        let logits = self.output_layer.forward(var_emb).squeeze(2);
        logits
    }
}

// ==========================================
// 5. VALIDATOR SIÊU TỐC (1ms/10k MẪU TRÊN CPU)
// ==========================================
// Trả về tuple: (Có tìm thấy nghiệm không?, Số mẫu đã check được trong thời gian đó)
fn verify_spectrum_time_bound(
    spectrum: &[f32],
    incidence_flat: &[f32],
    time_limit_ms: u64,
) -> (bool, u32) {
    let mut rng = SmallRng::from_entropy();
    let start_time = Instant::now();
    let time_limit = Duration::from_millis(time_limit_ms);

    let mut attempts = 0;

    loop {
        attempts += 1;
        let mut sample = [false; MAX_VARS];

        // 1. Gen mẫu dựa trên phổ
        for i in 0..MAX_VARS {
            sample[i] = rng.gen_bool(spectrum[i] as f64);
        }

        // 2. Check ngoặc (Fail-fast)
        let mut all_clauses_sat = true;
        for c in 0..MAX_CLAUSES {
            let mut clause_has_vars = false;
            let mut clause_sat = false;

            for v in 0..MAX_VARS {
                let sign = incidence_flat[c * MAX_VARS + v];
                if sign != 0.0 {
                    clause_has_vars = true;
                    if (sign > 0.0 && sample[v]) || (sign < 0.0 && !sample[v]) {
                        clause_sat = true;
                        break;
                    }
                }
            }

            if clause_has_vars && !clause_sat {
                all_clauses_sat = false;
                break; // Tạch ngoặc này -> vứt cả mẫu
            }
        }

        if all_clauses_sat {
            return (true, attempts); // Tìm thấy nghiệm sớm!
        }

        // 3. Tối ưu: Chỉ xem đồng hồ sau mỗi 256 lần thử để tiết kiệm CPU
        // 255 trong hệ nhị phân là 11111111. Phép AND (&) này cực kỳ nhẹ.
        if attempts & 255 == 0 {
            if start_time.elapsed() >= time_limit {
                break; // Hết giờ! Rút quân!
            }
        }
    }

    (false, attempts) // Trả về false và báo cáo xem trong 1ms đã "băm" được bao nhiêu mẫu
}

fn generate_fixed_validation_set(num_samples: usize) -> SatBatchData {
    let mut rng = SmallRng::from_entropy();
    let mut batch_incidence = vec![0.0; num_samples * MAX_CLAUSES * MAX_VARS];
    let mut batch_targets = vec![0.0; num_samples * MAX_VARS];

    for b in 0..num_samples {
        let n = rng.gen_range(100..=MAX_VARS);
        let m = rng.gen_range(n..=std::cmp::min(8 * n, MAX_CLAUSES));
        let solution: Vec<bool> = (0..n).map(|_| rng.gen_bool(0.5)).collect();

        for i in 0..n {
            batch_targets[b * MAX_VARS + i] = if solution[i] { 1.0 } else { 0.0 };
        }

        for clause_idx in 0..m {
            let k = rng.gen_range(3..=15);
            let mut is_satisfied = false;
            let mut clause_vars = Vec::with_capacity(k);
            let mut clause_signs = Vec::with_capacity(k);

            for _ in 0..k {
                let v = rng.gen_range(0..n);
                let sign = rng.gen_bool(0.5);
                clause_vars.push(v);
                clause_signs.push(sign);
                if solution[v] == sign {
                    is_satisfied = true;
                }
            }

            if !is_satisfied {
                let lucky = rng.gen_range(0..k);
                clause_signs[lucky] = !clause_signs[lucky];
            }

            for idx in 0..k {
                let v = clause_vars[idx];
                let sign_val = if clause_signs[idx] { 1.0 } else { -1.0 };
                let flat_idx = b * (MAX_CLAUSES * MAX_VARS) + clause_idx * MAX_VARS + v;
                batch_incidence[flat_idx] = sign_val;
            }
        }
    }
    SatBatchData {
        incidence_matrix: batch_incidence,
        targets: batch_targets,
    }
}

// ==========================================
// 6. MAIN LOOP: TRÁI TIM CỦA HỆ THỐNG
// ==========================================
fn main() {
    type MyBackend = Autodiff<Wgpu>;
    let device = WgpuDevice::default();

    // Khởi tạo Model & Optimizer
    let mut model = BipartiteSatModel::<MyBackend>::new(&device);
    let mut optim = AdamConfig::new().init();

    // Khởi tạo Loss function
    let bce_loss = BinaryCrossEntropyLossConfig::new()
        .with_logits(true)
        .init(&device);

    // Tạo Channel Buffer 5 Batches (CPU gen chạy trước, GPU chỉ việc húp)
    let (tx, rx) = sync_channel::<SatBatchData>(5);

    // Bắn Thread Data Generator
    thread::spawn(move || {
        println!("🚀 Khởi động Lò phản ứng Data (CPU Worker)...");
        run_data_worker(tx);
    });

    // TẠO BỘ ĐỀ THI CHUẨN 100 CÂU CỐ ĐỊNH
    println!("🧪 Đang khởi tạo bộ đề thi chuẩn 100 câu...");
    let val_set = generate_fixed_validation_set(100);
    let val_incidence_tensor =
        Tensor::<MyBackend, 1>::from_floats(val_set.incidence_matrix.as_slice(), &device)
            .reshape(Shape::new([100, MAX_CLAUSES, MAX_VARS]));

    let mut iteration = 0;
    let start_time = Instant::now();

    println!("🔥 Bắt đầu Train loop (Train xong vứt)...");

    // set time per test
    let time_per_test = Duration::from_millis(10);

    loop {
        iteration += 1;

        // 1. Kéo data từ Channel (Chớp mắt)
        let batch_data = rx.recv().unwrap();

        // 2. Chuyển lên VRAM (GPU)
        let incidence_tensor =
            Tensor::<MyBackend, 1>::from_floats(batch_data.incidence_matrix.as_slice(), &device)
                .reshape(Shape::new([BATCH_SIZE, MAX_CLAUSES, MAX_VARS]));

        let target_tensor =
            Tensor::<MyBackend, 1>::from_floats(batch_data.targets.as_slice(), &device)
                .reshape(Shape::new([BATCH_SIZE, MAX_VARS]));

        // 3. Forward Pass
        let logits = model.forward(incidence_tensor.clone());
        let loss = bce_loss.forward(logits.clone(), target_tensor.int());

        // 4. Backward Pass & Update Trọng số
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(1e-3, model, grads);

        // 5. TEST TRÊN BỘ ĐỀ CỐ ĐỊNH (Cứ mỗi 50 Iterations)
        if iteration % 50 == 0 {
            let loss_val = loss.into_data().value[0];

            // Forward pass lấy phổ trên bộ đề 100 câu
            let val_logits = model.forward(val_incidence_tensor.clone());
            let val_probs = sigmoid(val_logits).into_data();

            let mut solved_count = 0;
            let mut total_attempts = 0;
            let num_tests = 100;

            for b in 0..num_tests {
                let start_idx_var = b * MAX_VARS;
                let end_idx_var = start_idx_var + MAX_VARS;
                let spectrum_slice = &val_probs.value[start_idx_var..end_idx_var];

                let start_idx_inc = b * (MAX_CLAUSES * MAX_VARS);
                let end_idx_inc = start_idx_inc + (MAX_CLAUSES * MAX_VARS);
                let incidence_slice = &val_set.incidence_matrix[start_idx_inc..end_idx_inc];

                let (is_success, attempts_made) = verify_spectrum_time_bound(
                    spectrum_slice,
                    incidence_slice,
                    time_per_test.as_millis() as u64,
                );

                if is_success {
                    solved_count += 1;
                }
                total_attempts += attempts_made;
            }

            let success_rate = (solved_count as f32 / num_tests as f32) * 100.0;
            let avg_attempts = total_attempts / num_tests as u32;

            println!(
                "Iter: {:04} | Train Loss: {:.4} | Val Success: {:>5.1}% ({}/{}) | Avg Attempts/{}ms: {} | Total Time: {}s",
                iteration,
                loss_val,
                success_rate,
                solved_count,
                num_tests,
                time_per_test.as_millis(),
                avg_attempts,
                start_time.elapsed().as_secs()
            );
        }
    }
}
