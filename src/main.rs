use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::{relu, sigmoid};
use burn::tensor::loss::binary_cross_entropy_with_logits;
use burn::tensor::{backend::Backend, Shape, Tensor};
use burn::optim::{AdamConfig, Optimizer};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::sync::mpsc::{sync_channel};
use std::thread;
use std::time::{Instant, Duration};

// ==========================================
// 1. CẤU HÌNH HỆ THỐNG "THỊT ĐÈ NGƯỜI"
// ==========================================
const MAX_VARS: usize = 300;     // N
const MAX_CLAUSES: usize = 2400; // M tối đa (8 * 300)
const BATCH_SIZE: usize = 32;
const HIDDEN_DIM: usize = 128;
const MP_STEPS: usize = 3;       // Số vòng Message Passing (Lan truyền tín hiệu)

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
                    if solution[v] == sign { is_satisfied = true; }
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
        if tx.send(SatBatchData { incidence_matrix: batch_incidence, targets: batch_targets }).is_err() {
            break; // Main thread đã chết -> Thoát worker
        }
    }
}

// ==========================================
// 4. KIẾN TRÚC BURN MODEL (WGPU)
// ==========================================
#[derive(Module, Debug)]
pub struct BipartiteSatModel<B: Backend> {
    var_mlp: Linear<B>,
    clause_mlp: Linear<B>,
    output_layer: Linear<B>,
}

impl<B: Backend> BipartiteSatModel<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            var_mlp: LinearConfig::new(MAX_VARS, HIDDEN_DIM).init(device),
            clause_mlp: LinearConfig::new(MAX_CLAUSES, HIDDEN_DIM).init(device),
            output_layer: LinearConfig::new(HIDDEN_DIM, 1).init(device),
        }
    }

    pub fn forward(
        &self, 
        incidence: Tensor<B, 3>, // [Batch, M, N]
    ) -> Tensor<B, 2> { // Trả về [Batch, N] (Logits)
        
        // Khởi tạo Embedding cơ bản (Tận dụng batch size)
        let batch_size = incidence.dims()[0];
        let mut var_emb = Tensor::<B, 3>::zeros([batch_size, MAX_VARS, HIDDEN_DIM], &incidence.device());

        let incidence_t = incidence.clone().transpose(); // [Batch, N, M]

        // Message Passing Vòng lặp
        for _ in 0..MP_STEPS {
            // Bước 1: Biến -> Ngoặc (Nhân ma trận [B, M, N] x [B, N, D] -> [B, M, D])
            let clause_signals = incidence.clone().matmul(var_emb.clone());
            let clause_emb = relu(self.clause_mlp.forward(clause_signals));

            // Bước 2: Ngoặc -> Biến (Nhân ma trận [B, N, M] x [B, M, D] -> [B, N, D])
            let var_signals = incidence_t.clone().matmul(clause_emb);
            var_emb = relu(self.var_mlp.forward(var_signals));
        }

        // Xuất phổ logits [Batch, N, 1] -> Bóp lại thành [Batch, N]
        let logits = self.output_layer.forward(var_emb).squeeze(2);
        logits
    }
}

// ==========================================
// 5. VALIDATOR SIÊU TỐC (1ms/10k MẪU TRÊN CPU)
// ==========================================
// Trả về tuple: (Có tìm thấy nghiệm không?, Số mẫu đã check được trong thời gian đó)
fn verify_spectrum_time_bound(spectrum: &[f32], incidence_flat: &[f32], time_limit_ms: u64) -> (bool, u32) {
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

// ==========================================
// 6. MAIN LOOP: TRÁI TIM CỦA HỆ THỐNG
// ==========================================
fn main() {
    type MyBackend = Autodiff<Wgpu>;
    let device = WgpuDevice::default();

    // Khởi tạo Model & Optimizer
    let mut model = BipartiteSatModel::<MyBackend>::new(&device);
    let mut optim = AdamConfig::new().with_learning_rate(1e-3).init();

    // Tạo Channel Buffer 5 Batches (CPU gen chạy trước, GPU chỉ việc húp)
    let (tx, rx) = sync_channel::<SatBatchData>(5);

    // Bắn Thread Data Generator
    thread::spawn(move || {
        println!("🚀 Khởi động Lò phản ứng Data (CPU Worker)...");
        run_data_worker(tx);
    });

    let mut iteration = 0;
    let start_time = Instant::now();

    println!("🔥 Bắt đầu Train loop (Train xong vứt)...");

    loop {
        iteration += 1;

        // 1. Kéo data từ Channel (Chớp mắt)
        let batch_data = rx.recv().unwrap();

        // 2. Chuyển lên VRAM (GPU)
        let incidence_tensor = Tensor::<MyBackend, 1>::from_floats(batch_data.incidence_matrix.as_slice(), &device)
            .reshape(Shape::new([BATCH_SIZE, MAX_CLAUSES, MAX_VARS]));
        
        let target_tensor = Tensor::<MyBackend, 1>::from_floats(batch_data.targets.as_slice(), &device)
            .reshape(Shape::new([BATCH_SIZE, MAX_VARS]));

        // 3. Forward Pass
        let logits = model.forward(incidence_tensor.clone());
        let loss = binary_cross_entropy_with_logits(logits.clone(), target_tensor);

        // 4. Backward Pass & Update Trọng số
        let grads = loss.backward();
        let grads = optim.step(1e-3, model.clone(), grads);
        model = model.valid(); // Cập nhật model

        // 5. Validation (Cứ mỗi 50 Iterations check tốc độ 1 phát)
        if iteration % 50 == 0 {
            let loss_val = loss.into_data().value[0];
            
            // Lấy phổ xác suất của 1 sample trong batch bằng sigmoid
            let probs = sigmoid(logits).into_data();
            let spectrum_slice = &probs.value[0..MAX_VARS]; // Phổ biến 0 -> 300 của sample đầu tiên
            
            // Trích cấu trúc ngoặc của sample đó để Verifier check
            let incidence_slice = &batch_data.incidence_matrix[0..(MAX_CLAUSES * MAX_VARS)];
            
            // Giới hạn 1 mili-giây
            let (is_success, attempts_made) = verify_spectrum_time_bound(spectrum_slice, incidence_slice, 1);

            println!(
                "Iter: {:04} | Loss: {:.4} | Success: {} | Attempts in 1ms: {} | Total Time: {}s",
                iteration, loss_val, if is_success {"✅"} else {"❌"}, attempts_made, start_time.elapsed().as_secs()
            );
        }
    }
}
