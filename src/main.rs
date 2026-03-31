use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::module::Param;
use burn::module::{AutodiffModule, Module};
use burn::nn::{LayerNorm, LayerNormConfig};
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::Distribution;
use burn::tensor::activation::relu;
use burn::tensor::activation::sigmoid;
use burn::tensor::{Shape, Tensor, backend::Backend};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::sync::mpsc::sync_channel;
use std::thread;
use std::time::{Duration, Instant};

// ==========================================
// 1. CẤU HÌNH HỆ THỐNG
// ==========================================
const MAX_VARS: usize = 300;
const MAX_CLAUSES: usize = 2400;
const BATCH_SIZE: usize = 64; // Giữ Batch 64 để VRAM thở mượt mà
const HIDDEN_DIM: usize = 64;
const MP_STEPS: usize = 8;

// ==========================================
// 2. DATA STRUCTURES
// ==========================================
pub struct SatBatchData {
    pub incidence_matrix: Vec<f32>,
    pub targets: Vec<f32>,
    pub var_mask: Vec<f32>,
    pub clause_mask: Vec<f32>,
}

// ==========================================
// 3. GENERATOR THREAD (CPU CỰC NHẸ - ĐÃ TỐI ƯU)
// ==========================================
fn run_data_worker(tx: std::sync::mpsc::SyncSender<SatBatchData>) {
    let mut rng = SmallRng::from_entropy();

    // TUYỆT CHIÊU: Cấp phát mảng 1 LẦN DUY NHẤT. Tránh CPU phải tạo mảng 184MB liên tục.
    let mut batch_incidence = vec![0.0; BATCH_SIZE * MAX_CLAUSES * MAX_VARS];
    let mut batch_targets = vec![0.0; BATCH_SIZE * MAX_VARS];
    let mut batch_var_mask = vec![0.0; BATCH_SIZE * MAX_VARS];
    let mut batch_clause_mask = vec![0.0; BATCH_SIZE * MAX_CLAUSES];

    loop {
        // Reset sạch mảng cực nhanh bằng .fill() thay vì vec![]
        batch_incidence.fill(0.0);
        batch_targets.fill(0.0);
        batch_var_mask.fill(0.0);
        batch_clause_mask.fill(0.0);

        for b in 0..BATCH_SIZE {
            let n = rng.gen_range(100..=MAX_VARS);
            let m = rng.gen_range(n..=std::cmp::min(8 * n, MAX_CLAUSES));

            for i in 0..n {
                batch_var_mask[b * MAX_VARS + i] = 1.0;
            }
            for j in 0..m {
                batch_clause_mask[b * MAX_CLAUSES + j] = 1.0;
            }

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

        if tx
            .send(SatBatchData {
                incidence_matrix: batch_incidence.clone(),
                targets: batch_targets.clone(),
                var_mask: batch_var_mask.clone(),
                clause_mask: batch_clause_mask.clone(),
            })
            .is_err()
        {
            break;
        }
    }
}

// ==========================================
// 4. KIẾN TRÚC GNN (QUAY LẠI MATMUL - SIÊU TỐC TRÊN GPU)
// ==========================================
#[derive(Module, Debug)]
pub struct BipartiteSatModel<B: Backend> {
    var_init: Param<Tensor<B, 1>>,
    clause_init: Param<Tensor<B, 1>>,
    msg_var_to_pos: Linear<B>,
    msg_var_to_neg: Linear<B>,
    msg_clause_to_pos: Linear<B>,
    msg_clause_to_neg: Linear<B>,
    var_mlp: Linear<B>,
    clause_mlp: Linear<B>,
    var_norm: LayerNorm<B>,
    clause_norm: LayerNorm<B>,
    output_layer: Linear<B>,
}

impl<B: Backend> BipartiteSatModel<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            var_init: Param::from_tensor(Tensor::random(
                [HIDDEN_DIM],
                Distribution::Normal(0.0, 0.1),
                device,
            )),
            clause_init: Param::from_tensor(Tensor::random(
                [HIDDEN_DIM],
                Distribution::Normal(0.0, 0.1),
                device,
            )),
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
        incidence: Tensor<B, 3>, // [Batch, M, N]
        var_mask: Tensor<B, 2>,
        clause_mask: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let batch_size = incidence.dims()[0];
        let device = incidence.device();

        let var_mask_3d = var_mask
            .clone()
            .reshape(Shape::new([batch_size, MAX_VARS, 1]));
        let clause_mask_3d = clause_mask
            .clone()
            .reshape(Shape::new([batch_size, MAX_CLAUSES, 1]));

        let zeros_v = Tensor::<B, 3>::zeros([batch_size, MAX_VARS, HIDDEN_DIM], &device);
        let zeros_c = Tensor::<B, 3>::zeros([batch_size, MAX_CLAUSES, HIDDEN_DIM], &device);

        let mut var_emb =
            (zeros_v + self.var_init.val().reshape([1, 1, HIDDEN_DIM])) * var_mask_3d.clone();
        let mut clause_emb =
            (zeros_c + self.clause_init.val().reshape([1, 1, HIDDEN_DIM])) * clause_mask_3d.clone();

        // TÍNH MASK 1 LẦN DUY NHẤT TRƯỚC VÒNG LẶP. KHÔNG ĐƯỢC TÍNH LẠI TRONG LOOP!
        let pos_mask = relu(incidence.clone());
        let neg_mask = relu(incidence.neg());
        let pos_mask_t = pos_mask.clone().transpose();
        let neg_mask_t = neg_mask.clone().transpose();

        for _ in 0..MP_STEPS {
            // Bước A: Biến -> Ngoặc
            let var_pos_msg = self.msg_var_to_pos.forward(var_emb.clone());
            let var_neg_msg = self.msg_var_to_neg.forward(var_emb.clone());

            // GPU làm matmul cực nhanh vì không tốn bộ nhớ trung gian Gather
            let clause_signals =
                pos_mask.clone().matmul(var_pos_msg) + neg_mask.clone().matmul(var_neg_msg);
            let clause_update = relu(self.clause_mlp.forward(clause_signals));
            clause_emb =
                self.clause_norm.forward(clause_emb + clause_update) * clause_mask_3d.clone();

            // Bước B: Ngoặc -> Biến
            let clause_pos_msg = self.msg_clause_to_pos.forward(clause_emb.clone());
            let clause_neg_msg = self.msg_clause_to_neg.forward(clause_emb.clone());

            let var_signals = pos_mask_t.clone().matmul(clause_pos_msg)
                + neg_mask_t.clone().matmul(clause_neg_msg);
            let var_update = relu(self.var_mlp.forward(var_signals));
            var_emb = self.var_norm.forward(var_emb + var_update) * var_mask_3d.clone();
        }

        let logits = self.output_layer.forward(var_emb).squeeze(2);
        logits + (var_mask - 1.0) * 10000.0
    }
}

// ==========================================
// 5. CÁC HÀM VALIDATOR CŨ (GIỮ NGUYÊN)
// ==========================================
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
        for i in 0..MAX_VARS {
            sample[i] = rng.gen_bool(spectrum[i] as f64);
        }

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
                break;
            }
        }
        if all_clauses_sat {
            return (true, attempts);
        }
        if attempts & 255 == 0 {
            if start_time.elapsed() >= time_limit {
                break;
            }
        }
    }
    (false, attempts)
}

fn generate_fixed_validation_set(num_samples: usize) -> SatBatchData {
    let mut rng = SmallRng::from_entropy();
    let mut batch_incidence = vec![0.0; num_samples * MAX_CLAUSES * MAX_VARS];
    let mut batch_targets = vec![0.0; num_samples * MAX_VARS];
    let mut batch_var_mask = vec![0.0; num_samples * MAX_VARS];
    let mut batch_clause_mask = vec![0.0; num_samples * MAX_CLAUSES];

    for b in 0..num_samples {
        let n = rng.gen_range(100..=MAX_VARS);
        let m = rng.gen_range(n..=std::cmp::min(8 * n, MAX_CLAUSES));
        for i in 0..n {
            batch_var_mask[b * MAX_VARS + i] = 1.0;
        }
        for j in 0..m {
            batch_clause_mask[b * MAX_CLAUSES + j] = 1.0;
        }

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
        var_mask: batch_var_mask,
        clause_mask: batch_clause_mask,
    }
}

fn masked_bce_loss<B: Backend>(
    logits: Tensor<B, 2>,
    targets: Tensor<B, 2>,
    mask: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let probs = burn::tensor::activation::sigmoid(logits);
    let eps = 1e-7;
    let p_safe = probs.clamp_min(eps).clamp_max(1.0 - eps);
    let loss_elements = ((targets.clone() * p_safe.clone().log())
        + ((targets.neg() + 1.0) * (p_safe.neg() + 1.0).log()))
    .neg();
    (loss_elements * mask.clone()).sum() / mask.sum().clamp_min(1.0)
}

// ==========================================
// 6. MAIN LOOP: VUA TỐC ĐỘ
// ==========================================
fn main() {
    type MyBackend = Autodiff<Wgpu>;
    let device = WgpuDevice::default();

    let mut model = BipartiteSatModel::<MyBackend>::new(&device);
    let mut optim = AdamConfig::new().init();
    let (tx, rx) = sync_channel::<SatBatchData>(5);

    thread::spawn(move || {
        println!("🚀 Khởi động Lò phản ứng Data (CPU tối ưu 100%)...");
        run_data_worker(tx);
    });

    println!("🧪 Đang khởi tạo bộ đề thi chuẩn 100 câu...");
    let val_set = generate_fixed_validation_set(100);

    let val_incidence =
        Tensor::<Wgpu, 1>::from_floats(val_set.incidence_matrix.as_slice(), &device).reshape([
            100,
            MAX_CLAUSES,
            MAX_VARS,
        ]);
    let val_var_mask = Tensor::<Wgpu, 1>::from_floats(val_set.var_mask.as_slice(), &device)
        .reshape([100, MAX_VARS]);
    let val_clause_mask = Tensor::<Wgpu, 1>::from_floats(val_set.clause_mask.as_slice(), &device)
        .reshape([100, MAX_CLAUSES]);

    let mut iteration = 0;
    let start_time = Instant::now();
    let time_per_test = Duration::from_millis(10);

    println!(
        "🔥 Bắt đầu Train loop... (LƯU Ý: 10 vòng lặp đầu tiên WGPU cần biên dịch Kernel, vui lòng kiên nhẫn chờ!)"
    );

    loop {
        iteration += 1;
        let is_log_step = iteration % 10 == 0;

        let incidence_tensor;
        let target_tensor;
        let var_mask_tensor;
        let clause_mask_tensor;
        {
            let batch_data = rx.recv().unwrap();
            incidence_tensor = Tensor::<MyBackend, 1>::from_floats(
                batch_data.incidence_matrix.as_slice(),
                &device,
            )
            .reshape([BATCH_SIZE, MAX_CLAUSES, MAX_VARS]);
            target_tensor =
                Tensor::<MyBackend, 1>::from_floats(batch_data.targets.as_slice(), &device)
                    .reshape([BATCH_SIZE, MAX_VARS]);
            var_mask_tensor =
                Tensor::<MyBackend, 1>::from_floats(batch_data.var_mask.as_slice(), &device)
                    .reshape([BATCH_SIZE, MAX_VARS]);
            clause_mask_tensor =
                Tensor::<MyBackend, 1>::from_floats(batch_data.clause_mask.as_slice(), &device)
                    .reshape([BATCH_SIZE, MAX_CLAUSES]);
        }

        let logits = model.forward(
            incidence_tensor,
            var_mask_tensor.clone(),
            clause_mask_tensor,
        );
        let loss = masked_bce_loss(logits, target_tensor, var_mask_tensor);

        // Không block CPU ở mỗi vòng lặp nếu không phải log step
        let loss_val: f32 = if is_log_step {
            loss.clone().into_data().value[0]
        } else {
            0.0
        };

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(1e-3, model, grads);
        MyBackend::sync(&device);

        if is_log_step {
            let val_probs_vec: Vec<f32>;
            {
                let test_model = model.valid();
                let val_logits = test_model.forward(
                    val_incidence.clone(),
                    val_var_mask.clone(),
                    val_clause_mask.clone(),
                );
                val_probs_vec = sigmoid(val_logits).into_data().value;
                <Wgpu as Backend>::sync(&device);
            }

            let mut solved_count = 0;
            let mut total_attempts = 0;
            let num_tests = 100;

            for b in 0..num_tests {
                let spectrum_slice = &val_probs_vec[b * MAX_VARS..(b + 1) * MAX_VARS];
                let incidence_slice = &val_set.incidence_matrix
                    [b * (MAX_CLAUSES * MAX_VARS)..(b + 1) * (MAX_CLAUSES * MAX_VARS)];
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

            println!(
                "Iter: {:04} | Train Loss: {:.4} | Val Success: {:>5.1}% ({}/{}) | Avg Attempts/{}ms: {} | Total Time: {}s",
                iteration,
                loss_val,
                (solved_count as f32 / num_tests as f32) * 100.0,
                solved_count,
                num_tests,
                time_per_test.as_millis(),
                total_attempts / num_tests as u32,
                start_time.elapsed().as_secs()
            );
        }
    }
}
