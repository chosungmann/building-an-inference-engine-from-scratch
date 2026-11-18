//! 선형 층의 곱셈을 GPU에 맡긴다 (23.8절).
//!
//! [`Offload`]가 엔진의 행렬 하나를 받아, 가중치를 GPU의 버퍼에 올리고,
//! 곱셈을 GPU에서 하는 행렬로 바꿔 돌려준다. 순전파의 코드는 고치지 않는다.
//! 20장에서 만든 [`LinearWeight`]의 구현이 하나 더 생길 뿐이다.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use badak_core::ops::{self, LinearWeight};
use badak_core::packed::{Delegated, Matrix};
use badak_core::{Result, Tensor, profile};

use crate::Gpu;
use crate::linear::{Linear, Variant};

// ANCHOR: offload
/// 선형 층을 GPU에 맡기는 창구. 행렬들이 함께 쓴다.
pub struct Offload {
    gpu: Arc<Gpu>,
    /// 값 열여섯을 맡는 커널. 표준 WebGPU만으로 돈다.
    block: Linear,
    /// 행렬 연산을 쓰는 커널. 쓸 수 있는 GPU에서만 있다.
    cooperative: Option<Linear>,
    /// MLP의 가운데(`silu(gate) × up`)를 계산하는 커널.
    silu_mul: crate::Kernel,
    /// 입력과 결과를 담는 버퍼. 행렬들이 돌려 쓴다.
    scratch: Mutex<Scratch>,
    /// 입력 벡터가 이보다 적으면 GPU에 보내지 않고 CPU에서 계산한다.
    min_count: AtomicUsize,
    timing: Timing,
}

/// 입력, 결과, 형상을 담는 버퍼.
#[derive(Default)]
struct Scratch {
    x: Option<wgpu::Buffer>,
    y: Option<wgpu::Buffer>,
    shape: Option<wgpu::Buffer>,
    /// MLP를 통째로 계산할 때 가운데 결과 둘을 담는 버퍼.
    gate: Option<wgpu::Buffer>,
    up: Option<wgpu::Buffer>,
}
// ANCHOR_END: offload

/// GPU에 맡긴 일에 든 시간의 합(나노초)과 횟수.
#[derive(Default)]
struct Timing {
    upload: AtomicU64,
    compute: AtomicU64,
    download: AtomicU64,
    calls: AtomicU64,
    on_cpu: AtomicU64,
}

/// [`Offload::statistics`]가 돌려주는 기록.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Statistics {
    /// 입력을 GPU의 버퍼에 적는 데 든 시간.
    pub upload: Duration,
    /// 명령을 보내고 계산이 끝나기를 기다린 시간.
    pub compute: Duration,
    /// 결과를 CPU의 메모리로 읽어 오는 데 든 시간.
    pub download: Duration,
    /// GPU에서 계산한 선형 층의 수.
    pub calls: u64,
    /// 입력이 적어 CPU에서 계산한 선형 층의 수.
    pub on_cpu: u64,
}

impl Offload {
    /// 창구를 연다. 커널들을 미리 옮겨 둔다.
    pub fn new(gpu: Arc<Gpu>) -> Arc<Self> {
        Arc::new(Self {
            block: Linear::new(&gpu, Variant::Block),
            silu_mul: gpu.kernel("silu_mul", crate::shaders::SILU_MUL, "main"),
            cooperative: gpu
                .can_cooperate()
                .then(|| Linear::new(&gpu, Variant::Cooperative)),
            scratch: Mutex::default(),
            min_count: AtomicUsize::new(1),
            timing: Timing::default(),
            gpu,
        })
    }

    /// 입력 벡터가 `count`개보다 적은 일은 CPU에서 하게 한다.
    pub fn set_min_count(&self, count: usize) {
        self.min_count.store(count, Ordering::Relaxed);
    }

    /// 행렬 연산 커널을 쓰지 않게 한다. 표준 WebGPU만 쓸 때의 속도를 볼 때 쓴다.
    pub fn without_cooperative(gpu: Arc<Gpu>) -> Arc<Self> {
        Arc::new(Self {
            block: Linear::new(&gpu, Variant::Block),
            silu_mul: gpu.kernel("silu_mul", crate::shaders::SILU_MUL, "main"),
            cooperative: None,
            scratch: Mutex::default(),
            min_count: AtomicUsize::new(1),
            timing: Timing::default(),
            gpu,
        })
    }

    // ANCHOR: matrix
    /// 행렬의 값을 `f32`로 GPU에 올리고, 곱셈을 GPU에 맡기는 행렬을 돌려준다.
    pub fn matrix(self: &Arc<Self>, original: Matrix) -> Result<Matrix> {
        let values = original.to_f32()?;
        let &[rows, cols] = values.shape() else {
            unreachable!("행렬은 2차원이다");
        };
        let weight = self.gpu.upload(values.contiguous().as_slice::<f32>()?);
        let linear = OnGpu {
            shared: Arc::clone(self),
            weight,
            rows,
            cols,
            fallback: original.clone(),
        };
        Ok(Matrix::Delegated(Delegated::new(
            original,
            Arc::new(linear),
        )))
    }
    // ANCHOR_END: matrix

    // ANCHOR: mlp
    /// MLP의 세 행렬을 GPU에 올리고, MLP 전체를 GPU에서 계산하는 것을 돌려준다.
    ///
    /// 선형 층 셋과 그 사이의 계산을 한 번의 명령 묶음으로 보낸다. 가운데
    /// 결과는 GPU의 버퍼에만 있고 CPU로 오지 않는다.
    pub fn mlp(
        self: &Arc<Self>,
        gate: &Matrix,
        up: &Matrix,
        down: &Matrix,
    ) -> Result<Arc<OnGpuMlp>> {
        let upload = |matrix: &Matrix| -> Result<wgpu::Buffer> {
            let values = matrix.to_f32()?;
            Ok(self.gpu.upload(values.contiguous().as_slice::<f32>()?))
        };
        let (inner, width) = (gate.shape()[0], gate.shape()[1]);
        assert_eq!(
            up.shape(),
            [inner, width],
            "위로 펼치는 층의 형상이 다릅니다"
        );
        assert_eq!(
            down.shape(),
            [width, inner],
            "아래로 모으는 층의 형상이 다릅니다"
        );
        Ok(Arc::new(OnGpuMlp {
            shared: Arc::clone(self),
            weights: [upload(gate)?, upload(up)?, upload(down)?],
            width,
            inner,
            fallback: [gate.clone(), up.clone(), down.clone()],
        }))
    }
    // ANCHOR_END: mlp

    /// 지금까지 든 시간. `reset`이 참이면 기록을 0으로 되돌린다.
    pub fn statistics(&self, reset: bool) -> Statistics {
        let read = |value: &AtomicU64| {
            if reset {
                value.swap(0, Ordering::Relaxed)
            } else {
                value.load(Ordering::Relaxed)
            }
        };
        Statistics {
            upload: Duration::from_nanos(read(&self.timing.upload)),
            compute: Duration::from_nanos(read(&self.timing.compute)),
            download: Duration::from_nanos(read(&self.timing.download)),
            calls: read(&self.timing.calls),
            on_cpu: read(&self.timing.on_cpu),
        }
    }

    /// 이 형상에 쓸 커널. 맞는 것이 없으면 `None`.
    fn kernel(&self, rows: usize, cols: usize) -> Option<(&Linear, usize)> {
        if let Some(cooperative) = &self.cooperative
            && Variant::Cooperative.supports(32, rows, cols)
        {
            return Some((cooperative, 32));
        }
        Variant::Block
            .supports(4, rows, cols)
            .then_some((&self.block, 4))
    }
}

/// GPU에 올린 가중치 하나.
struct OnGpu {
    shared: Arc<Offload>,
    weight: wgpu::Buffer,
    rows: usize,
    cols: usize,
    /// 입력이 적을 때 CPU에서 계산할 원래의 행렬.
    fallback: Matrix,
}

// ANCHOR: apply
impl LinearWeight for OnGpu {
    fn dimensions(&self) -> Option<(usize, usize)> {
        Some((self.rows, self.cols))
    }

    fn apply(&self, x: &[f32], count: usize, y: &mut [f32]) -> Result<()> {
        let shared = &self.shared;
        let kernel = shared.kernel(self.rows, self.cols);
        let (Some((kernel, multiple)), true) =
            (kernel, count >= shared.min_count.load(Ordering::Relaxed))
        else {
            // 23.9절: 작은 일은 보내는 값이 더 크다. CPU에서 한다.
            shared.timing.on_cpu.fetch_add(1, Ordering::Relaxed);
            return self.fallback.apply(x, count, y);
        };
        let gpu = &shared.gpu;
        // 커널은 벡터를 넷이나 서른둘씩 묶어 계산한다. 모자라는 만큼은 0으로
        // 채운 벡터가 있는 것으로 한다.
        let padded = count.next_multiple_of(multiple);
        let mut scratch = shared.scratch.lock().expect("버퍼를 쥔 스레드가 죽었다");
        let Scratch {
            x: input,
            y: output,
            shape,
            ..
        } = &mut *scratch;
        let input = grown(gpu, input, padded * self.cols, true);
        let output = grown(gpu, output, padded * self.rows, false);
        let shape = shape.get_or_insert_with(|| gpu.uniform(&[0; 4]));

        // 1. 입력을 GPU의 버퍼에 적는다.
        let start = Instant::now();
        let scope = profile::scope("gpu_upload");
        gpu.write(input, 0, x);
        if padded > count {
            let zeros = vec![0.0_f32; (padded - count) * self.cols];
            gpu.write(input, x.len(), &zeros);
        }
        gpu.write(
            shape,
            0,
            &[padded as u32, self.rows as u32, self.cols as u32, 0],
        );
        let uploaded = Instant::now();
        drop(scope);

        // 2. 계산을 명령하고 기다린다.
        let scope = profile::scope("gpu_compute");
        gpu.run(
            kernel.kernel(),
            &[shape, input, &self.weight, output],
            kernel.groups(padded, self.rows),
        );
        gpu.wait();
        let computed = Instant::now();
        drop(scope);

        // 3. 결과를 읽어 온다.
        let scope = profile::scope("gpu_download");
        gpu.read_into(output, y);
        let downloaded = Instant::now();
        drop(scope);

        let timing = &shared.timing;
        let nanoseconds = |from: Instant, to: Instant| (to - from).as_nanos() as u64;
        timing
            .upload
            .fetch_add(nanoseconds(start, uploaded), Ordering::Relaxed);
        timing
            .compute
            .fetch_add(nanoseconds(uploaded, computed), Ordering::Relaxed);
        timing
            .download
            .fetch_add(nanoseconds(computed, downloaded), Ordering::Relaxed);
        timing.calls.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}
// ANCHOR_END: apply

/// GPU에 올린 MLP 하나.
pub struct OnGpuMlp {
    shared: Arc<Offload>,
    /// 문, 위로, 아래로.
    weights: [wgpu::Buffer; 3],
    /// 입력과 결과의 길이(`hidden_size`).
    width: usize,
    /// 가운데의 길이(`intermediate_size`).
    inner: usize,
    /// 입력이 적을 때 CPU에서 계산할 원래의 행렬들.
    fallback: [Matrix; 3],
}

// ANCHOR: mlp_apply
impl ops::Operator for OnGpuMlp {
    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let shared = &self.shared;
        let count = x.len() / self.width;
        let kernel = shared.kernel(self.inner, self.width);
        let (Some((kernel, multiple)), true) =
            (kernel, count >= shared.min_count.load(Ordering::Relaxed))
        else {
            // CPU에서 9.9절의 식 그대로 계산한다.
            shared.timing.on_cpu.fetch_add(3, Ordering::Relaxed);
            let [gate, up, down] = &self.fallback;
            let gate = ops::silu(&ops::linear(x, gate, None)?)?;
            return ops::linear(&ops::mul(&gate, &ops::linear(x, up, None)?)?, down, None);
        };
        let gpu = &shared.gpu;
        let padded = count.next_multiple_of(multiple);
        let (width, inner) = (self.width, self.inner);
        let mut scratch = shared.scratch.lock().expect("버퍼를 쥔 스레드가 죽었다");
        let Scratch {
            x: input,
            y: output,
            gate,
            up,
            ..
        } = &mut *scratch;
        let input = grown(gpu, input, padded * width, true);
        let output = grown(gpu, output, padded * width, false);
        let gate = grown(gpu, gate, padded * inner, false);
        let up = grown(gpu, up, padded * inner, false);

        let x = x.contiguous();
        let values = x.as_slice::<f32>()?;
        let scope = profile::scope("gpu_upload");
        gpu.write(input, 0, values);
        if padded > count {
            let zeros = vec![0.0_f32; (padded - count) * width];
            gpu.write(input, values.len(), &zeros);
        }
        // 형상은 층마다 다르므로 작은 버퍼 셋을 따로 만든다.
        let wide = gpu.uniform(&[padded as u32, inner as u32, width as u32, 0]);
        let narrow = gpu.uniform(&[padded as u32, width as u32, inner as u32, 0]);
        let length = gpu.uniform(&[(padded * inner) as u32, 0, 0, 0]);
        drop(scope);

        // 네 단계를 명령 묶음 하나에 적어 한 번에 보낸다.
        let scope = profile::scope("gpu_compute");
        let [gate_weight, up_weight, down_weight] = &self.weights;
        let mut encoder = gpu.device().create_command_encoder(&Default::default());
        let linear = kernel.kernel();
        let wide_groups = kernel.groups(padded, inner);
        gpu.encode(
            &mut encoder,
            linear,
            &[&wide, input, gate_weight, gate],
            wide_groups,
        );
        gpu.encode(
            &mut encoder,
            linear,
            &[&wide, input, up_weight, up],
            wide_groups,
        );
        gpu.encode(
            &mut encoder,
            &shared.silu_mul,
            &[&length, gate, up],
            crate::groups_for(padded * inner),
        );
        gpu.encode(
            &mut encoder,
            linear,
            &[&narrow, gate, down_weight, output],
            kernel.groups(padded, width),
        );
        gpu.queue().submit([encoder.finish()]);
        gpu.wait();
        drop(scope);

        let scope = profile::scope("gpu_download");
        let result = Tensor::build(x.shape(), |data| {
            gpu.read_into(output, data);
            Ok(())
        });
        drop(scope);
        shared.timing.calls.fetch_add(3, Ordering::Relaxed);
        result
    }
}
// ANCHOR_END: mlp_apply

/// `f32` `len`개가 들어가는 버퍼를 돌려준다. 가진 것이 작으면 새로 만든다.
/// `written`이 참이면 CPU가 값을 적어 넣는 버퍼로 만든다.
fn grown<'a>(
    gpu: &Gpu,
    buffer: &'a mut Option<wgpu::Buffer>,
    len: usize,
    written: bool,
) -> &'a wgpu::Buffer {
    let bytes = (len * size_of::<f32>()) as u64;
    if buffer.as_ref().is_none_or(|buffer| buffer.size() < bytes) {
        *buffer = Some(if written {
            gpu.input::<f32>(len)
        } else {
            gpu.zeros::<f32>(len)
        });
    }
    buffer.as_ref().expect("방금 채웠다")
}

#[cfg(test)]
mod tests {
    use super::*;
    use badak_core::ops::Operator;
    use badak_core::random::Pcg64;

    fn values(len: usize, seed: u64) -> Vec<f32> {
        let mut rng = Pcg64::new(seed);
        (0..len)
            .map(|_| (rng.next_u64() >> 40) as f32 / (1 << 23) as f32 - 1.0)
            .collect()
    }

    #[test]
    fn a_whole_mlp_on_the_gpu_agrees_with_the_cpu() {
        let Some(gpu) = Gpu::new() else {
            return;
        };
        let offload = Offload::new(Arc::new(gpu));
        let (width, inner) = (64, 96);
        let matrix = |rows: usize, cols: usize, seed: u64| {
            Matrix::from(Tensor::from_vec(values(rows * cols, seed), &[rows, cols]).unwrap())
        };
        let (gate, up, down) = (
            matrix(inner, width, 1),
            matrix(inner, width, 2),
            matrix(width, inner, 3),
        );
        let mlp = offload.mlp(&gate, &up, &down).unwrap();
        for count in [1, 32, 33] {
            let x = Tensor::from_vec(values(count * width, 10 + count as u64), &[count, width])
                .unwrap();
            let hidden = ops::silu(&ops::linear(&x, &gate, None).unwrap()).unwrap();
            let hidden = ops::mul(&hidden, &ops::linear(&x, &up, None).unwrap()).unwrap();
            let expected = ops::linear(&hidden, &down, None).unwrap();
            let got = mlp.apply(&x).unwrap();
            assert_eq!(got.shape(), [count, width]);
            for (got, want) in got
                .as_slice::<f32>()
                .unwrap()
                .iter()
                .zip(expected.as_slice::<f32>().unwrap())
            {
                assert!(
                    (got - want).abs() <= 1e-3 * want.abs().max(1.0),
                    "{got} 대 {want}"
                );
            }
        }
    }

    #[test]
    fn a_delegated_matrix_gives_the_cpu_result_for_any_count() {
        let Some(gpu) = Gpu::new() else {
            eprintln!("GPU가 없어 건너뜁니다");
            return;
        };
        let gpu = Arc::new(gpu);
        for offload in [
            Offload::new(Arc::clone(&gpu)),
            Offload::without_cooperative(Arc::clone(&gpu)),
        ] {
            let (rows, cols) = (64, 96);
            let weight = Tensor::from_vec(values(rows * cols, 7), &[rows, cols]).unwrap();
            let matrix = offload.matrix(Matrix::from(weight.clone())).unwrap();
            assert_eq!(matrix.shape(), [rows, cols]);
            // 32의 배수가 아닌 수도, 큰 것 다음에 작은 것도 맞아야 한다.
            for count in [1, 3, 32, 45, 64, 5] {
                let x =
                    Tensor::from_vec(values(count * cols, count as u64), &[count, cols]).unwrap();
                let expected = ops::linear(&x, &weight, None).unwrap();
                let got = ops::linear(&x, &matrix, None).unwrap();
                for (got, want) in got
                    .as_slice::<f32>()
                    .unwrap()
                    .iter()
                    .zip(expected.as_slice::<f32>().unwrap())
                {
                    assert!(
                        (got - want).abs() <= 1e-4 * want.abs().max(1.0),
                        "{got} 대 {want}"
                    );
                }
            }
            assert_eq!(offload.statistics(true).calls, 6);

            // 한도보다 적은 입력은 CPU에서 계산한다. 결과의 비트가 CPU와 같다.
            offload.set_min_count(8);
            let x = Tensor::from_vec(values(4 * cols, 9), &[4, cols]).unwrap();
            let expected = ops::linear(&x, &weight, None).unwrap();
            let got = ops::linear(&x, &matrix, None).unwrap();
            assert_eq!(
                got.as_slice::<f32>().unwrap(),
                expected.as_slice::<f32>().unwrap()
            );
            let statistics = offload.statistics(false);
            assert_eq!((statistics.calls, statistics.on_cpu), (0, 1));
        }
    }
}
