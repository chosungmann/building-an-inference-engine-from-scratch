//! 23장: 입력 벡터가 몇 개일 때부터 GPU에 맡기는 것이 이득인지 잰다.
//!
//! 실행: `cargo run --release -p badak_gpu --example ch23_split`
//!
//! MLP의 첫 층(행 8,192개, 길이 2,048)에 벡터를 1개부터 256개까지 넣는다.
//! GPU 쪽은 입력을 올리고 결과를 내려받는 시간까지 넣어서 잰다. 순전파가
//! 선형 층을 부를 때 실제로 치르는 값이다.

use std::sync::Arc;
use std::time::Instant;

use badak_bench::stats::Summary;
use badak_core::packed::Matrix;
use badak_core::random::Pcg64;
use badak_core::{Tensor, ops};
use badak_gpu::Gpu;
use badak_gpu::offload::Offload;

fn values(len: usize, seed: u64) -> Vec<f32> {
    let mut rng = Pcg64::new(seed);
    (0..len)
        .map(|_| (rng.next_u64() >> 40) as f32 / (1 << 23) as f32 - 1.0)
        .collect()
}

fn milliseconds(repetitions: usize, mut work: impl FnMut()) -> f64 {
    work();
    let times: Vec<f64> = (0..repetitions)
        .map(|_| {
            let start = Instant::now();
            work();
            start.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    Summary::new(&times).median
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(gpu) = Gpu::new() else {
        println!("이 기계에는 쓸 수 있는 GPU가 없습니다.");
        return Ok(());
    };
    let gpu = Arc::new(gpu);
    let (rows, cols) = (8192, 2048);
    let weight = Tensor::from_vec(values(rows * cols, 1), &[rows, cols])?;
    let fast = Offload::new(Arc::clone(&gpu));
    let standard = Offload::without_cooperative(Arc::clone(&gpu));
    let on_gpu = fast.matrix(Matrix::from(weight.clone()))?;
    let on_gpu_standard = standard.matrix(Matrix::from(weight.clone()))?;

    println!(
        "GPU: {}. 선형 층 하나(행 {rows}개, 길이 {cols})를 부르는 시간 (밀리초)",
        gpu.name()
    );
    println!(
        "{:>8} {:>10} {:>16} {:>14} {:>12}",
        "벡터 수", "CPU", "GPU 값 열여섯", "GPU 행렬 연산", "CPU / GPU"
    );
    for count in [1, 2, 4, 8, 16, 32, 64, 128, 256] {
        let x = Tensor::from_vec(values(count * cols, 2), &[count, cols])?;
        let repetitions = if count >= 64 { 31 } else { 101 };
        let cpu = milliseconds(repetitions, || {
            std::hint::black_box(ops::linear(&x, &weight, None).unwrap());
        });
        let block = milliseconds(repetitions, || {
            std::hint::black_box(ops::linear(&x, &on_gpu_standard, None).unwrap());
        });
        let cooperative = milliseconds(repetitions, || {
            std::hint::black_box(ops::linear(&x, &on_gpu, None).unwrap());
        });
        println!(
            "{:>8} {:>10.3} {:>16.3} {:>14.3} {:>11.2}배",
            count,
            cpu,
            block,
            cooperative,
            cpu / block.min(cooperative)
        );
    }
    Ok(())
}
