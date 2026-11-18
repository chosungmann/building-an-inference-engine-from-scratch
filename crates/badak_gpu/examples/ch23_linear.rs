//! 23장: 선형 층의 커널들을 CPU와 견준다.
//!
//! 실행: `cargo run --release -p badak_gpu --example ch23_linear`
//!
//! Llama 3.2 1B의 선형 층에서 나오는 형상들로 잰다. 값은 f32다.

use std::time::Instant;

use badak_bench::stats::Summary;
use badak_core::random::Pcg64;
use badak_core::{Tensor, ops};
use badak_gpu::Gpu;
use badak_gpu::linear::{Linear, Variant};

fn values(len: usize, seed: u64) -> Vec<f32> {
    let mut rng = Pcg64::new(seed);
    (0..len)
        .map(|_| (rng.next_u64() >> 40) as f32 / (1 << 23) as f32 - 1.0)
        .collect()
}

fn median(samples: &[f64]) -> f64 {
    Summary::new(samples).median
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(gpu) = Gpu::new() else {
        println!("이 기계에는 쓸 수 있는 GPU가 없습니다.");
        return Ok(());
    };
    println!("GPU: {}", gpu.name());
    let kernels: Vec<(Variant, Linear)> = Variant::ALL
        .into_iter()
        .filter(|variant| variant.available(&gpu))
        .map(|variant| (variant, Linear::new(&gpu, variant)))
        .collect();

    let shapes = [
        ("토큰 1개, MLP의 첫 층", 1, 8192, 2048),
        ("토큰 32개, MLP의 첫 층", 32, 8192, 2048),
        ("토큰 256개, 질의를 만드는 층", 256, 2048, 2048),
        ("토큰 256개, MLP의 첫 층", 256, 8192, 2048),
        ("토큰 256개, MLP의 끝 층", 256, 2048, 8192),
    ];
    println!(
        "{:<18} {:>10} {:>10} {:>14} {:>12} {:>12}",
        "커널", "시간 (ms)", "GPU 시계", "곱셈/초 (억)", "가장 큰 차이", "같은 비트"
    );
    for (label, count, rows, cols) in shapes {
        let (x, w) = (values(count * cols, 1), values(rows * cols, 2));
        let products = (count * rows * cols) as f64;
        let repetitions = if products > 1e9 { 7 } else { 30 };

        // CPU: 18장까지의 커널. 스레드를 모두 쓴다.
        let (x_tensor, w_tensor) = (
            Tensor::from_vec(x.clone(), &[count, cols])?,
            Tensor::from_vec(w.clone(), &[rows, cols])?,
        );
        let expected = ops::linear(&x_tensor, &w_tensor, None)?;
        let times: Vec<f64> = (0..repetitions)
            .map(|_| {
                let start = Instant::now();
                std::hint::black_box(ops::linear(&x_tensor, &w_tensor, None).unwrap());
                start.elapsed().as_secs_f64()
            })
            .collect();
        let cpu = median(&times);
        println!("{label}: 벡터 {count}개 × 행 {rows}개 × 길이 {cols}");
        println!(
            "  {:<16} {:>10.3} {:>10} {:>14.0} {:>12} {:>12}",
            "CPU (18장)",
            cpu * 1e3,
            "-",
            products / cpu / 1e8,
            "-",
            "-"
        );
        let expected = expected.as_slice::<f32>()?;

        let (x_buffer, w_buffer) = (gpu.upload(&x), gpu.upload(&w));
        let y_buffer = gpu.zeros::<f32>(count * rows);
        for (variant, kernel) in &kernels {
            if !variant.supports(count, rows, cols) {
                continue;
            }
            let shape = gpu.uniform(&[count as u32, rows as u32, cols as u32, 0]);
            let buffers = [&shape, &x_buffer, &w_buffer, &y_buffer];
            let groups = kernel.groups(count, rows);
            gpu.run_and_wait(kernel.kernel(), &buffers, groups);
            let mut wall = Vec::new();
            let mut inside = Vec::new();
            for _ in 0..repetitions {
                let (outer, inner) = gpu.run_timed(kernel.kernel(), &buffers, groups);
                wall.push(outer.as_secs_f64());
                inside.extend(inner.map(|d| d.as_secs_f64()));
            }
            let y: Vec<f32> = gpu.read(&y_buffer, count * rows);
            // 가장 크게 어긋난 값의 상대 오차와, CPU와 비트까지 같은 값의 비율.
            let worst = y
                .iter()
                .zip(expected)
                .map(|(&got, &want)| (got - want).abs() / want.abs().max(1.0))
                .fold(0.0_f32, f32::max);
            let same = y
                .iter()
                .zip(expected)
                .filter(|(got, want)| got.to_bits() == want.to_bits())
                .count();
            let seconds = median(&wall);
            // 이름의 한글은 화면에서 두 칸을 차지한다.
            let name = variant.name();
            let width = 16 - name.chars().filter(|c| !c.is_ascii()).count();
            println!(
                "  {:<width$} {:>10.3} {:>10} {:>14.0} {:>12.1e} {:>11.1}%",
                name,
                seconds * 1e3,
                if inside.is_empty() {
                    "-".to_string()
                } else {
                    format!("{:.3}", median(&inside) * 1e3)
                },
                products / seconds / 1e8,
                worst,
                100.0 * same as f64 / y.len() as f64
            );
        }
    }
    Ok(())
}
