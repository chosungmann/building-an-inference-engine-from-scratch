//! 23장: GPU에 첫 일을 시키고, 일을 맡기는 값을 잰다.
//!
//! 실행: `cargo run --release -p badak_gpu --example ch23_first`
//!
//! 원소마다 SiLU를 적용하는 일을 CPU와 GPU에서 한다. 원소의 수를 늘려 가며
//! 어디서부터 GPU가 이기는지 본다.

use std::time::{Duration, Instant};

use badak_bench::stats::Summary;
use badak_core::{Tensor, nn, ops};
use badak_gpu::{Gpu, groups_for, shaders};

fn microseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e6
}

/// `work`를 `repetitions`번 되풀이해 걸린 시간의 중앙값(마이크로초)을 낸다.
fn time(repetitions: usize, mut work: impl FnMut() -> Duration) -> f64 {
    work();
    let times: Vec<f64> = (0..repetitions).map(|_| microseconds(work())).collect();
    Summary::new(&times).median
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(gpu) = Gpu::new() else {
        println!("이 기계에는 쓸 수 있는 GPU가 없습니다.");
        return Ok(());
    };
    let limits = gpu.limits();
    println!("GPU: {} ({:?})", gpu.name(), gpu.info().backend);
    println!(
        "버퍼 하나의 최대 크기 {:.1}GB, 작업 그룹 하나의 최대 스레드 {}개",
        limits.max_buffer_size as f64 / 1e9,
        limits.max_compute_invocations_per_workgroup
    );
    println!(
        "계산에 쓰는 버퍼를 CPU가 곧바로 읽기: {}, GPU의 시계 읽기: {}",
        if gpu.can_map_storage() {
            "됨"
        } else {
            "안 됨"
        },
        if gpu.can_time() { "됨" } else { "안 됨" }
    );

    // ANCHOR: first
    // 1. 셰이더를 GPU의 기계어로 옮긴다.
    let start = Instant::now();
    let kernel = gpu.kernel("silu", shaders::SILU, "main");
    let compile = start.elapsed();

    // 2. 값을 버퍼에 담고, 결과를 받을 버퍼를 만든다.
    let x: Vec<f32> = (0..8).map(|i| i as f32 - 3.5).collect();
    let (input, output) = (gpu.upload(&x), gpu.zeros::<f32>(x.len()));

    // 3. 실행을 명령하고, 결과를 읽어 온다.
    gpu.run(&kernel, &[&input, &output], groups_for(x.len()));
    let y: Vec<f32> = gpu.read(&output, x.len());
    // ANCHOR_END: first

    println!();
    println!(
        "셰이더를 옮기는 데 {:.1}밀리초",
        compile.as_secs_f64() * 1e3
    );
    println!("{:>8} {:>14} {:>14} {:>10}", "x", "GPU", "CPU", "비트");
    for (&x, &y) in x.iter().zip(&y) {
        let expected = nn::silu(x);
        let same = if y.to_bits() == expected.to_bits() {
            "같음"
        } else {
            "다름"
        };
        println!("{x:>8.1} {y:>14.8} {expected:>14.8} {same:>10}");
    }

    // 일을 맡기는 값: 원소 256개(그룹 하나)짜리 일을 되풀이한다.
    println!();
    println!("일을 맡기는 값 (원소 256개, 되풀이 500번의 중앙값)");
    let small: Vec<f32> = vec![0.5; 256];
    let (input, output) = (gpu.upload(&small), gpu.zeros::<f32>(256));
    let round_trip = time(500, || {
        gpu.run_and_wait(&kernel, &[&input, &output], [1, 1, 1])
    });
    let mut gpu_side = Vec::new();
    for _ in 0..500 {
        if let (_, Some(inside)) = gpu.run_timed(&kernel, &[&input, &output], [1, 1, 1]) {
            gpu_side.push(microseconds(inside));
        }
    }
    let submit = time(500, || {
        let start = Instant::now();
        gpu.run(&kernel, &[&input, &output], [1, 1, 1]);
        let elapsed = start.elapsed();
        gpu.wait();
        elapsed
    });
    let cpu = time(500, || {
        let start = Instant::now();
        let y: Vec<f32> = small.iter().map(|&v| nn::silu(v)).collect();
        std::hint::black_box(y);
        start.elapsed()
    });
    println!("  명령을 보내고 끝나기를 기다림     {round_trip:>9.1} 마이크로초");
    println!("  그 가운데 명령을 만들어 보내기     {submit:>9.1} 마이크로초");
    if !gpu_side.is_empty() {
        println!(
            "  GPU가 스스로 잰 계산 시간          {:>9.1} 마이크로초",
            Summary::new(&gpu_side).median
        );
    }
    println!("  같은 일을 CPU의 스레드 하나로      {cpu:>9.2} 마이크로초");

    // 원소의 수를 늘려 간다.
    println!();
    println!("원소마다 SiLU (마이크로초, 되풀이의 중앙값)");
    println!(
        "{:>12} {:>12} {:>14} {:>12} {:>16} {:>10}",
        "원소 수", "CPU 1스레드", "CPU 16스레드", "GPU 계산만", "GPU 올리고 내리기", "GPU의 시계"
    );
    for power in [8, 10, 12, 14, 16, 18, 20, 22, 24] {
        let len = 1_usize << power;
        let repetitions = if len >= 1 << 22 { 10 } else { 50 };
        let values: Vec<f32> = (0..len).map(|i| (i % 1000) as f32 / 100.0 - 5.0).collect();
        let tensor = Tensor::from_vec(values.clone(), &[len])?;
        let single = time(repetitions, || {
            let start = Instant::now();
            let y: Vec<f32> = values.iter().map(|&v| nn::silu(v)).collect();
            std::hint::black_box(y);
            start.elapsed()
        });
        let threaded = time(repetitions, || {
            let start = Instant::now();
            std::hint::black_box(ops::silu(&tensor).unwrap());
            start.elapsed()
        });
        let groups = groups_for(len);
        let (input, output) = (gpu.upload(&values), gpu.zeros::<f32>(len));
        let compute = time(repetitions, || {
            gpu.run_and_wait(&kernel, &[&input, &output], groups)
        });
        let mut result = vec![0.0_f32; len];
        let with_transfer = time(repetitions, || {
            let start = Instant::now();
            let (input, output) = (gpu.upload(&values), gpu.zeros::<f32>(len));
            gpu.run(&kernel, &[&input, &output], groups);
            gpu.read_into(&output, &mut result);
            start.elapsed()
        });
        let inside: Vec<f64> = (0..repetitions)
            .filter_map(|_| gpu.run_timed(&kernel, &[&input, &output], groups).1)
            .map(microseconds)
            .collect();
        println!(
            "{:>12} {:>12.1} {:>14.1} {:>12.1} {:>16.1} {:>10}",
            len,
            single,
            threaded,
            compute,
            with_transfer,
            if inside.is_empty() {
                "-".to_string()
            } else {
                format!("{:.1}", Summary::new(&inside).median)
            }
        );
    }
    Ok(())
}
