//! 32장: YOLOX-Tiny의 시간을 잰다. 연산의 종류마다 얼마가 드는지도 본다.
//!
//! 실행: `cargo run --release -p badak_graph --example ch32_bench [neon|sme]`
//!
//! 스레드의 수는 환경 변수 `BADAK_THREADS`로 정한다.

use std::collections::BTreeMap;
use std::hint::black_box;
use std::time::Duration;

use badak_bench::stats::Summary;
use badak_bench::timing::{Plan, measure};
use badak_core::threads;
use badak_formats::onnx::Model;
use badak_formats::safetensors::SafeTensors;
use badak_graph::{Graph, Interpreter, Registry, Strategy, models_dir, optimize};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let kernel = std::env::args().nth(1).unwrap_or_else(|| "neon".into());
    #[cfg_attr(not(target_arch = "aarch64"), allow(unused_mut))]
    let mut registry = Registry::builtin();
    match kernel.as_str() {
        "neon" => {}
        #[cfg(target_arch = "aarch64")]
        "sme" if badak_core::simd::sme::available() => {
            registry.register("CONV_2D", badak_graph::outer::conv_2d_outer);
        }
        other => return Err(format!("쓸 수 없는 커널 {other}").into()),
    }
    let directory = models_dir().join("vision_transformer");
    let mut graph = Graph::from_onnx(&Model::open(directory.join("yolox_tiny.onnx"))?)?;
    optimize(&mut graph, &registry)?;
    let mut interpreter = Interpreter::in_place(&graph, &registry, &[], Strategy::BySize)?;
    let reference = SafeTensors::open(directory.join("reference.safetensors"))?;
    let input = reference.tensor("yolox.cup.input")?.to_vec::<f32>()?;

    let plan = Plan {
        warmup: 5,
        repetitions: 51,
    };
    let samples = measure(plan, || {
        interpreter.input_mut(0).copy_from_slice(&input);
        interpreter.run().expect("실행");
        black_box(interpreter.output(0)[0]);
    });
    let summary = Summary::new(&samples);
    println!(
        "yolox_tiny {kernel:<4} 스레드 {:>2} 중앙값 {:>7.2}ms 최소 {:>7.2}ms",
        threads::global().pool().threads(),
        summary.median * 1e3,
        summary.min * 1e3
    );

    let mut times = vec![Duration::ZERO; interpreter.step_count()];
    let rounds = 20;
    for _ in 0..rounds {
        interpreter.input_mut(0).copy_from_slice(&input);
        interpreter.run_timed(&mut times)?;
    }
    let mut kinds: BTreeMap<&str, (usize, f64)> = BTreeMap::new();
    for (step, time) in interpreter.steps().zip(&times) {
        let entry = kinds.entry(step.name).or_default();
        entry.0 += 1;
        entry.1 += time.as_secs_f64() * 1e3 / f64::from(rounds);
    }
    let total: f64 = kinds.values().map(|&(_, time)| time).sum();
    for (name, (count, time)) in &kinds {
        println!(
            "  {name:<14} {count:>3}개 {time:>7.2}ms {:>5.1}%",
            100.0 * time / total
        );
    }
    Ok(())
}
