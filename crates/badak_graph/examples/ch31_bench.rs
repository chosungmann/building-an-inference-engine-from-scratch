//! 31장: 합성곱의 커널을 바꿔 가며 사진 한 장의 시간을 잰다.
//!
//! 실행: `cargo run --release -p badak_graph --example ch31_bench <모델>
//! <neon|sme>`
//!
//! 모델은 27장의 `.tflite`이고, 30장의 패스와 제자리 계산을 켠다. `sme`는
//! 합성곱을 SME의 외적 명령으로 계산한다. 스레드의 수는 환경 변수
//! `BADAK_THREADS`로 정한다. 한 번 실행에 한 가지만 잰다(22.12절).

use std::hint::black_box;

use badak_bench::stats::Summary;
use badak_bench::timing::{Plan, measure};
use badak_core::threads;
use badak_formats::safetensors::SafeTensors;
use badak_formats::tflite;
use badak_graph::{Graph, Interpreter, Registry, Strategy, models_dir, optimize};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let [name, kernel] = &arguments[..] else {
        return Err("사용법: ch31_bench <모델> <neon|sme>".into());
    };
    #[cfg_attr(not(target_arch = "aarch64"), allow(unused_mut))]
    let mut registry = Registry::builtin();
    match kernel.as_str() {
        "neon" => {}
        #[cfg(target_arch = "aarch64")]
        "sme" => {
            if !badak_core::simd::sme::available() {
                return Err("이 기계에는 SME가 없습니다".into());
            }
            registry.register("CONV_2D", badak_graph::outer::conv_2d_outer);
        }
        other => return Err(format!("모르는 커널 {other}").into()),
    }
    let directory = models_dir().join("vision_tflite");
    let model = tflite::Model::open(directory.join(format!("{name}.tflite")))?;
    let mut graph = Graph::from_tflite(&model)?;
    optimize(&mut graph, &registry)?;
    let mut interpreter = Interpreter::in_place(&graph, &registry, &[], Strategy::BySize)?;

    let reference = SafeTensors::open(directory.join("reference.safetensors"))?;
    let input = reference.tensor("cup.input")?.to_vec::<f32>()?;
    let expected = reference
        .tensor(&format!("{name}.cup.litert"))?
        .to_vec::<f32>()?;
    let plan = Plan {
        warmup: 5,
        repetitions: 51,
    };
    let samples = measure(plan, || {
        interpreter.input_mut(0).copy_from_slice(&input);
        interpreter.run().expect("실행");
        black_box(interpreter.output(0)[0]);
    });
    if std::env::var_os("BADAK_PROFILE").is_some() {
        // 연산의 종류마다 걸린 시간을 모은다.
        let mut times = vec![std::time::Duration::ZERO; interpreter.step_count()];
        let rounds = 20;
        for _ in 0..rounds {
            interpreter.input_mut(0).copy_from_slice(&input);
            interpreter.run_timed(&mut times)?;
        }
        let mut by_kind = std::collections::BTreeMap::new();
        for (step, time) in interpreter.steps().zip(&times) {
            let window = match (step.name, step.inputs.get(1)) {
                ("CONV_2D", Some(&(_, &[_, rows, columns, _]))) => format!(" {rows}×{columns}"),
                _ => String::new(),
            };
            *by_kind
                .entry(format!("{}{window}", step.name))
                .or_insert(0.0) += time.as_secs_f64() * 1e3 / f64::from(rounds);
        }
        for (kind, time) in &by_kind {
            println!("  {kind:<22} {time:>7.2}ms");
        }
    }
    let gap = interpreter
        .output(0)
        .iter()
        .zip(&expected)
        .map(|(ours, theirs)| (ours - theirs).abs())
        .fold(0.0, f32::max);
    let summary = Summary::new(&samples);
    println!(
        "{name} {kernel:<4} 스레드 {:>2} 중앙값 {:>7.2}ms 최소 {:>7.2}ms LiteRT와의 차이 {gap:.2e}",
        threads::global().pool().threads(),
        summary.median * 1e3,
        summary.min * 1e3
    );
    Ok(())
}
