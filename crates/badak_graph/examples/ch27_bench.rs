//! 27장: 합성곱 모델의 시간과 메모리를 잰다.
//!
//! 실행: `cargo run --release -p badak_graph --example ch27_bench -- [모델]`
//!
//! 모델은 `resnet50`(기본값)이나 `mobilenetv2`다. 재는 것은 넷이다. 준비하는
//! 시간, 중간 결과의 자리, 사진 한 장의 시간(스레드 수를 바꿔 가며), 그리고
//! 그 시간이 어느 연산에 쓰이는지.

use std::collections::BTreeMap;
use std::hint::black_box;
use std::time::Duration;

use badak_bench::stats::Summary;
use badak_bench::timing::{Plan, measure};
use badak_core::threads;
use badak_formats::safetensors::SafeTensors;
use badak_formats::tflite::Model;
use badak_graph::memory::{self, Strategy};
use badak_graph::{Graph, Interpreter, Registry, conv, models_dir};

fn median(plan: Plan, work: impl FnMut()) -> f64 {
    Summary::new(&measure(plan, work)).median
}

/// 마디를 종류로 나눈다. 합성곱은 창의 크기로 더 나눈다.
fn kind(name: &str, inputs: &[(&str, &[usize])]) -> String {
    match (name, inputs.get(1)) {
        ("CONV_2D", Some((_, filter))) => format!("CONV_2D {}×{}", filter[1], filter[2]),
        _ => name.to_string(),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let name = std::env::args().nth(1).unwrap_or_else(|| "resnet50".into());
    let directory = models_dir().join("vision_tflite");
    let path = directory.join(format!("{name}.tflite"));
    let reference = SafeTensors::open(directory.join("reference.safetensors"))?;
    let input = reference.tensor("cup.input")?.to_vec::<f32>()?;
    let registry = Registry::builtin();
    let quick = Plan {
        warmup: 3,
        repetitions: 31,
    };
    badak_bench::environment::print();
    println!("\n{name}.tflite");

    // 1. 준비.
    let open = median(quick, || {
        black_box(Model::open(&path).unwrap());
    });
    let model = Model::open(&path)?;
    let convert = median(quick, || {
        black_box(Graph::from_tflite(&model).unwrap());
    });
    let graph = Graph::from_tflite(&model)?;
    let plan = median(quick, || {
        black_box(Interpreter::new(&graph, &registry).unwrap());
    });
    println!("준비");
    println!("  파일 열기와 해석   {:>9.1} µs", open * 1e6);
    println!("  그래프로 옮기기    {:>9.1} µs", convert * 1e6);
    println!("  계획 세우기        {:>9.1} µs", plan * 1e6);

    // 2. 중간 결과의 자리.
    let mut interpreter = Interpreter::new(&graph, &registry)?;
    let lifetimes = interpreter.lifetimes();
    let megabytes = |elements: usize| elements as f64 * 4.0 / 1e6;
    println!("중간 결과의 자리 (값 {}개)", lifetimes.len());
    for (label, strategy) in [
        ("값마다 따로", Strategy::Separate),
        ("순서대로 놓기", Strategy::InOrder),
        ("큰 것부터 놓기", Strategy::BySize),
    ] {
        let total = memory::plan(&lifetimes, strategy).total;
        println!("  {label:<16} {:>8.2} MB", megabytes(total));
    }
    println!(
        "  {:<16} {:>8.2} MB",
        "하한",
        megabytes(memory::lower_bound(&lifetimes))
    );
    // 창을 펼친 행렬 가운데 가장 큰 것. 계획 밖에서 따로 쥔다.
    let unfolded = interpreter
        .steps()
        .filter(|step| step.name == "CONV_2D")
        .filter_map(|step| {
            let (filter, output) = (step.inputs[1].1, step.outputs[0].1);
            let patch = filter[1] * filter[2] * filter[3];
            let same_size = step.inputs[0].1[1..3] == output[1..3];
            // 1 × 1이고 크기가 그대로이면 펼치지 않는다.
            (patch != filter[3] || !same_size).then_some(output[1] * output[2] * patch)
        })
        .max()
        .unwrap_or(0);
    println!(
        "  {:<16} {:>8.2} MB",
        "펼친 행렬 (최대)",
        megabytes(unfolded)
    );

    // 3. 한 장의 시간.
    let pool = threads::global();
    let available = pool.pool().threads();
    println!("사진 한 장");
    let mut counts = vec![1, 2, 4, 8, 12, available];
    counts.retain(|&count| count <= available);
    counts.dedup();
    for count in counts {
        pool.set(count, count);
        let seconds = median(quick, || {
            interpreter.input_mut(0).copy_from_slice(&input);
            interpreter.run().unwrap();
            black_box(interpreter.output(0)[0]);
        });
        println!("  스레드 {count:>2}개        {:>9.2} ms", seconds * 1e3);
    }
    pool.set(available, available);

    // 4. 어느 연산에 쓰이는가.
    const RUNS: usize = 30;
    let mut times = vec![Duration::ZERO; interpreter.step_count()];
    for _ in 0..RUNS {
        interpreter.input_mut(0).copy_from_slice(&input);
        interpreter.run_timed(&mut times)?;
    }
    let mut by_kind: BTreeMap<String, (usize, f64)> = BTreeMap::new();
    for (step, time) in interpreter.steps().zip(&times) {
        let entry = by_kind.entry(kind(step.name, &step.inputs)).or_default();
        entry.0 += 1;
        entry.1 += time.as_secs_f64() / RUNS as f64;
    }
    let total: f64 = by_kind.values().map(|(_, seconds)| seconds).sum();
    println!("연산별 시간 (스레드 {available}개)");
    for (kind, (count, seconds)) in &by_kind {
        println!(
            "  {kind:<20} {count:>3}개 {:>8.2} ms {:>5.1}%",
            seconds * 1e3,
            100.0 * seconds / total
        );
    }

    // 5. 합성곱을 계산하는 세 방법. 정의를 그대로 옮긴 것은 느려서 적게 잰다.
    println!("합성곱의 방법 (스레드 {available}개)");
    let slow = Plan {
        warmup: 0,
        repetitions: 3,
    };
    for (label, factory, plan) in [
        (
            "정의 그대로",
            conv::conv_2d_direct as badak_graph::registry::Factory,
            slow,
        ),
        ("언제나 펼치기", conv::conv_2d_im2col, quick),
        ("1×1은 그대로", conv::conv_2d, quick),
    ] {
        let mut registry = Registry::builtin();
        registry.register("CONV_2D", factory);
        let mut interpreter = Interpreter::new(&graph, &registry)?;
        let seconds = median(plan, || {
            interpreter.input_mut(0).copy_from_slice(&input);
            interpreter.run().unwrap();
            black_box(interpreter.output(0)[0]);
        });
        println!("  {label:<16} {:>9.2} ms", seconds * 1e3);
    }
    Ok(())
}
