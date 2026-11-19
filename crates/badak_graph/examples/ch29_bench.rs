//! 29장: 정수 모델과 실수 모델을 견준다. 답, 크기, 메모리, 시간.
//!
//! 실행: `cargo run --release -p badak_graph --example ch29_bench`

use std::collections::BTreeMap;
use std::hint::black_box;
use std::time::Duration;

use badak_bench::stats::Summary;
use badak_bench::timing::{Plan, measure};
use badak_core::threads;
use badak_formats::safetensors::SafeTensors;
use badak_formats::tflite::Model;
use badak_graph::quantized::QuantizedInterpreter;
use badak_graph::{Graph, Interpreter, Registry, models_dir};

const IMAGES: [&str; 5] = ["cup", "golden_retriever", "guitar", "pelican", "tabby_cat"];

fn median(plan: Plan, work: impl FnMut()) -> f64 {
    Summary::new(&measure(plan, work)).median
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let directory = models_dir().join("vision_tflite");
    let labels = std::fs::read_to_string(directory.join("labels.txt"))?;
    let labels: Vec<&str> = labels.lines().collect();
    let floats = SafeTensors::open(directory.join("reference.safetensors"))?;
    let float_model = Model::open(directory.join("mobilenetv2.tflite"))?;
    let int_model = Model::open(directory.join("mobilenetv2_int8.tflite"))?;
    let mut float = Interpreter::new(&Graph::from_tflite(&float_model)?, &Registry::builtin())?;
    let mut int = QuantizedInterpreter::new(&Graph::from_tflite(&int_model)?)?;
    let (in_scale, in_zero) = int.input_quantization(0);
    let (out_scale, out_zero) = int.output_quantization(0);
    badak_bench::environment::print();

    println!("\n크기");
    println!("  {:<24} {:>12} {:>12}", "", "f32", "int8");
    println!(
        "  {:<24} {:>10.2}MB {:>10.2}MB",
        "모델 파일",
        float_model.file_len() as f64 / 1e6,
        int_model.file_len() as f64 / 1e6
    );
    println!(
        "  {:<24} {:>10.2}MB {:>10.2}MB",
        "중간 결과의 자리",
        float.arena_len() as f64 * 4.0 / 1e6,
        int.arena_len() as f64 / 1e6
    );

    // 답. 같은 사진을 두 모델에 넣고 점수(실수로 되돌린 것)를 견준다.
    println!("\n답 (f32 모델 → int8 모델)");
    for image in IMAGES {
        let input = floats.tensor(&format!("{image}.input"))?.to_vec::<f32>()?;
        float.input_mut(0).copy_from_slice(&input);
        float.run()?;
        for (target, &value) in int.input_mut(0).iter_mut().zip(&input) {
            // 실수를 정수로: 눈금으로 나누고 영점을 더한다.
            *target = ((value / in_scale).round() as i32 + in_zero).clamp(-128, 127) as i8;
        }
        int.run()?;
        let restored: Vec<f32> = int
            .output(0)
            .iter()
            .map(|&q| out_scale * (i32::from(q) - out_zero) as f32)
            .collect();
        let top = |scores: &[f32]| {
            let probabilities = badak_core::prob::softmax(scores);
            let best = (0..scores.len())
                .fold(0, |best, i| if scores[i] > scores[best] { i } else { best });
            (best, 100.0 * probabilities[best])
        };
        let (a, b) = (top(float.output(0)), top(&restored));
        let gap = float
            .output(0)
            .iter()
            .zip(&restored)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max);
        println!(
            "  {image:<18} {:<22} {:>5.1}% → {:<22} {:>5.1}%  점수의 차이 {gap:.2}",
            labels[a.0], a.1, labels[b.0], b.1
        );
    }

    // 시간.
    let pool = threads::global();
    let available = pool.pool().threads();
    let plan = Plan {
        warmup: 3,
        repetitions: 31,
    };
    let float_input = floats.tensor("cup.input")?.to_vec::<f32>()?;
    let int_input: Vec<i8> = float_input
        .iter()
        .map(|&value| ((value / in_scale).round() as i32 + in_zero).clamp(-128, 127) as i8)
        .collect();
    println!("\n사진 한 장 (ms)");
    println!("  {:<12} {:>10} {:>10}", "스레드", "f32", "int8");
    for count in [1, 2, 4, 8, 12] {
        if count > available {
            continue;
        }
        pool.set(count, count);
        let a = median(plan, || {
            float.input_mut(0).copy_from_slice(&float_input);
            float.run().unwrap();
            black_box(float.output(0)[0]);
        });
        let b = median(plan, || {
            int.input_mut(0).copy_from_slice(&int_input);
            int.run().unwrap();
            black_box(int.output(0)[0]);
        });
        println!("  {count:<12} {:>10.2} {:>10.2}", a * 1e3, b * 1e3);
    }

    // 정수 모델의 시간이 어느 연산에 쓰이는가. 스레드 하나로 본다.
    pool.set(1, 1);
    const RUNS: usize = 20;
    let mut times = vec![Duration::ZERO; int.step_count()];
    for _ in 0..RUNS {
        int.input_mut(0).copy_from_slice(&int_input);
        int.run_timed(&mut times)?;
    }
    let names: Vec<String> = int.step_names().map(str::to_string).collect();
    let sizes = int.step_sizes();
    let mut by_kind: BTreeMap<&str, (usize, f64, usize)> = BTreeMap::new();
    for ((name, time), size) in names.iter().zip(&times).zip(sizes) {
        let entry = by_kind.entry(name).or_default();
        entry.0 += 1;
        entry.1 += time.as_secs_f64() / RUNS as f64;
        entry.2 += size;
    }
    let total: f64 = by_kind.values().map(|(_, seconds, _)| seconds).sum();
    println!("\nint8 모델의 연산별 시간 (스레드 1개)");
    println!(
        "  {:<20} {:>4} {:>11} {:>6} {:>12} {:>14}",
        "연산", "개", "시간", "비율", "출력 값", "값 하나에"
    );
    for (kind, (count, seconds, values)) in &by_kind {
        println!(
            "  {kind:<20} {count:>3}개 {:>8.2} ms {:>5.1}% {values:>12} {:>11.2} ns",
            seconds * 1e3,
            100.0 * seconds / total,
            seconds / *values as f64 * 1e9
        );
    }
    pool.set(available, available);
    Ok(())
}
