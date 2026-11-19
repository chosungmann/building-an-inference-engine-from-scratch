//! 30장: ONNX에서 옮긴 그래프를 최적화하고, 무엇이 얼마나 바뀌었는지 본다.
//!
//! 실행: `cargo run --release -p badak_graph --example ch30_optimize`

use std::collections::BTreeMap;
use std::time::Instant;

use badak_formats::safetensors::SafeTensors;
use badak_formats::{onnx, tflite};
use badak_graph::{Graph, Interpreter, Registry, Strategy, models_dir, optimize};

const IMAGES: [&str; 5] = ["cup", "golden_retriever", "guitar", "pelican", "tabby_cat"];

type Outcome<T> = Result<T, Box<dyn std::error::Error>>;

fn counts(graph: &Graph) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for node in &graph.nodes {
        *counts.entry(node.op.name().to_string()).or_insert(0) += 1;
    }
    counts
}

/// 그래프에 든 상수의 원소 수.
fn constants(graph: &Graph) -> usize {
    graph
        .values
        .iter()
        .filter_map(|value| value.constant.as_ref())
        .map(|tensor| tensor.len())
        .sum()
}

/// 사진 다섯 장을 넣어 점수를 받는다.
fn scores(graph: &Graph, registry: &Registry, inputs: &SafeTensors) -> Outcome<Vec<Vec<f32>>> {
    let mut interpreter = Interpreter::new(graph, registry)?;
    let mut all = Vec::new();
    for image in IMAGES {
        let input = inputs.tensor(&format!("{image}.input"))?.to_vec::<f32>()?;
        interpreter.input_mut(0).copy_from_slice(&input);
        interpreter.run()?;
        all.push(interpreter.output(0).to_vec());
    }
    Ok(all)
}

fn gap(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    a.iter()
        .flatten()
        .zip(b.iter().flatten())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn megabytes(elements: usize) -> f64 {
    elements as f64 * 4.0 / 1e6
}

fn compare(before: &Graph, after: &Graph) {
    let (old, new) = (counts(before), counts(after));
    let names: Vec<&String> = old.keys().chain(new.keys()).collect();
    let mut names: Vec<&String> = names;
    names.sort();
    names.dedup();
    println!("  {:<20} {:>8} {:>8}", "연산", "앞", "뒤");
    for name in names {
        let count = |counts: &BTreeMap<String, usize>| counts.get(name).copied().unwrap_or(0);
        println!("  {name:<20} {:>8} {:>8}", count(&old), count(&new));
    }
    println!(
        "  {:<20} {:>8} {:>8}",
        "모두",
        before.nodes.len(),
        after.nodes.len()
    );
    println!(
        "  {:<20} {:>8} {:>8}",
        "값",
        before.values.len(),
        after.values.len()
    );
    println!(
        "  {:<18} {:>8.2} {:>8.2}",
        "상수 (백만 원소)",
        constants(before) as f64 / 1e6,
        constants(after) as f64 / 1e6
    );
}

fn memory(before: &Graph, graph: &Graph, registry: &Registry) -> Outcome<()> {
    let original = Interpreter::new(before, registry)?;
    let plain = Interpreter::new(graph, registry)?;
    let shared = Interpreter::in_place(graph, registry, &[], Strategy::BySize)?;
    println!(
        "  중간 결과의 자리: 앞 {:.2}MB, 뒤 {:.2}MB, 제자리 계산을 켜면 {:.2}MB (단계 {}개 가운데 {}개가 제자리)",
        megabytes(original.arena_len()),
        megabytes(plain.arena_len()),
        megabytes(shared.arena_len()),
        shared.step_count(),
        shared.in_place_count()
    );
    Ok(())
}

fn main() -> Outcome<()> {
    let inputs = SafeTensors::open(models_dir().join("vision_tflite/reference.safetensors"))?;
    let directory = models_dir().join("vision_onnx");
    let reference = SafeTensors::open(directory.join("reference.safetensors"))?;
    let registry = Registry::builtin();

    for name in ["resnet50", "mobilenetv2"] {
        let start = Instant::now();
        let model = onnx::Model::open(directory.join(format!("{name}.onnx")))?;
        let opened = start.elapsed();
        let start = Instant::now();
        let mut graph = Graph::from_onnx(&model)?;
        graph.infer_shapes(&registry)?;
        let imported = start.elapsed();
        let before = graph.clone();
        let start = Instant::now();
        let applied = optimize(&mut graph, &registry)?;
        let optimized = start.elapsed();

        println!("{name}.onnx (마디 {}개)", model.graph.nodes.len());
        println!(
            "  파일 읽기 {:.1}ms, 옮기기 {:.1}ms, 최적화 {:.1}ms",
            opened.as_secs_f64() * 1e3,
            imported.as_secs_f64() * 1e3,
            optimized.as_secs_f64() * 1e3
        );
        compare(&before, &graph);
        println!("  패스마다 고친 횟수");
        for pass in &applied {
            println!("    {:<24} {:>4}", pass.name, pass.changes);
        }

        let expected: Vec<Vec<f32>> = IMAGES
            .iter()
            .map(|image| {
                let key = format!("{name}.{image}.onnxruntime");
                Ok(reference.tensor(&key)?.to_vec::<f32>()?)
            })
            .collect::<Outcome<_>>()?;
        let (plain, tuned) = (
            scores(&before, &registry, &inputs)?,
            scores(&graph, &registry, &inputs)?,
        );
        println!("  점수의 가장 큰 차이 (사진 다섯 장, 점수 5,000개)");
        println!(
            "    옮긴 그대로 대 ONNX Runtime   {:.2e}",
            gap(&plain, &expected)
        );
        println!(
            "    최적화한 것 대 ONNX Runtime   {:.2e}",
            gap(&tuned, &expected)
        );
        println!(
            "    옮긴 그대로 대 최적화한 것    {:.2e}",
            gap(&plain, &tuned)
        );
        memory(&before, &graph, &registry)?;
        println!();
    }

    // 변환기가 이미 다듬어 준 .tflite에도 남은 일이 있다.
    let directory = models_dir().join("vision_tflite");
    for name in ["resnet50", "mobilenetv2"] {
        let model = tflite::Model::open(directory.join(format!("{name}.tflite")))?;
        let before = Graph::from_tflite(&model)?;
        let mut graph = before.clone();
        let applied = optimize(&mut graph, &registry)?;
        println!("{name}.tflite");
        compare(&before, &graph);
        let done: Vec<String> = applied
            .iter()
            .filter(|pass| pass.changes > 0)
            .map(|pass| format!("{} {}", pass.name, pass.changes))
            .collect();
        println!(
            "  고친 것: {}",
            if done.is_empty() {
                "없음".into()
            } else {
                done.join(", ")
            }
        );
        let (plain, tuned) = (
            scores(&before, &registry, &inputs)?,
            scores(&graph, &registry, &inputs)?,
        );
        println!("  점수의 가장 큰 차이: {:.2e}", gap(&plain, &tuned));
        memory(&before, &graph, &registry)?;
        println!();
    }
    Ok(())
}
