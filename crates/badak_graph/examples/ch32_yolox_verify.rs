//! 32장: YOLOX-Tiny의 출력을 ONNX Runtime의 출력과 견준다.
//!
//! 실행: `cargo run --release -p badak_graph --example ch32_yolox_verify`
//!
//! 입력은 정답 파일에 든 것(Python이 줄이고 채운 사진)을 그대로 쓴다.

use std::collections::BTreeMap;
use std::time::Instant;

use badak_formats::onnx::Model;
use badak_formats::safetensors::SafeTensors;
use badak_graph::{Graph, Interpreter, Registry, Strategy, models_dir, optimize};

const IMAGES: [&str; 5] = ["cup", "golden_retriever", "guitar", "pelican", "tabby_cat"];

fn counts(graph: &Graph) -> String {
    let mut counts = BTreeMap::new();
    for node in &graph.nodes {
        *counts.entry(node.op.name()).or_insert(0) += 1;
    }
    let parts: Vec<String> = counts
        .iter()
        .map(|(name, count)| format!("{name} {count}"))
        .collect();
    parts.join(", ")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let directory = models_dir().join("vision_transformer");
    let reference = SafeTensors::open(directory.join("reference.safetensors"))?;
    let registry = Registry::builtin();
    let model = Model::open(directory.join("yolox_tiny.onnx"))?;
    let mut graph = Graph::from_onnx(&model)?;
    graph.infer_shapes(&registry)?;
    println!(
        "yolox_tiny.onnx: 파일의 노드 {}개 (연산 버전 {}, {})",
        model.graph.nodes.len(),
        model.opset,
        model.producer
    );
    println!("  옮기면 {}개: {}", graph.nodes.len(), counts(&graph));
    let mut plain = Interpreter::new(&graph, &registry)?;
    let applied = optimize(&mut graph, &registry)?;
    println!("  최적화하면 {}개: {}", graph.nodes.len(), counts(&graph));
    for pass in applied.iter().filter(|pass| pass.changes > 0) {
        println!("    {:<26} {:>4}", pass.name, pass.changes);
    }
    let mut tuned = Interpreter::in_place(&graph, &registry, &[], Strategy::BySize)?;
    println!(
        "  입력 {:?}, 출력 {:?}, 중간 결과의 자리 {:.1}MB → {:.1}MB",
        tuned.input_shape(0),
        tuned.output_shape(0),
        plain.arena_len() as f64 * 4.0 / 1e6,
        tuned.arena_len() as f64 * 4.0 / 1e6
    );
    println!(
        "  {:<18} {:>14} {:>14} {:>10} {:>10}",
        "사진", "옮긴 그대로", "최적화한 것", "시간", "시간"
    );
    for image in IMAGES {
        let input = reference
            .tensor(&format!("yolox.{image}.input"))?
            .to_vec::<f32>()?;
        let expected = reference
            .tensor(&format!("yolox.{image}.output"))?
            .to_vec::<f32>()?;
        let mut gaps = [0.0_f32; 2];
        let mut times = [0.0_f64; 2];
        for (index, interpreter) in [&mut plain, &mut tuned].into_iter().enumerate() {
            interpreter.input_mut(0).copy_from_slice(&input);
            let start = Instant::now();
            interpreter.run()?;
            times[index] = start.elapsed().as_secs_f64() * 1e3;
            gaps[index] = interpreter
                .output(0)
                .iter()
                .zip(&expected)
                .map(|(ours, theirs)| (ours - theirs).abs())
                .fold(0.0, f32::max);
        }
        println!(
            "  {image:<18} {:>14.2e} {:>14.2e} {:>8.1}ms {:>8.1}ms",
            gaps[0], gaps[1], times[0], times[1]
        );
        if gaps.iter().any(|&gap| gap.is_nan() || gap > 5e-3) {
            return Err(format!("{image}: ONNX Runtime과 다릅니다").into());
        }
    }
    Ok(())
}
