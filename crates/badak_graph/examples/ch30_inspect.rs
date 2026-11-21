//! 30장: ONNX 파일 안을 들여다본다.
//!
//! 실행: `cargo run --release -p badak_graph --example ch30_inspect [파일]`

use std::collections::BTreeMap;

use badak_formats::onnx::{Attribute, Model};
use badak_graph::models_dir;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).map_or_else(
        || models_dir().join("vision_onnx/mobilenetv2.onnx"),
        std::path::PathBuf::from,
    );
    let model = Model::open(&path)?;
    let graph = &model.graph;
    println!("{}", path.file_name().unwrap_or_default().to_string_lossy());
    println!("  크기           {:.1}MB", model.file_len() as f64 / 1e6);
    println!("  형식의 버전(IR)  {}", model.ir_version);
    println!("  연산의 버전      {}", model.opset);
    println!("  만든 프로그램  {}", model.producer);
    for (label, values) in [("입력", &graph.inputs), ("출력", &graph.outputs)] {
        for value in values {
            let shape: Vec<String> = value
                .shape
                .iter()
                .map(|dim| dim.map_or("?".into(), |dim| dim.to_string()))
                .collect();
            println!(
                "  {label}           {:?} ({})",
                value.name,
                shape.join(", ")
            );
        }
    }

    let elements: usize = graph.initializers.iter().map(|(_, t)| t.len()).sum();
    let borrowed = graph
        .initializers
        .iter()
        .filter(|(_, tensor)| tensor.is_borrowed())
        .count();
    println!(
        "  가중치         텐서 {}개, 원소 {:.2}백만 개 (파일을 빌려 쓰는 것 {}개)",
        graph.initializers.len(),
        elements as f64 / 1e6,
        borrowed
    );

    let mut counts = BTreeMap::new();
    for node in &graph.nodes {
        *counts.entry(node.op_type.as_str()).or_insert(0_usize) += 1;
    }
    println!("  노드           {}개", graph.nodes.len());
    for (op, count) in &counts {
        println!("    {op:<22} {count:>4}");
    }

    println!("\n처음 여섯 노드");
    for node in graph.nodes.iter().take(6) {
        println!("  {} ({})", node.op_type, node.name);
        println!("    입력 {:?}", node.inputs);
        println!("    출력 {:?}", node.outputs);
        for (name, value) in &node.attributes {
            let text = match value {
                Attribute::Int(value) => value.to_string(),
                Attribute::Float(value) => value.to_string(),
                Attribute::Ints(values) => format!("{values:?}"),
                Attribute::Floats(values) => format!("{values:?}"),
                Attribute::String(value) => format!("{value:?}"),
                Attribute::Tensor(tensor) => {
                    format!("텐서 {:?} {:?}", tensor.shape(), tensor.to_vec::<f32>()?)
                }
                Attribute::Other(kind) => format!("(종류 {kind})"),
            };
            println!("    {name} = {text}");
        }
    }
    Ok(())
}
