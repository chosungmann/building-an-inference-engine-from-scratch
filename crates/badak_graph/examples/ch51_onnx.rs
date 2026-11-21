//! 51장: PyTorch가 내보낸 ONNX 파일을 badak으로 읽어, PyTorch의 답과 견준다.
//!
//! 실행:
//!     python reference/ch51_export.py onnx
//!     cargo run --release -p badak_graph --example ch51_onnx
//!
//! `models/ch51/mlp.onnx`는 7장의 신경망을 `torch.onnx.export`로 내보낸
//! 것이고, `mlp_reference.safetensors`에는 입력 열여섯 장과 PyTorch가 낸
//! 답이 들어 있다.

use std::time::Instant;

use badak_formats::onnx;
use badak_formats::safetensors::SafeTensors;
use badak_graph::{Graph, Interpreter, Registry, models_dir};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let directory = models_dir().join("ch51");
    let model = onnx::Model::open(directory.join("mlp.onnx"))?;
    let registry = Registry::builtin();
    // ANCHOR: run
    let mut graph = Graph::from_onnx(&model)?;
    // 파일은 묶음의 크기를 정하지 않았다. 한 장씩 넣기로 한다.
    let input = graph.inputs[0];
    graph.values[input].shape = vec![1, 784];
    graph.infer_shapes(&registry)?;
    let mut interpreter = Interpreter::new(&graph, &registry)?;
    // ANCHOR_END: run
    let names: Vec<&str> = graph.nodes.iter().map(|node| node.op.name()).collect();
    println!("badak 노드 {names:?}");

    let reference = SafeTensors::open(directory.join("mlp_reference.safetensors"))?;
    let (x, y) = (
        reference.tensor("x")?.to_vec::<f32>()?,
        reference.tensor("y")?.to_vec::<f32>()?,
    );
    let mut worst = 0.0_f32;
    let mut answers = Vec::new();
    let (images, _) = x.as_chunks::<784>();
    let (answers_expected, _) = y.as_chunks::<10>();
    for (image, expected) in images.iter().zip(answers_expected) {
        interpreter.input_mut(0).copy_from_slice(image);
        interpreter.run()?;
        let scores = interpreter.output(0);
        for (got, want) in scores.iter().zip(expected) {
            worst = worst.max((got - want).abs());
        }
        answers.push(
            (0..10)
                .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
                .unwrap_or(0),
        );
    }
    println!("badak 답 {answers:?}");
    println!("badak PyTorch와의 가장 큰 차이 {worst:.2e}");

    let started = Instant::now();
    let rounds = 20_000;
    for _ in 0..rounds {
        interpreter.input_mut(0).copy_from_slice(&x[..784]);
        interpreter.run()?;
    }
    println!(
        "badak 한 장에 {:.1}us",
        started.elapsed().as_secs_f64() * 1e6 / f64::from(rounds)
    );
    Ok(())
}
