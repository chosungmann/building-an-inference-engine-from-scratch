//! YOLOX-Tiny로 32장의 연산과 패스를 검증한다.
//!
//! 파일이 없으면 건너뛴다. `sh scripts/download_ch32.sh`로 받는다.

use badak_formats::onnx::Model;
use badak_formats::safetensors::SafeTensors;
use badak_graph::detect::{COCO, decode_yolox, non_max_suppression};
use badak_graph::{Graph, Interpreter, Registry, Strategy, models_dir, optimize};

fn count(graph: &Graph, name: &str) -> usize {
    graph
        .nodes
        .iter()
        .filter(|node| node.op.name() == name)
        .count()
}

#[test]
fn optimized_yolox_matches_onnxruntime() {
    let directory = models_dir().join("vision_transformer");
    if !directory.join("yolox_tiny.onnx").exists() {
        eprintln!("건너뜀: {}에 모델이 없습니다", directory.display());
        return;
    }
    let reference = SafeTensors::open(directory.join("reference.safetensors")).unwrap();
    let registry = Registry::builtin();
    let model = Model::open(directory.join("yolox_tiny.onnx")).unwrap();
    assert_eq!((model.opset, model.graph.nodes.len()), (11, 279));
    let mut graph = Graph::from_onnx(&model).unwrap();
    // 옮기면서 형상까지 정해진다.
    assert_eq!(graph.values[graph.outputs[0]].shape, [1, 3549, 85]);
    assert_eq!(graph.nodes.len(), 612);
    let mut plain = Interpreter::new(&graph, &registry).unwrap();

    optimize(&mut graph, &registry).unwrap();
    assert_eq!(graph.nodes.len(), 129);
    // 입력을 NHWC로 바꾸는 하나만 남는다. 끝의 TRANSPOSE까지 사라진다.
    assert_eq!(count(&graph, "TRANSPOSE"), 1);
    // SiLU는 모두 합성곱에 붙고, 남는 시그모이드는 출력의 것 여섯이다.
    assert_eq!((count(&graph, "MUL"), count(&graph, "SIGMOID")), (0, 6));
    assert_eq!(count(&graph, "SLICE"), 6);
    let mut tuned = Interpreter::in_place(&graph, &registry, &[], Strategy::BySize).unwrap();
    assert!(tuned.arena_len() < plain.arena_len());

    // (사진, 찾아야 하는 물체들)
    let cases = [
        ("cup", vec!["cup"]),
        ("guitar", vec![]),
        ("tabby_cat", vec!["cat", "chair"]),
    ];
    for (image, wanted) in cases {
        let tensor = |key: &str| {
            let key = format!("yolox.{image}.{key}");
            reference.tensor(&key).unwrap().to_vec::<f32>().unwrap()
        };
        let (input, expected) = (tensor("input"), tensor("output"));
        for interpreter in [&mut plain, &mut tuned] {
            interpreter.scribble(f32::NAN);
            interpreter.input_mut(0).copy_from_slice(&input);
            interpreter.run().unwrap();
            let output = interpreter.output(0);
            for (ours, theirs) in output.iter().zip(&expected) {
                assert!((ours - theirs).abs() < 2e-3, "{image}: {ours} 대 {theirs}");
            }
            let candidates = decode_yolox(output, 416, &[8, 16, 32], COCO.len(), 0.3);
            let found = non_max_suppression(candidates, 0.45);
            let names: Vec<&str> = found.iter().map(|d| COCO[d.class]).collect();
            assert_eq!(names, wanted, "{image}");
        }
    }
}
