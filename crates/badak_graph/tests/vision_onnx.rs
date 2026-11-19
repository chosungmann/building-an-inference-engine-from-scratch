//! ONNX에서 옮긴 합성곱 모델로 그래프 최적화 전체를 검증한다 (30장).
//!
//! 파일이 없으면 건너뛴다. `sh scripts/download_ch27.sh`와
//! `sh scripts/download_ch30.sh`로 받는다.

use badak_formats::safetensors::SafeTensors;
use badak_formats::{onnx, tflite};
use badak_graph::{Graph, Interpreter, Registry, Strategy, models_dir, optimize};

const IMAGES: [&str; 5] = ["cup", "golden_retriever", "guitar", "pelican", "tabby_cat"];

fn run(interpreter: &mut Interpreter, input: &[f32]) -> Vec<f32> {
    // 앞의 실행이 남긴 값에 기대면 여기서 드러난다.
    interpreter.scribble(f32::NAN);
    interpreter.input_mut(0).copy_from_slice(input);
    interpreter.run().unwrap();
    interpreter.output(0).to_vec()
}

fn largest_gap(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn count(graph: &Graph, name: &str) -> usize {
    graph
        .nodes
        .iter()
        .filter(|node| node.op.name() == name)
        .count()
}

#[test]
fn optimized_onnx_graphs_match_onnxruntime() {
    let directory = models_dir().join("vision_onnx");
    if !directory.join("resnet50.onnx").exists() {
        eprintln!("건너뜀: {}에 모델이 없습니다", directory.display());
        return;
    }
    let inputs =
        SafeTensors::open(models_dir().join("vision_tflite/reference.safetensors")).unwrap();
    let reference = SafeTensors::open(directory.join("reference.safetensors")).unwrap();
    let registry = Registry::builtin();
    for (name, imported, optimized) in [("resnet50", 448, 73), ("mobilenetv2", 414, 65)] {
        let model = onnx::Model::open(directory.join(format!("{name}.onnx"))).unwrap();
        assert_eq!(model.opset, 18);
        let mut graph = Graph::from_onnx(&model).unwrap();
        graph.infer_shapes(&registry).unwrap();
        assert_eq!(graph.nodes.len(), imported);
        let mut plain = Interpreter::new(&graph, &registry).unwrap();

        optimize(&mut graph, &registry).unwrap();
        assert_eq!(graph.nodes.len(), optimized);
        // 남는 TRANSPOSE는 입력을 NHWC로 바꾸는 하나뿐이다.
        assert_eq!(count(&graph, "TRANSPOSE"), 1);
        for gone in ["BATCH_NORM", "RELU", "CLIP", "IDENTITY", "FLATTEN"] {
            assert_eq!(count(&graph, gone), 0, "{name}: {gone}");
        }
        let mut tuned = Interpreter::new(&graph, &registry).unwrap();
        let mut shared = Interpreter::in_place(&graph, &registry, &[], Strategy::BySize).unwrap();
        assert!(shared.in_place_count() >= 10);
        assert!(shared.arena_len() <= tuned.arena_len());
        assert!(tuned.arena_len() <= plain.arena_len());

        for image in IMAGES {
            let input = inputs.tensor(&format!("{image}.input")).unwrap();
            let input = input.to_vec::<f32>().unwrap();
            let expected = reference
                .tensor(&format!("{name}.{image}.onnxruntime"))
                .unwrap()
                .to_vec::<f32>()
                .unwrap();
            let before = run(&mut plain, &input);
            let after = run(&mut tuned, &input);
            assert!(largest_gap(&before, &expected) < 2e-4, "{name} {image}");
            assert!(largest_gap(&after, &expected) < 2e-4, "{name} {image}");
            assert!(largest_gap(&before, &after) < 1e-4, "{name} {image}");
            // 제자리 계산은 계산의 순서를 바꾸지 않는다. 비트까지 같다.
            assert_eq!(run(&mut shared, &input), after, "{name} {image}");
        }
    }
}

#[test]
fn pads_in_converted_models_are_merged() {
    let directory = models_dir().join("vision_tflite");
    if !directory.join("resnet50.tflite").exists() {
        eprintln!("건너뜀: {}에 모델이 없습니다", directory.display());
        return;
    }
    let inputs = SafeTensors::open(directory.join("reference.safetensors")).unwrap();
    let registry = Registry::builtin();
    // ResNet-50의 PAD 하나는 풀링 앞에 있어 남는다.
    for (name, before, after, pads) in [("resnet50", 78, 74, 1), ("mobilenetv2", 70, 65, 0)] {
        let model = tflite::Model::open(directory.join(format!("{name}.tflite"))).unwrap();
        let original = Graph::from_tflite(&model).unwrap();
        let mut graph = original.clone();
        assert_eq!(graph.nodes.len(), before);
        let applied = optimize(&mut graph, &registry).unwrap();
        assert_eq!(graph.nodes.len(), after);
        assert_eq!(count(&graph, "PAD"), pads);
        // 변환기가 이미 한 일은 다시 할 것이 없다.
        for pass in applied {
            assert_eq!(pass.changes > 0, pass.name == "PAD 합치기", "{}", pass.name);
        }
        let mut plain = Interpreter::new(&original, &registry).unwrap();
        let mut tuned = Interpreter::in_place(&graph, &registry, &[], Strategy::BySize).unwrap();
        for image in IMAGES {
            let input = inputs.tensor(&format!("{image}.input")).unwrap();
            let input = input.to_vec::<f32>().unwrap();
            let gap = largest_gap(&run(&mut plain, &input), &run(&mut tuned, &input));
            assert!(gap < 1e-5, "{name} {image}: {gap}");
        }
    }
}

#[test]
fn quantized_model_is_not_touched() {
    let path = models_dir().join("vision_tflite/mobilenetv2_int8.tflite");
    if !path.exists() {
        eprintln!("건너뜀: {}이 없습니다", path.display());
        return;
    }
    let mut graph = Graph::from_tflite(&tflite::Model::open(path).unwrap()).unwrap();
    let before = graph.nodes.clone();
    let applied = optimize(&mut graph, &Registry::builtin()).unwrap();
    assert!(applied.iter().all(|pass| pass.changes == 0));
    assert_eq!(graph.nodes, before);
}
