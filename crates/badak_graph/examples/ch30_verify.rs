//! 30장: ONNX에서 옮겨 최적화한 그래프의 점수를 ONNX Runtime의 점수와 견준다.
//!
//! 실행: `cargo run --release -p badak_graph --example ch30_verify`
//!
//! 입력은 27장의 정답 파일에 든 것(Python이 전처리한 사진)을 그대로 쓴다.

use badak_formats::onnx::Model;
use badak_formats::safetensors::SafeTensors;
use badak_graph::{Graph, Interpreter, Registry, Strategy, models_dir, optimize};

const IMAGES: [&str; 5] = ["cup", "golden_retriever", "guitar", "pelican", "tabby_cat"];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let inputs = SafeTensors::open(models_dir().join("vision_tflite/reference.safetensors"))?;
    let labels = std::fs::read_to_string(models_dir().join("vision_tflite/labels.txt"))?;
    let labels: Vec<&str> = labels.lines().collect();
    let directory = models_dir().join("vision_onnx");
    let reference = SafeTensors::open(directory.join("reference.safetensors"))?;
    let registry = Registry::builtin();

    for name in ["resnet50", "mobilenetv2"] {
        let model = Model::open(directory.join(format!("{name}.onnx")))?;
        let mut graph = Graph::from_onnx(&model)?;
        graph.infer_shapes(&registry)?;
        let imported = graph.nodes.len();
        let mut plain = Interpreter::new(&graph, &registry)?;
        optimize(&mut graph, &registry)?;
        let mut tuned = Interpreter::in_place(&graph, &registry, &[], Strategy::BySize)?;
        println!(
            "{name}.onnx: 파일의 노드 {}개, 옮기면 {imported}개, 최적화하면 {}개",
            model.graph.nodes.len(),
            tuned.step_count()
        );
        println!(
            "  {:<18} {:<28} {:>14} {:>14}",
            "사진", "고른 이름", "옮긴 그대로", "최적화한 것"
        );
        for image in IMAGES {
            let input = inputs.tensor(&format!("{image}.input"))?.to_vec::<f32>()?;
            let expected = reference
                .tensor(&format!("{name}.{image}.onnxruntime"))?
                .to_vec::<f32>()?;
            let mut gaps = [0.0_f32; 2];
            let mut best = 0;
            for (interpreter, gap) in [&mut plain, &mut tuned].into_iter().zip(&mut gaps) {
                interpreter.input_mut(0).copy_from_slice(&input);
                interpreter.run()?;
                let scores = interpreter.output(0);
                *gap = scores
                    .iter()
                    .zip(&expected)
                    .map(|(ours, theirs)| (ours - theirs).abs())
                    .fold(0.0, f32::max);
                best = (0..scores.len())
                    .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
                    .expect("점수가 천 개 있다");
            }
            println!(
                "  {image:<18} {:<28} {:>14.2e} {:>14.2e}",
                labels[best], gaps[0], gaps[1]
            );
            if gaps.iter().any(|&gap| gap.is_nan() || gap > 2e-4) {
                return Err(format!("{name} {image}: ONNX Runtime과 다릅니다").into());
            }
        }
    }
    Ok(())
}
