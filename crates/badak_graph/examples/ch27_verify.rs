//! 27장: 합성곱 모델의 점수를 PyTorch와 LiteRT의 점수와 견준다.
//!
//! 실행: `cargo run --release -p badak_graph --example ch27_verify`
//!
//! 입력은 정답 파일에 든 것(Python이 전처리한 사진)을 그대로 쓴다. 전처리의
//! 차이를 빼고 모델의 계산만 견주기 위해서다.

use std::time::Instant;

use badak_formats::safetensors::SafeTensors;
use badak_formats::tflite::Model;
use badak_graph::{Graph, Interpreter, Registry, models_dir};

const IMAGES: [&str; 5] = ["cup", "golden_retriever", "guitar", "pelican", "tabby_cat"];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let directory = models_dir().join("vision_tflite");
    let reference = SafeTensors::open(directory.join("reference.safetensors"))?;
    let labels = std::fs::read_to_string(directory.join("labels.txt"))?;
    let labels: Vec<&str> = labels.lines().collect();
    let registry = Registry::builtin();

    for name in ["resnet50", "mobilenetv2"] {
        let model = Model::open(directory.join(format!("{name}.tflite")))?;
        let graph = Graph::from_tflite(&model)?;
        let mut interpreter = Interpreter::new(&graph, &registry)?;
        println!(
            "{name}.tflite: 노드 {}개, 중간 결과의 자리 {:.1}MB",
            interpreter.step_count(),
            interpreter.arena_len() as f64 * 4.0 / 1e6
        );
        println!(
            "  {:<18} {:<28} {:>10} {:>10} {:>9}",
            "사진", "고른 이름", "LiteRT와", "PyTorch와", "시간"
        );
        for image in IMAGES {
            let input = reference
                .tensor(&format!("{image}.input"))?
                .to_vec::<f32>()?;
            interpreter.input_mut(0).copy_from_slice(&input);
            let start = Instant::now();
            interpreter.run()?;
            let elapsed = start.elapsed();
            let scores = interpreter.output(0);
            let gap = |other: &str| -> Result<f32, Box<dyn std::error::Error>> {
                let expected = reference
                    .tensor(&format!("{name}.{image}.{other}"))?
                    .to_vec::<f32>()?;
                Ok(scores
                    .iter()
                    .zip(&expected)
                    .map(|(ours, theirs)| (ours - theirs).abs())
                    .fold(0.0, f32::max))
            };
            let best = (0..scores.len())
                .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
                .expect("점수가 천 개 있다");
            println!(
                "  {image:<18} {:<28} {:>10.2e} {:>10.2e} {:>7.1}ms",
                labels[best],
                gap("litert")?,
                gap("pytorch")?,
                elapsed.as_secs_f64() * 1e3
            );
        }
    }
    Ok(())
}
