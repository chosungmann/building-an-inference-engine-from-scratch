//! 32장: 사진 파일에서 상자까지. YOLOX-Tiny로 물체를 찾는다.
//!
//! 실행: `cargo run --release -p badak_image --example ch32_detect`
//!
//! 사진마다 세 가지를 본다. 줄이고 채운 입력이 Python이 만든 것과 같은가,
//! 모델의 출력이 ONNX Runtime과 같은가, 그리고 찾아낸 상자가 무엇인가.

use std::time::Instant;

use badak_formats::onnx::Model;
use badak_formats::safetensors::SafeTensors;
use badak_graph::detect::{COCO, Detection, decode_yolox, non_max_suppression};
use badak_graph::{Graph, Interpreter, Registry, Strategy, models_dir, optimize};
use badak_image::Image;

const IMAGES: [&str; 5] = ["cup", "golden_retriever", "guitar", "pelican", "tabby_cat"];
/// 모델의 입력의 한 변.
const SIDE: usize = 416;
/// 격자마다의 칸의 크기(픽셀).
const STRIDES: [usize; 3] = [8, 16, 32];
/// 이 점수를 넘는 상자만 본다.
const SCORE: f32 = 0.3;
/// 같은 종류의 두 상자가 이보다 많이 겹치면 하나를 버린다.
const OVERLAP: f32 = 0.45;

fn detect(output: &[f32]) -> (usize, Vec<Detection>) {
    let candidates = decode_yolox(output, SIDE, &STRIDES, COCO.len(), SCORE);
    (candidates.len(), non_max_suppression(candidates, OVERLAP))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let models = models_dir();
    let directory = models.join("vision_transformer");
    let reference = SafeTensors::open(directory.join("reference.safetensors"))?;
    let registry = Registry::builtin();
    let mut graph = Graph::from_onnx(&Model::open(directory.join("yolox_tiny.onnx"))?)?;
    optimize(&mut graph, &registry)?;
    let mut interpreter = Interpreter::in_place(&graph, &registry, &[], Strategy::BySize)?;

    for name in IMAGES {
        let start = Instant::now();
        let image = Image::open(models.join(format!("images/{name}.jpg")))?;
        let decoded = start.elapsed();
        let start = Instant::now();
        let (input, ratio) = image.letterbox(SIDE, 114);
        let shaped = start.elapsed();
        let theirs = reference
            .tensor(&format!("yolox.{name}.input"))?
            .to_vec::<f32>()?;
        let different = input.iter().zip(&theirs).filter(|(a, b)| a != b).count();

        interpreter.input_mut(0).copy_from_slice(&input);
        let start = Instant::now();
        interpreter.run()?;
        let ran = start.elapsed();
        let start = Instant::now();
        let (candidates, found) = detect(interpreter.output(0));
        let decoded_boxes = start.elapsed();

        // 같은 해독과 억제를 ONNX Runtime의 출력에 해 본다.
        let expected = reference
            .tensor(&format!("yolox.{name}.output"))?
            .to_vec::<f32>()?;
        let (_, wanted) = detect(&expected);
        println!(
            "{name}.jpg ({} × {}, 줄인 비율 {ratio:.4})",
            image.width, image.height
        );
        println!("  입력: Python과 다른 값 {different}개 / {}개", input.len());
        println!(
            "  풀기 {:.1}ms, 줄이고 채우기 {:.1}ms, 모델 {:.1}ms, 해독과 억제 {:.2}ms",
            decoded.as_secs_f64() * 1e3,
            shaped.as_secs_f64() * 1e3,
            ran.as_secs_f64() * 1e3,
            decoded_boxes.as_secs_f64() * 1e3
        );
        println!(
            "  문턱을 넘은 상자 {candidates}개 → 억제 뒤 {}개 (ONNX Runtime의 출력으로는 {}개)",
            found.len(),
            wanted.len()
        );
        for (detection, other) in found.iter().zip(&wanted) {
            // 상자를 원래 사진의 좌표로 되돌린다.
            let [left, top, right, bottom] = detection.bounds.map(|value| value / ratio);
            let drift = detection
                .bounds
                .iter()
                .zip(&other.bounds)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            println!(
                "    {:<14} {:>5.1}%  ({left:>6.0}, {top:>6.0}) – ({right:>6.0}, {bottom:>6.0})  ONNX Runtime의 상자와 {drift:.3}픽셀",
                COCO[detection.class],
                100.0 * detection.score
            );
            if detection.class != other.class {
                return Err(format!("{name}: 종류가 다릅니다").into());
            }
        }
        if found.len() != wanted.len() {
            return Err(format!("{name}: 상자의 수가 다릅니다").into());
        }
    }
    Ok(())
}
