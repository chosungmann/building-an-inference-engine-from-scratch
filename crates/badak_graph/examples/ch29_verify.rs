//! 29장: 정수 모델의 출력을 LiteRT의 출력과 비트 단위로 견준다.
//!
//! 실행: `cargo run --release -p badak_graph --example ch29_verify`
//!
//! LiteRT에는 길이 둘 있다. 자기 커널로 계산하는 길(builtin)과 XNNPACK에
//! 넘기는 길. 두 길의 출력이 서로 조금 다르다. 우리 엔진은 앞의 것과 같은
//! 정수를 내야 한다.

use badak_formats::safetensors::SafeTensors;
use badak_formats::tflite::Model;
use badak_graph::quantized::QuantizedInterpreter;
use badak_graph::{Graph, models_dir};

const IMAGES: [&str; 5] = ["cup", "golden_retriever", "guitar", "pelican", "tabby_cat"];

fn signed(bytes: Vec<u8>) -> Vec<i8> {
    bytes.into_iter().map(|byte| byte as i8).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let directory = models_dir().join("vision_tflite");
    let reference = SafeTensors::open(directory.join("reference_int8.safetensors"))?;
    let labels = std::fs::read_to_string(directory.join("labels.txt"))?;
    let labels: Vec<&str> = labels.lines().collect();
    let model = Model::open(directory.join("mobilenetv2_int8.tflite"))?;
    let graph = Graph::from_tflite(&model)?;
    let mut interpreter = QuantizedInterpreter::new(&graph)?;
    let (in_scale, in_zero) = interpreter.input_quantization(0);
    let (out_scale, out_zero) = interpreter.output_quantization(0);
    println!(
        "mobilenetv2_int8.tflite: {}바이트, 노드 {}개, 중간 결과의 자리 {:.2}MB",
        model.file_len(),
        interpreter.step_count(),
        interpreter.arena_len() as f64 / 1e6
    );
    println!("  입력: 실수 = {in_scale:.6} × (정수 − ({in_zero}))");
    println!("  출력: 실수 = {out_scale:.6} × (정수 − ({out_zero}))");
    println!(
        "  {:<18} {:<34} {:>14} {:>14}",
        "사진", "고른 이름", "builtin과", "XNNPACK과"
    );
    for image in IMAGES {
        let tensor = |name: &str| -> Result<Vec<i8>, Box<dyn std::error::Error>> {
            Ok(signed(
                reference
                    .tensor_as_stored(&format!("{image}.{name}"))?
                    .to_vec::<u8>()?,
            ))
        };
        interpreter.input_mut(0).copy_from_slice(&tensor("input")?);
        interpreter.run()?;
        let ours = interpreter.output(0);
        let gap = |theirs: &[i8]| -> (usize, i32) {
            let different = ours.iter().zip(theirs).filter(|(a, b)| a != b).count();
            let worst = ours
                .iter()
                .zip(theirs)
                .map(|(&a, &b)| (i32::from(a) - i32::from(b)).abs())
                .max()
                .unwrap_or(0);
            (different, worst)
        };
        let (builtin, delegated) = (gap(&tensor("builtin")?), gap(&tensor("xnnpack")?));
        // 점수가 같은 것이 여럿이면 앞의 것을 고른다. 정수 출력에서는 동점이
        // 드물지 않다.
        let best = (0..ours.len()).fold(0, |best, index| {
            if ours[index] > ours[best] {
                index
            } else {
                best
            }
        });
        let ties = ours.iter().filter(|&&score| score == ours[best]).count();
        let label = if ties > 1 {
            format!("{} (동점 {ties}개)", labels[best])
        } else {
            labels[best].to_string()
        };
        let describe = |(different, worst): (usize, i32)| {
            if different == 0 {
                "같은 비트".to_string()
            } else {
                format!("{different}개, 최대 {worst}")
            }
        };
        println!(
            "  {image:<18} {:<34} {:>14} {:>14}",
            label,
            describe(builtin),
            describe(delegated)
        );
    }
    Ok(())
}
