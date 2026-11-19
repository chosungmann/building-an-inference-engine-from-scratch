//! 29장: 층마다의 출력을 LiteRT의 것과 비트 단위로 견준다.
//!
//! 실행: `cargo run --release -p badak_graph --example ch29_layers`
//!
//! 정답 파일에는 첫 사진(`cup`)에 대해 LiteRT 자신의 커널이 계산한 중간
//! 결과가 모두 들어 있다. 단계마다 우리 출력과 다른 값이 몇 개인지 센다.

use badak_formats::safetensors::SafeTensors;
use badak_formats::tflite::Model;
use badak_graph::quantized::QuantizedInterpreter;
use badak_graph::{Graph, models_dir};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let directory = models_dir().join("vision_tflite");
    let reference = SafeTensors::open(directory.join("reference_int8.safetensors"))?;
    let model = Model::open(directory.join("mobilenetv2_int8.tflite"))?;
    let graph = Graph::from_tflite(&model)?;
    let mut interpreter = QuantizedInterpreter::new(&graph)?;
    let input = reference.tensor_as_stored("cup.input")?.to_vec::<u8>()?;
    for (target, byte) in interpreter.input_mut(0).iter_mut().zip(input) {
        *target = byte as i8;
    }
    let outputs = interpreter.step_outputs();
    let names: Vec<String> = interpreter.step_names().map(str::to_string).collect();
    let (mut compared, mut exact, mut first) = (0, 0, None);
    let mut lines = Vec::new();
    interpreter.run_visiting(|index, ours| {
        let Ok(tensor) = reference.tensor_as_stored(&format!("cup.tensor.{}", outputs[index]))
        else {
            return;
        };
        let theirs = tensor.to_vec::<u8>().expect("u8로 적었다");
        let different = ours
            .iter()
            .zip(&theirs)
            .filter(|&(&a, &b)| a != b as i8)
            .count();
        let worst = ours
            .iter()
            .zip(&theirs)
            .map(|(&a, &b)| (i32::from(a) - i32::from(b as i8)).abs())
            .max()
            .unwrap_or(0);
        compared += 1;
        if different == 0 {
            exact += 1;
        } else if first.is_none() {
            first = Some(index);
        }
        lines.push(format!(
            "  {index:>2} {:<18} 값 {:>8}개 가운데 다른 것 {different:>7}개, 가장 큰 차이 {worst}",
            names[index],
            ours.len()
        ));
    })?;
    for line in &lines {
        println!("{line}");
    }
    println!("견준 단계 {compared}개 가운데 {exact}개가 비트까지 같음");
    if let Some(index) = first {
        println!("처음 달라지는 단계: {index} ({})", names[index]);
    }
    Ok(())
}
