//! 26장: `.tflite` 파일에 무엇이 들어 있는지 본다.
//!
//! 실행: `cargo run --release -p badak_graph --example ch26_inspect -- mlp`

use badak_formats::tflite::Model;
use badak_graph::models_dir;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let name = std::env::args().nth(1).unwrap_or_else(|| "mlp".into());
    let path = models_dir().join(format!("mnist_tflite/{name}.tflite"));
    let model = Model::open(&path)?;
    println!("파일        {name}.tflite, {}바이트", model.file_len());
    println!("스키마      {}판", model.version);
    println!("설명        {:?}", model.description);

    println!("\n연산의 종류");
    for (index, code) in model.operator_codes.iter().enumerate() {
        println!("  {index}: 번호 {:<3} 판 {}", code.builtin, code.version);
    }

    let graph = &model.subgraphs[0];
    println!("\n부분 그래프 {:?}", graph.name);
    println!("  입력 {:?}, 출력 {:?}", graph.inputs, graph.outputs);
    println!("\n텐서");
    let mut stored = 0;
    for (index, tensor) in graph.tensors.iter().enumerate() {
        let constant = model.constant(tensor)?;
        let place = match &constant {
            Some(value) if value.is_borrowed() => "파일의 값을 그대로",
            Some(_) => "파일의 값을 복사",
            None => "실행하면서 계산",
        };
        stored += constant.map_or(0, |value| value.size_in_bytes());
        println!(
            "  {index:>2} {:<12} {:<44} {place}",
            format!("{:?}", tensor.shape),
            tensor.name
        );
    }
    println!(
        "  파일에 든 값 {stored}바이트 (파일의 {:.1}%)",
        100.0 * stored as f64 / model.file_len() as f64
    );

    println!("\n연산 (실행할 순서)");
    for (index, operator) in graph.operators.iter().enumerate() {
        let inputs: Vec<String> = operator
            .inputs
            .iter()
            .map(|input| input.map_or("-".into(), |value| value.to_string()))
            .collect();
        println!(
            "  {index:>2} 종류 {} 입력 [{}] 출력 {:?} {:?}",
            operator.code,
            inputs.join(", "),
            operator.outputs,
            operator.options
        );
    }
    Ok(())
}
