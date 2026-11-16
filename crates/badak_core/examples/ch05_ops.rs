//! 5장: 텐서 연산자를 써 본다.
//!
//! 실행: `cargo run --release -p badak_core --example ch05_ops`

use badak_core::{Result, Tensor, ops};

fn show(label: &str, tensor: &Tensor) -> Result<()> {
    let values: Vec<String> = match tensor.dtype() {
        badak_core::DType::I32 => tensor
            .to_vec::<i32>()?
            .iter()
            .map(|v| v.to_string())
            .collect(),
        _ => tensor
            .to_vec::<f32>()?
            .iter()
            .map(|v| format!("{v:.3}"))
            .collect(),
    };
    println!(
        "{label:<26} 형상 {:<8} [{}]",
        format!("{:?}", tensor.shape()),
        values.join(", ")
    );
    Ok(())
}

fn main() -> Result<()> {
    // 점수 세 개짜리 줄이 두 개 있는 텐서. 분류 모델의 출력이라고 생각하자.
    let scores = Tensor::from_vec(vec![2.0_f32, 1.0, 0.1, 0.5, 2.5, 0.0], &[2, 3])?;
    let bias = Tensor::from_vec(vec![0.0_f32, 0.0, 1.0], &[3])?;

    println!("== 원소별 연산과 브로드캐스팅 ==");
    show("scores", &scores)?;
    show("bias", &bias)?;
    let shifted = ops::add(&scores, &bias)?;
    show("add(scores, bias)", &shifted)?;
    show(
        "relu(sub(scores, 1))",
        &ops::relu(&ops::sub(&scores, &Tensor::from_vec(vec![1.0_f32], &[])?)?)?,
    )?;

    println!();
    println!("== 축소 ==");
    show("sum(axis 0)", &ops::sum(&scores, 0, false)?)?;
    show("sum(axis 1)", &ops::sum(&scores, 1, false)?)?;
    show("sum(axis 1, keepdim)", &ops::sum(&scores, 1, true)?)?;
    show("mean(axis 1)", &ops::mean(&scores, 1, false)?)?;
    show("max(axis 0)", &ops::max(&scores, 0, false)?)?;

    println!();
    println!("== 점수에서 답까지 ==");
    let probabilities = ops::softmax(&shifted, 1)?;
    show("softmax(axis 1)", &probabilities)?;
    show("각 줄의 합", &ops::sum(&probabilities, 1, false)?)?;
    show("argmax(axis 1)", &ops::argmax(&probabilities, 1)?)?;

    println!();
    println!("== 형상이 맞지 않으면 ==");
    let wrong = Tensor::from_vec(vec![0.0_f32; 4], &[4])?;
    println!(
        "add(scores, 길이 4)        {}",
        ops::add(&scores, &wrong).unwrap_err()
    );
    println!(
        "matmul(scores, scores)     {}",
        ops::matmul(&scores, &scores).unwrap_err()
    );
    Ok(())
}
