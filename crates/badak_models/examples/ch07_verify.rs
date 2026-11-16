//! 7장: badak의 계산을 PyTorch의 계산과 층마다 대조한다.
//!
//! 실행: `cargo run --release -p badak_models --example ch07_verify`

use badak_core::approx::compare;
use badak_core::{Tensor, ops};
use badak_formats::idx;
use badak_formats::safetensors::SafeTensors;
use badak_models::mnist::MnistMlp;
use badak_models::models_dir;

// ANCHOR: report
/// 5장에서 정한 기준: 많은 수를 더하는 계산의 상대 오차.
const RTOL: f32 = 1e-5;

/// 한 층의 결과를 비교해 한 줄로 적는다.
fn report(
    name: &str,
    actual: &Tensor,
    expected: &Tensor,
) -> Result<bool, Box<dyn std::error::Error>> {
    let (actual, expected) = (actual.to_vec::<f32>()?, expected.to_vec::<f32>()?);
    // 절대 오차는 이 텐서에서 가장 큰 값의 크기에 상대 오차를 곱한 것으로 잡는다.
    let scale = expected
        .iter()
        .fold(0.0_f32, |max, value| max.max(value.abs()));
    let result = compare(&actual, &expected, RTOL, RTOL * scale);
    println!(
        "{:<12} {:>8} {:>12.3} {:>14.3e} {:>12.3e} {:>8}",
        name,
        result.len,
        scale,
        result.max_abs_diff,
        RTOL * scale,
        if result.is_close() {
            "통과"
        } else {
            "실패"
        }
    );
    Ok(result.is_close())
}

// ANCHOR_END: report

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let models = models_dir();
    let model = MnistMlp::load(models.join("mnist_mlp/model.safetensors"))?;
    let reference = SafeTensors::open(models.join("mnist_mlp/reference.safetensors"))?;
    let images = idx::open(models.join("mnist/t10k-images-idx3-ubyte"))?;

    // PyTorch가 층별 출력을 남겨 둔 앞의 몇 장.
    let expected_input = reference.tensor("input")?;
    let traced = expected_input.shape()[0];
    let input = model.preprocess(&images.slice(0, 0..traced)?)?;
    let trace = model.trace(&input)?;

    println!("== 층별 대조: 앞의 {traced}장 ==");
    println!(
        "{:<12} {:>8} {:>12} {:>14} {:>12} {:>8}",
        "층", "원소 수", "최대 크기", "최대 오차", "허용 오차", "결과"
    );
    let mut ok = true;
    ok &= report("전처리", &input, &expected_input)?;
    ok &= report("fc1", &trace.fc1, &reference.tensor("fc1")?)?;
    ok &= report("relu", &trace.relu, &reference.tensor("relu")?)?;
    ok &= report("logits", &trace.logits, &reference.tensor("logits")?)?;

    println!();
    println!("== 전체 대조: 시험 데이터 {}장 ==", images.shape()[0]);
    println!(
        "{:<12} {:>8} {:>12} {:>14} {:>12} {:>8}",
        "층", "원소 수", "최대 크기", "최대 오차", "허용 오차", "결과"
    );
    let logits = model.forward(&model.preprocess(&images)?)?;
    ok &= report("logits", &logits, &reference.tensor("all_logits")?)?;

    let predictions = ops::argmax(&logits, 1)?.to_vec::<i32>()?;
    let expected = reference.tensor("predictions")?.to_vec::<i32>()?;
    let different = predictions
        .iter()
        .zip(&expected)
        .filter(|(a, b)| a != b)
        .count();
    println!();
    println!(
        "예측이 PyTorch와 다른 사진: {}장 중 {different}장",
        expected.len()
    );
    ok &= different == 0;

    println!();
    println!(
        "{}",
        if ok {
            "모든 대조를 통과했습니다."
        } else {
            "대조에 실패했습니다."
        }
    );
    if !ok {
        std::process::exit(1);
    }
    Ok(())
}
