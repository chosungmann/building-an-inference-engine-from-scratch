//! 10장: 모델 전체의 순전파를 PyTorch와 블록마다 대조한다.
//!
//! 실행: `cargo run --release -p badak_models --example ch10_verify -- [모델 디렉터리]`
//!
//! 9장과 달리 블록의 출력을 다음 블록에 그대로 넘긴다. 오차가 블록을 지나며
//! 어떻게 쌓이는지 볼 수 있다.

use badak_core::approx::compare;
use badak_core::{Tensor, ops};
use badak_formats::safetensors::SafeTensors;
use badak_models::llama::model::LlamaModel;
use badak_models::models_dir;

// ANCHOR: tolerance
/// 블록을 여러 개 지난 결과에 허용하는 상대 오차.
///
/// 연산 하나에는 1e-5를 썼다(5장). 블록을 지날 때마다 그만큼의 오차가 새로
/// 생겨 쌓이므로 열 배의 여유를 둔다.
const RTOL: f32 = 1e-4;
// ANCHOR_END: tolerance

// ANCHOR: report
/// 값들 가운데 가장 큰 크기.
fn magnitude(values: &[f32]) -> f32 {
    values.iter().fold(0.0, |max, value| max.max(value.abs()))
}

/// 한 단계를 비교해 한 줄로 적는다. 돌려주는 값은 통과 여부다.
///
/// 전체를 한 번 비교하고, 첫 줄(첫 토큰)을 뺀 나머지를 한 번 더 비교한다. 첫 줄에 유난히 큰 값이 있어서, 전체의 최댓값을 기준으로
/// 삼으면 나머지 줄의 오차가 가려지기 때문이다.
fn report(
    name: &str,
    actual: &Tensor,
    expected: &Tensor,
) -> Result<bool, Box<dyn std::error::Error>> {
    let width = expected.shape()[1];
    let (actual, expected) = (actual.to_vec::<f32>()?, expected.to_vec::<f32>()?);

    let scale = magnitude(&expected);
    let all = compare(&actual, &expected, RTOL, RTOL * scale);
    let rest_scale = magnitude(&expected[width..]);
    let rest = compare(
        &actual[width..],
        &expected[width..],
        RTOL,
        RTOL * rest_scale,
    );

    let ok = all.is_close() && rest.is_close();
    println!(
        "{:<14} {:>9.3} {:>10.2e} {:>10.2e} {:>9.3} {:>10.2e} {:>10.2e} {:>5}",
        name,
        scale,
        all.max_abs_diff,
        all.max_abs_diff / scale,
        rest_scale,
        rest.max_abs_diff,
        rest.max_abs_diff / rest_scale,
        if ok { "통과" } else { "실패" }
    );
    Ok(ok)
}
// ANCHOR_END: report

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let name = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "llama32_1b".to_string());
    let directory = models_dir().join(&name);
    let model = LlamaModel::load(&directory)?;
    let reference = SafeTensors::open(directory.join("reference.safetensors"))?;
    let ids: Vec<u32> = reference
        .tensor("ids")?
        .to_vec::<i32>()?
        .iter()
        .map(|&id| id as u32)
        .collect();
    println!(
        "모델 {name}, 글 {:?}, 토큰 {}개",
        reference.metadata()["prompt"],
        ids.len()
    );

    // PyTorch가 남긴 단계별 결과: (블록 수 + 1, 토큰 수, hidden_size)
    let expected_states = reference.tensor("hidden_states")?;
    let states = model.hidden_states(&ids)?;

    println!();
    println!("{:<14} {:^31} {:^31}", "", "전체", "첫 줄을 뺀 나머지");
    println!(
        "{:<14} {:>9} {:>10} {:>10} {:>9} {:>10} {:>10} {:>5}",
        "단계",
        "최대 크기",
        "최대 오차",
        "상대 오차",
        "최대 크기",
        "최대 오차",
        "상대 오차",
        "결과"
    );
    // ANCHOR: chain
    let last = states.len() - 1;
    let mut ok = true;
    for (index, state) in states.iter().enumerate() {
        let expected = expected_states.slice(0, index..index + 1)?.squeeze(0)?;
        ok &= match index {
            0 => report("임베딩", state, &expected)?,
            // PyTorch가 남긴 마지막 결과는 마지막 정규화까지 거친 것이다.
            _ if index == last => report(
                &format!("블록 {index} + 정규화"),
                &model.final_norm(state)?,
                &expected,
            )?,
            _ => report(&format!("블록 {index}"), state, &expected)?,
        };
    }

    let logits = model.logits(&states[last])?;
    ok &= report("점수", &logits, &reference.tensor("logits")?)?;
    // ANCHOR_END: chain

    // 토큰마다 가장 높은 점수를 받은 토큰이 PyTorch와 같은가.
    let ours = ops::argmax(&logits, 1)?.to_vec::<i32>()?;
    let theirs = ops::argmax(&reference.tensor("logits")?, 1)?.to_vec::<i32>()?;
    let same = ours.iter().zip(&theirs).filter(|(a, b)| a == b).count();
    println!();
    println!(
        "가장 높은 점수의 토큰이 같은 자리: {}곳 중 {same}곳",
        ours.len()
    );
    ok &= same == ours.len();

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
