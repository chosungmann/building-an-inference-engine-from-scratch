//! 10장: 글을 넣고 다음에 올 토큰을 예측한다.
//!
//! 실행: `cargo run --release -p badak_models --example ch10_next_token -- [모델 디렉터리] [글] [앞에 붙일 토큰 번호]`
//!
//! 모델 디렉터리의 이름을 주지 않으면 `llama32_1b`를 쓴다.

use std::time::Instant;

use badak_core::prob::softmax;
use badak_models::llama::model::LlamaModel;
use badak_models::models_dir;
use badak_models::tokenizer::Tokenizer;

fn seconds<R>(f: impl FnOnce() -> R) -> (R, f64) {
    let start = Instant::now();
    let result = f();
    (result, start.elapsed().as_secs_f64())
}

/// 여러 번 실행해 가장 빠른 시간을 초로 돌려준다.
fn best_of<R>(repeats: usize, mut f: impl FnMut() -> R) -> f64 {
    (0..repeats)
        .map(|_| seconds(&mut f).1)
        .fold(f64::INFINITY, f64::min)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let name = arguments.next().unwrap_or_else(|| "llama32_1b".to_string());
    let prompt = arguments
        .next()
        .unwrap_or_else(|| "The capital of France is".to_string());
    // 실험용: 셋째 인자로 글 앞에 붙일 토큰 번호를 직접 정한다. 빈 문자열이면
    // 아무것도 붙이지 않는다.
    let prefix: Option<Vec<u32>> = arguments
        .next()
        .map(|list| list.split(',').filter_map(|id| id.parse().ok()).collect());
    let directory = models_dir().join(&name);

    let (model, load_model) = seconds(|| LlamaModel::load(&directory));
    let model = model?;
    let (tokenizer, load_tokenizer) =
        seconds(|| Tokenizer::from_tokenizer_json(directory.join("tokenizer.json")));
    let tokenizer = tokenizer?;

    // ANCHOR: predict
    // 글을 토큰으로 바꾼다. 토크나이저 설정이 정한 토큰을 맨 앞에 붙인다.
    let mut ids = prefix.unwrap_or_else(|| tokenizer.prefix().to_vec());
    ids.extend(tokenizer.encode(&prompt, false));

    // 마지막 토큰 다음에 올 토큰의 점수를 얻고, 확률로 바꾼다.
    let (logits, forward) = seconds(|| model.next_token_logits(&ids));
    let probabilities = softmax(logits?.as_slice::<f32>()?);

    // 확률이 높은 순서로 줄을 세운다.
    let mut ranked: Vec<(usize, f32)> = probabilities.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    // ANCHOR_END: predict

    println!("모델 {name} (파라미터 {}개)", model.parameter_count());
    println!("글: {prompt:?}");
    println!("토큰 {}개: {ids:?}", ids.len());
    println!();
    println!("다음 토큰의 후보");
    println!("{:>4} {:>8} {:<16} {:>8}", "순위", "번호", "토큰", "확률");
    for (rank, &(id, probability)) in ranked.iter().take(10).enumerate() {
        println!(
            "{:>4} {:>8} {:<16} {:>7.2}%",
            rank + 1,
            id,
            format!("{:?}", tokenizer.decode(&[id as u32])),
            probability * 100.0
        );
    }

    println!();
    println!("걸린 시간");
    println!("{:<28} {:>8.2} 초", "가중치 읽기", load_model);
    println!("{:<28} {:>8.2} 초", "토크나이저 읽기", load_tokenizer);
    println!("{:<28} {:>8.2} 초", "순전파 (첫 실행)", forward);

    // 순전파를 두 부분으로 나누어 다시 잰다. 세 번 가운데 가장 빠른 값이다.
    let states = model.hidden_states(&ids)?;
    let last = states.last().expect("임베딩이 있다");
    let row = last.slice(0, ids.len() - 1..ids.len())?;
    let blocks = best_of(3, || model.hidden_states(&ids).map(|states| states.len()));
    let one_row = best_of(3, || model.logits(&row).map(|logits| logits.len()));
    let all_rows = best_of(3, || model.logits(last).map(|logits| logits.len()));
    println!("{:<28} {:>8.2} 초", "블록 전체", blocks);
    println!("{:<28} {:>8.2} 초", "출력 층 (마지막 줄만)", one_row);
    println!("{:<28} {:>8.2} 초", "출력 층 (모든 줄)", all_rows);
    Ok(())
}
