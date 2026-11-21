//! 14장: 언어 모델의 속도 지표를 잰다.
//!
//! 실행: `cargo run --release -p badak_models --example ch14_metrics -- [모델 디렉터리] [새 토큰 수] [되풀이 횟수]`
//!
//! 같은 질문을 여러 번 처리하면서 첫 토큰까지의 시간(TTFT)과 토큰 사이의
//! 간격(ITL)을 모으고, 분위수로 요약한다.

use std::time::Instant;

use badak_bench::Summary;
use badak_models::llama::cache::KvCache;
use badak_models::llama::generate::generate;
use badak_models::llama::model::LlamaModel;
use badak_models::models_dir;
use badak_models::tokenizer::Tokenizer;
use badak_models::tokenizer::chat::{ChatFormat, Message};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let name = arguments
        .next()
        .unwrap_or_else(|| "smollm2_135m".to_string());
    let new_tokens: usize = arguments.next().map_or(Ok(64), |count| count.parse())?;
    let requests: usize = arguments.next().map_or(Ok(8), |count| count.parse())?;

    let directory = models_dir().join(&name);
    let model = LlamaModel::load(&directory)?;
    let tokenizer = Tokenizer::from_tokenizer_json(directory.join("tokenizer.json"))?;
    // 틀의 뼈대만 쓴다(기본 지시문 없이. 47.3절).
    let format = ChatFormat::detect(&tokenizer);
    let question = "Write a short story about a robot who learns to paint.";
    let mut prompt = tokenizer.prefix().to_vec();
    prompt.extend(tokenizer.encode(
        &format.render(&[Message::new("user", question)], true),
        true,
    ));

    // ANCHOR: collect
    let (mut first_token, mut gaps, mut speeds) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..requests {
        let mut cache = KvCache::new(&model.config);
        let start = Instant::now();
        // 토큰이 나온 시각들.
        let mut arrivals = Vec::new();
        generate(&model, &prompt, new_tokens, &mut cache, |_| {
            arrivals.push(start.elapsed().as_secs_f64())
        })?;

        // 첫 토큰까지의 시간: 요청을 받은 때부터 첫 토큰이 나온 때까지.
        first_token.push(arrivals[0]);
        // 토큰 사이의 간격: 이웃한 두 토큰이 나온 시각의 차이.
        gaps.extend(arrivals.windows(2).map(|pair| pair[1] - pair[0]));
        // 생성 속도: 첫 토큰 뒤의 토큰 수를 그 뒤에 흐른 시간으로 나눈 것.
        let last = arrivals[arrivals.len() - 1];
        speeds.push((arrivals.len() - 1) as f64 / (last - arrivals[0]));
    }
    // ANCHOR_END: collect

    println!(
        "모델 {name}, 프롬프트 {}토큰, 새 토큰 {new_tokens}개, 요청 {requests}번",
        prompt.len()
    );
    println!();
    println!(
        "{:<26} {:>6} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "지표", "표본", "최소", "중앙값", "평균", "p90", "p99", "최대"
    );
    let line = |label: &str, samples: &[f64], scale: f64| {
        let s = Summary::new(samples);
        println!(
            "{:<26} {:>6} {:>9.2} {:>9.2} {:>9.2} {:>9.2} {:>9.2} {:>9.2}",
            label,
            s.count,
            s.min * scale,
            s.median * scale,
            s.mean * scale,
            s.p90 * scale,
            s.p99 * scale,
            s.max * scale
        );
    };
    line("첫 토큰까지 (밀리초)", &first_token, 1e3);
    line("토큰 사이의 간격 (밀리초)", &gaps, 1e3);
    line("생성 속도 (토큰/초)", &speeds, 1.0);
    Ok(())
}
