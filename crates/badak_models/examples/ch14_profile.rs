//! 14장: 순전파의 시간이 어느 연산에 쓰이는지 엔진이 스스로 적게 한다.
//!
//! 실행: `cargo run --release -p badak_models --example ch14_profile -- [모델 디렉터리] [프롬프트 토큰 수] [새 토큰 수]`
//!
//! 프롬프트 64토큰을 처리하고 토큰 16개를 하나씩 더 처리하면서, 계측
//! 프로파일러로 연산별 시간을 모으고 하드웨어 카운터를 읽는다.

use std::time::Instant;

use badak_bench::Counters;
use badak_core::profile;
use badak_core::random::Pcg64;
use badak_models::llama::cache::KvCache;
use badak_models::llama::model::LlamaModel;
use badak_models::models_dir;
use badak_models::tokenizer::TokenId;

const PROMPT_TOKENS: usize = 64;
const NEW_TOKENS: usize = 16;

/// 프롬프트를 처리하고 토큰을 하나씩 더 처리한다. 걸린 시간을 초로 돌려준다.
fn run(
    model: &LlamaModel,
    tokens: &[TokenId],
    prompt: usize,
) -> Result<f64, Box<dyn std::error::Error>> {
    let mut cache = KvCache::new(&model.config);
    let start = Instant::now();
    model.forward_cached(&tokens[..prompt], &mut cache)?;
    for token in &tokens[prompt..] {
        model.forward_cached(std::slice::from_ref(token), &mut cache)?;
    }
    Ok(start.elapsed().as_secs_f64())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let name = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "llama32_1b".to_string());
    let model = LlamaModel::load(models_dir().join(&name))?;
    let mut rng = Pcg64::new(14);
    let vocab = model.config.vocab_size as u64;
    // 토큰의 수는 둘째, 셋째 인자로 바꿀 수 있다.
    let argument = |index: usize, default: usize| {
        std::env::args()
            .nth(index)
            .and_then(|text| text.parse().ok())
            .unwrap_or(default)
    };
    let (prompt, new) = (argument(2, PROMPT_TOKENS), argument(3, NEW_TOKENS));
    let tokens: Vec<TokenId> = (0..prompt + new)
        .map(|_| (rng.next_u64() % vocab) as TokenId)
        .collect();

    // 프로파일러를 끈 채로 한 번, 켠 채로 한 번 실행한다.
    let plain = run(&model, &tokens, prompt)?;
    // ANCHOR: profile
    profile::reset();
    profile::enable(true);
    let before = Counters::now();
    let profiled = run(&model, &tokens, prompt)?;
    let counters = before
        .zip(Counters::now())
        .map(|(before, after)| after.since(before));
    profile::enable(false);
    let records = profile::report();
    // ANCHOR_END: profile

    println!("모델 {name}, 프롬프트 {prompt}토큰을 한꺼번에, 그 뒤 {new}토큰을 하나씩");
    println!();
    println!(
        "{:<28} {:>8} {:>12} {:>12} {:>8}",
        "경로", "횟수", "전체 (초)", "직접 (초)", "비중"
    );
    // 직접 쓴 시간을 모두 더하면 측정한 구간들의 전체 시간이 된다.
    let measured: f64 = records
        .iter()
        .map(|(_, record)| record.own.as_secs_f64())
        .sum();
    for (path, record) in &records {
        println!(
            "{:<28} {:>8} {:>12.3} {:>12.3} {:>7.1}%",
            path,
            record.calls,
            record.total.as_secs_f64(),
            record.own.as_secs_f64(),
            record.own.as_secs_f64() / measured * 100.0
        );
    }

    // 같은 이름의 연산을 경로와 상관없이 모은다.
    println!();
    println!("{:<28} {:>12} {:>8}", "연산", "직접 (초)", "비중");
    let mut by_name: Vec<(&str, f64)> = Vec::new();
    for (path, record) in &records {
        let name = path.rsplit('/').next().unwrap_or(path);
        match by_name.iter_mut().find(|(existing, _)| *existing == name) {
            Some((_, seconds)) => *seconds += record.own.as_secs_f64(),
            None => by_name.push((name, record.own.as_secs_f64())),
        }
    }
    by_name.sort_by(|a, b| b.1.total_cmp(&a.1));
    for (name, seconds) in &by_name {
        println!(
            "{:<28} {:>12.3} {:>7.1}%",
            name,
            seconds,
            seconds / measured * 100.0
        );
    }

    println!();
    println!("{:<34} {:>10.3} 초", "프로파일러를 끄고 잰 시간", plain);
    println!("{:<34} {:>10.3} 초", "프로파일러를 켜고 잰 시간", profiled);
    println!("{:<34} {:>10.3} 초", "구간들이 직접 쓴 시간의 합", measured);
    println!(
        "{:<34} {:>9.2} %",
        "어느 구간에도 들지 않은 시간",
        (profiled - measured) / profiled * 100.0
    );

    if let Some(counters) = counters {
        // 선형 층이 한 곱셈의 수: 블록의 가중치 수 × 토큰 수, 그리고 출력
        // 층은 토큰을 넣을 때마다 한 줄씩.
        let c = &model.config;
        let block = 2 * c.hidden_size * c.hidden_size
            + 2 * c.num_kv_heads * c.head_dim() * c.hidden_size
            + 3 * c.intermediate_size * c.hidden_size;
        let rows = 1 + NEW_TOKENS;
        let multiplies =
            (block * c.num_layers * tokens.len() + c.vocab_size * c.hidden_size * rows) as f64;
        println!();
        println!("{:<34} {:>14.3e}", "선형 층의 곱셈 수", multiplies);
        println!(
            "{:<34} {:>14.3e}",
            "실행한 명령 수", counters.instructions as f64
        );
        println!("{:<34} {:>14.3e}", "쓴 사이클 수", counters.cycles as f64);
        println!(
            "{:<34} {:>14.2}",
            "사이클당 명령 (IPC)",
            counters.instructions_per_cycle()
        );
        println!(
            "{:<34} {:>14.2}",
            "곱셈 하나에 든 명령",
            counters.instructions as f64 / multiplies
        );
        println!(
            "{:<34} {:>14.2}",
            "곱셈 하나에 든 사이클",
            counters.cycles as f64 / multiplies
        );
        println!(
            "{:<34} {:>14.2} GHz",
            "사이클 수 ÷ 시간",
            counters.cycles as f64 / profiled / 1e9
        );
    }
    Ok(())
}
