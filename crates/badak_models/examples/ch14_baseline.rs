//! 14장: 3부 내내 견줄 기준 측정.
//!
//! 실행: `cargo run --release -p badak_models --example ch14_baseline -- [모델 디렉터리] [되풀이 횟수]`
//!
//! 13장과 같은 두 가지 일을 `badak_bench`의 도구로 잰다. 프롬프트 256토큰을
//! 한꺼번에 처리하는 일(pp256)과 토큰 32개를 하나씩 처리하는 일(tg32)이다.
//! 결과는 `bench/baseline/`에 적어 두고, 엔진을 고칠 때마다 다시 재서 견준다.

use badak_bench::{Counters, Plan, Summary, environment, measure};
use badak_core::random::Pcg64;
use badak_models::llama::cache::KvCache;
use badak_models::llama::model::LlamaModel;
use badak_models::models_dir;
use badak_models::tokenizer::{TokenId, Tokenizer};

const PROMPT_TOKENS: usize = 256;
const NEW_TOKENS: usize = 32;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let name = arguments.next().unwrap_or_else(|| "llama32_1b".to_string());
    let repetitions: usize = arguments.next().map_or(Ok(5), |count| count.parse())?;
    let plan = Plan {
        warmup: 1,
        repetitions,
    };

    let directory = models_dir().join(&name);
    let model = LlamaModel::load(&directory)?;
    let tokenizer = Tokenizer::from_tokenizer_json(directory.join("tokenizer.json"))?;
    let mut rng = Pcg64::new(13);
    let first = tokenizer.prefix().first().copied().unwrap_or(0);
    let vocab = model.config.vocab_size as u64;
    let tokens: Vec<TokenId> = std::iter::once(first)
        .chain(std::iter::repeat_with(|| {
            (rng.next_u64() % vocab) as TokenId
        }))
        .take(PROMPT_TOKENS)
        .collect();

    println!("기준 측정: badak, 모델 {name}");
    environment::print();
    println!(
        "측정 계획: 준비 운동 {}번, 되풀이 {}번",
        plan.warmup, plan.repetitions
    );
    println!();
    println!(
        "{:<8} {:>10} {:>10} {:>10} {:>10} {:>8} {:>12} {:>6} {:>6}",
        "시험",
        "토큰/초",
        "중앙값(초)",
        "최소(초)",
        "최대(초)",
        "흩어짐",
        "토큰당 명령",
        "IPC",
        "GHz"
    );

    // ANCHOR: baseline
    type Work<'a> = Box<dyn FnMut() -> usize + 'a>;
    let tests: [(&str, usize, Work); 2] = [
        (
            "pp256",
            PROMPT_TOKENS,
            Box::new(|| {
                let mut cache = KvCache::new(&model.config);
                model
                    .forward_cached(&tokens, &mut cache)
                    .map_or(0, |logits| logits.len())
            }),
        ),
        (
            "tg32",
            NEW_TOKENS,
            Box::new(|| {
                let mut cache = KvCache::new(&model.config);
                let mut done = 0;
                for token in &tokens[..NEW_TOKENS] {
                    done += model
                        .forward_cached(std::slice::from_ref(token), &mut cache)
                        .map_or(0, |logits| logits.len());
                }
                done
            }),
        ),
    ];
    for (label, token_count, mut work) in tests {
        let before = Counters::now();
        let samples = measure(plan, &mut work);
        let spent = before
            .zip(Counters::now())
            .map(|(before, after)| after.since(before));
        let summary = Summary::new(&samples);
        // 준비 운동까지 합친 실행 횟수로 나누어 토큰 하나의 몫을 구한다.
        let processed = ((plan.warmup + plan.repetitions) * token_count) as f64;
        // 사이클 수를 걸린 시간으로 나누면 프로세서가 실제로 돈 속도가 나온다.
        let elapsed = summary.mean * (plan.warmup + plan.repetitions) as f64;
        let (instructions, ipc, clock) = spent.map_or((f64::NAN, f64::NAN, f64::NAN), |c| {
            (
                c.instructions as f64 / processed,
                c.instructions_per_cycle(),
                c.cycles as f64 / elapsed / 1e9,
            )
        });
        println!(
            "{:<8} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>7.2}% {:>12.3e} {:>6.2} {:>6.2}",
            label,
            token_count as f64 / summary.median,
            summary.median,
            summary.min,
            summary.max,
            summary.relative_deviation() * 100.0,
            instructions,
            ipc,
            clock
        );
    }
    // ANCHOR_END: baseline
    Ok(())
}
