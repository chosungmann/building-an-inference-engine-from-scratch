//! 18장: 스레드의 수에 따라 속도가 어떻게 달라지는지 잰다.
//!
//! 실행: `cargo run --release -p badak_models --example ch18_scaling -- [모델 디렉터리] [스레드 수들]`
//!
//! 14장의 기준 측정과 같은 두 가지 일(pp256, tg32)을 스레드의 수를 바꿔 가며
//! 잰다. 다섯 번 가운데 가운데 값을 적는다.

use badak_bench::{Plan, Summary, environment, measure};
use badak_core::random::Pcg64;
use badak_core::threads;
use badak_models::llama::cache::KvCache;
use badak_models::llama::model::LlamaModel;
use badak_models::models_dir;
use badak_models::tokenizer::{TokenId, Tokenizer};

const PROMPT_TOKENS: usize = 256;
const NEW_TOKENS: usize = 32;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let name = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "llama32_1b".to_string());
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

    let pool = threads::global();
    println!("스레드의 수에 따른 속도: badak, 모델 {name}");
    environment::print();
    println!();
    println!(
        "{:>8} {:>16} {:>10} {:>16} {:>10}",
        "스레드", "프롬프트 처리", "배수", "토큰 생성", "배수"
    );
    let plan = Plan {
        warmup: 1,
        repetitions: 5,
    };
    let mut base = None;
    // 둘째 인자로 스레드의 수를 쉼표로 이어 줄 수 있다.
    let counts: Vec<usize> = match std::env::args().nth(2) {
        Some(list) => list.split(',').filter_map(|n| n.parse().ok()).collect(),
        None => vec![1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16],
    };
    for count in counts {
        if count > pool.pool().threads() {
            break;
        }
        // 두 가지 일 모두 같은 수의 스레드를 쓰게 한다.
        pool.set(count, count);
        let prompt = Summary::new(&measure(plan, || {
            let mut cache = KvCache::new(&model.config);
            model
                .forward_cached(&tokens, &mut cache)
                .map_or(0, |logits| logits.len())
        }));
        let generation = Summary::new(&measure(plan, || {
            let mut cache = KvCache::new(&model.config);
            let mut done = 0;
            for token in &tokens[..NEW_TOKENS] {
                done += model
                    .forward_cached(std::slice::from_ref(token), &mut cache)
                    .map_or(0, |logits| logits.len());
            }
            done
        }));
        let speeds = (
            PROMPT_TOKENS as f64 / prompt.median,
            NEW_TOKENS as f64 / generation.median,
        );
        let base = *base.get_or_insert(speeds);
        println!(
            "{:>8} {:>9.1} 토큰/초 {:>9.2}배 {:>9.1} 토큰/초 {:>9.2}배",
            count,
            speeds.0,
            speeds.0 / base.0,
            speeds.1,
            speeds.1 / base.1
        );
    }
    Ok(())
}
