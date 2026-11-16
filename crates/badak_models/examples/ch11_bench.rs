//! 11장: 토큰을 하나씩 뽑는 데 걸리는 시간을 캐시가 없을 때와 있을 때로 잰다.
//!
//! 실행: `cargo run --release -p badak_models --example ch11_bench -- [모델 디렉터리] [새 토큰 수]`

use std::time::Instant;

use badak_models::llama::cache::KvCache;
use badak_models::llama::generate::{generate, generate_without_cache};
use badak_models::llama::model::LlamaModel;
use badak_models::models_dir;
use badak_models::tokenizer::Tokenizer;

/// 토큰이 나올 때마다, 앞 토큰이 나온 뒤로 흐른 시간을 적는다.
struct Stopwatch {
    last: Instant,
    steps: Vec<f64>,
}

impl Stopwatch {
    fn new() -> Self {
        Self {
            last: Instant::now(),
            steps: Vec::new(),
        }
    }

    fn lap(&mut self) {
        let now = Instant::now();
        self.steps.push((now - self.last).as_secs_f64());
        self.last = now;
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let name = arguments.next().unwrap_or_else(|| "llama32_1b".to_string());
    let count: usize = arguments.next().map_or(Ok(24), |count| count.parse())?;

    let directory = models_dir().join(&name);
    let model = LlamaModel::load(&directory)?;
    let tokenizer = Tokenizer::from_tokenizer_json(directory.join("tokenizer.json"))?;
    let mut prompt = tokenizer.prefix().to_vec();
    prompt.extend(tokenizer.encode("The capital of France is", false));

    let mut slow = Stopwatch::new();
    let without = generate_without_cache(&model, &prompt, count, |_| slow.lap())?;

    let mut cache = KvCache::new(&model.config);
    let mut fast = Stopwatch::new();
    let with = generate(&model, &prompt, count, &mut cache, |_| fast.lap())?;
    assert_eq!(without, with, "캐시가 있든 없든 같은 토큰을 뽑아야 한다");

    println!(
        "모델 {name}, 프롬프트의 토큰 {}개, 새 토큰 {count}개",
        prompt.len()
    );
    println!("{:?}", tokenizer.decode(&with));
    println!();
    println!(
        "{:>4} {:>10} {:>12} {:>12}",
        "순서", "글의 길이", "캐시 없이", "캐시 사용"
    );
    for (step, (slow, fast)) in slow.steps.iter().zip(&fast.steps).enumerate() {
        println!(
            "{:>4} {:>10} {:>10.3} 초 {:>10.3} 초",
            step + 1,
            prompt.len() + step,
            slow,
            fast
        );
    }
    let (slow_total, fast_total): (f64, f64) = (slow.steps.iter().sum(), fast.steps.iter().sum());
    println!();
    println!(
        "{:<14} {:>10.2} 초 {:>10.2} 초",
        "전체", slow_total, fast_total
    );
    println!("{:<14} {:>10.1} 배", "캐시의 효과", slow_total / fast_total);
    Ok(())
}
