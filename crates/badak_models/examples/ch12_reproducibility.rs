//! 12장: 점수가 조금 달라지면 뽑히는 글이 얼마나 달라지는지 본다.
//!
//! 실행: `cargo run --release -p badak_models --example ch12_reproducibility -- [모델 디렉터리]`
//!
//! 같은 씨앗으로 글을 두 번 쓴다. 한 번은 그대로, 한 번은 점수마다 작은
//! 잡음을 더한다. 잡음은 "다른 엔진이 같은 모델을 계산했을 때의 차이"를
//! 흉내 낸 것이다. 두 글이 몇 번째 토큰에서 갈리는지 센다.

use badak_core::random::Pcg64;
use badak_models::llama::cache::KvCache;
use badak_models::llama::model::LlamaModel;
use badak_models::models_dir;
use badak_models::sampling::{Sampler, SamplerConfig};
use badak_models::tokenizer::{TokenId, Tokenizer};

const NEW_TOKENS: usize = 80;
const SEEDS: u64 = 20;

// ANCHOR: write
/// 글을 이어 쓴다. 점수마다 `-noise`에서 `+noise` 사이의 잡음을 더한다.
fn write(
    model: &LlamaModel,
    prompt: &[TokenId],
    config: SamplerConfig,
    noise: f32,
) -> Result<Vec<TokenId>, Box<dyn std::error::Error>> {
    let mut sampler = Sampler::new(config);
    // 잡음은 토큰을 뽑는 난수와 섞이지 않게 다른 생성기로 만든다.
    let mut noise_rng = Pcg64::new(config.seed ^ 0x5eed);
    let mut cache = KvCache::new(&model.config);
    let mut history = prompt.to_vec();
    let mut input = prompt.to_vec();
    for _ in 0..NEW_TOKENS {
        let mut logits = model.forward_cached(&input, &mut cache)?.to_vec::<f32>()?;
        if noise > 0.0 {
            for logit in &mut logits {
                *logit += noise * (2.0 * noise_rng.next_f64() as f32 - 1.0);
            }
        }
        let next = sampler.sample(&logits, &history);
        history.push(next);
        input = vec![next];
    }
    Ok(history.split_off(prompt.len()))
}
// ANCHOR_END: write

/// 두 글이 처음으로 달라지는 자리. 끝까지 같으면 `None`.
fn first_difference(a: &[TokenId], b: &[TokenId]) -> Option<usize> {
    a.iter().zip(b).position(|(a, b)| a != b)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let name = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "smollm2_135m".to_string());
    let directory = models_dir().join(&name);
    let model = LlamaModel::load(&directory)?;
    let tokenizer = Tokenizer::from_tokenizer_json(directory.join("tokenizer.json"))?;
    let mut prompt = tokenizer.prefix().to_vec();
    prompt.extend(tokenizer.encode("Once upon a time, there was a", false));

    println!("모델 {name}, 새 토큰 {NEW_TOKENS}개, 씨앗 {SEEDS}개");
    println!();
    println!(
        "{:<28} {:>10} {:>12} {:>16}",
        "설정", "잡음", "갈린 글", "갈린 자리(중앙값)"
    );
    let sampled = SamplerConfig {
        temperature: 0.8,
        top_k: 40,
        top_p: 0.95,
        ..SamplerConfig::default()
    };
    let greedy = SamplerConfig {
        temperature: 0.0,
        ..SamplerConfig::default()
    };
    for (label, config, seeds) in [
        ("늘 1위 (온도 0)", greedy, 1),
        ("온도 0.8, top-k 40, top-p 0.95", sampled, SEEDS),
    ] {
        for noise in [0.0_f32, 1e-4, 1e-2, 1.0] {
            let mut differences = Vec::new();
            for seed in 1..=seeds {
                let config = SamplerConfig { seed, ..config };
                let baseline = write(&model, &prompt, config, 0.0)?;
                let other = write(&model, &prompt, config, noise)?;
                differences.extend(first_difference(&baseline, &other));
            }
            differences.sort_unstable();
            let median = differences
                .get(differences.len() / 2)
                .map_or("-".to_string(), |step| format!("{}번째", step + 1));
            println!(
                "{:<28} {:>10} {:>9} / {:<2} {:>12}",
                label,
                noise,
                differences.len(),
                seeds,
                median
            );
        }
    }
    Ok(())
}
