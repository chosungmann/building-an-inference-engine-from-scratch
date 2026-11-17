//! 12장: 설정에 따라 다음 토큰의 확률분포가 어떻게 달라지는지 본다.
//!
//! 실행: `cargo run --release -p badak_models --example ch12_distribution -- [모델 디렉터리] [글]`

use badak_models::llama::cache::KvCache;
use badak_models::llama::model::LlamaModel;
use badak_models::models_dir;
use badak_models::sampling::{Sampler, SamplerConfig};
use badak_models::tokenizer::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let name = arguments.next().unwrap_or_else(|| "llama32_1b".to_string());
    let prompt = arguments
        .next()
        .unwrap_or_else(|| "Once upon a time, there was a".to_string());

    let directory = models_dir().join(&name);
    let model = LlamaModel::load(&directory)?;
    let tokenizer = Tokenizer::from_tokenizer_json(directory.join("tokenizer.json"))?;
    let mut ids = tokenizer.prefix().to_vec();
    ids.extend(tokenizer.encode(&prompt, false));
    let mut cache = KvCache::new(&model.config);
    let logits = model.forward_cached(&ids, &mut cache)?.to_vec::<f32>()?;

    println!("모델 {name}, 글 {prompt:?}");
    let plain = SamplerConfig::default();
    let settings = [
        ("손질 없음", plain),
        (
            "온도 0.5",
            SamplerConfig {
                temperature: 0.5,
                ..plain
            },
        ),
        (
            "온도 1.5",
            SamplerConfig {
                temperature: 1.5,
                ..plain
            },
        ),
        ("top-k 5", SamplerConfig { top_k: 5, ..plain }),
        (
            "top-p 0.5",
            SamplerConfig {
                top_p: 0.5,
                ..plain
            },
        ),
        (
            "top-p 0.9",
            SamplerConfig {
                top_p: 0.9,
                ..plain
            },
        ),
        (
            "온도 1.5 다음에 top-p 0.9",
            SamplerConfig {
                temperature: 1.5,
                top_p: 0.9,
                ..plain
            },
        ),
        (
            "top-p 0.9 다음에 온도 1.5",
            SamplerConfig {
                temperature: 1.5,
                top_p: 0.9,
                temperature_last: true,
                ..plain
            },
        ),
    ];
    for (label, config) in settings {
        let candidates = Sampler::new(config).distribution(&logits, &ids);
        println!();
        println!("{label}: 후보 {}개", candidates.len());
        for candidate in candidates.iter().take(8) {
            println!(
                "  {:<14} {:>6.2}%",
                format!("{:?}", tokenizer.decode(&[candidate.id])),
                candidate.probability * 100.0
            );
        }
    }

    // 글을 이어 가면서, 자리마다 top-p 0.9가 남기는 후보의 수를 센다.
    // 토큰은 늘 1위를 골라 이어 간다.
    println!();
    println!("자리마다 top-p 0.9가 남기는 후보의 수");
    println!(
        "{:>4} {:<14} {:>8} {:>10}",
        "자리", "고른 토큰", "후보 수", "1위 확률"
    );
    let nucleus = Sampler::new(SamplerConfig {
        top_p: 0.9,
        ..plain
    });
    let mut logits = logits;
    for step in 1..=24 {
        let candidates = nucleus.distribution(&logits, &ids);
        let best = candidates[0].id;
        // 1위의 확률은 손질 전의 것을 적는다.
        let before = Sampler::new(plain).distribution(&logits, &ids)[0].probability;
        println!(
            "{:>4} {:<14} {:>8} {:>9.1}%",
            step,
            format!("{:?}", tokenizer.decode(&[best])),
            candidates.len(),
            before * 100.0
        );
        ids.push(best);
        logits = model.forward_cached(&[best], &mut cache)?.to_vec::<f32>()?;
    }
    Ok(())
}
