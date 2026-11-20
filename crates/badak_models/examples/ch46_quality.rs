//! 46장: 줄인 모델들의 품질을 엔진으로 잰다.
//!
//! 실행: `cargo run --release -p badak_models --example ch46_quality -- [조각 수] 모델 ...`
//!       `cargo run --release -p badak_models --example ch46_quality -- rate 모델 ...`
//!       `cargo run --release -p badak_models --example ch46_quality -- check`
//!
//! 모델은 `models/` 아래의 폴더 이름이다. 첫 모델이 기준이다. 모델마다 두
//! 번 잰다. 파일에 적힌 그대로(f32)와, 엔진이 `Q4_0`으로 줄인 것. 줄인 것이
//! 폰에 실릴 것이다.
//!
//! `check`는 `reference/ch46_train.py`의 흉내 양자화가 엔진의 것과 비트까지
//! 같은지 본다. 먼저 `python reference/ch46_train.py vectors`를 돌린다.

use std::time::Instant;

use badak_core::packed::{Format, Packed};
use badak_formats::safetensors::SafeTensors;
use badak_models::llama::WeightFormat;
use badak_models::llama::cache::{KvCache, KvPrecision};
use badak_models::llama::generate::generate;
use badak_models::llama::model::LlamaModel;
use badak_models::llama::perplexity::{Losses, Protocol, evaluate};
use badak_models::models_dir;
use badak_models::tokenizer::Tokenizer;

const CONTEXT: usize = 512;

// ANCHOR: check
/// 파이썬이 줄였다 되돌린 값과 엔진이 줄였다 되돌린 값을 견준다.
fn check() -> Result<(), Box<dyn std::error::Error>> {
    let file = SafeTensors::open(models_dir().join("smollm2_135m_qat/fake_q4_0.safetensors"))?;
    let weight = file.tensor("weight")?;
    let (rows, cols) = (weight.shape()[0], weight.shape()[1]);
    let theirs = file.tensor("fake_q4_0")?.to_vec::<f32>()?;
    let packed = Packed::quantize(Format::Q4_0, &weight.to_vec::<f32>()?, rows, cols)?;
    let ours = packed.dequantize_rows(0..rows);
    let different = ours
        .iter()
        .zip(&theirs)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    println!("값 {}개 가운데 비트가 다른 것 {different}개", ours.len());
    if different > 0 {
        return Err("흉내 양자화가 엔진의 양자화와 다릅니다".into());
    }
    Ok(())
}
// ANCHOR_END: check

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments: Vec<String> = std::env::args().skip(1).collect();
    if arguments.first().map(String::as_str) == Some("check") {
        return check();
    }
    // `rate`를 앞에 주면 손실은 재지 않고 속도만 잰다. 손실은 돌릴 때마다
    // 같지만 속도는 흔들리므로, 속도만 여러 번 잴 때 쓴다.
    let only_rate = arguments.first().map(String::as_str) == Some("rate");
    if only_rate {
        arguments.remove(0);
    }
    let chunks: usize = match arguments.first().and_then(|text| text.parse().ok()) {
        Some(chunks) => {
            arguments.remove(0);
            chunks
        }
        None => 64,
    };
    if arguments.is_empty() {
        arguments.push("smollm2_135m".to_owned());
    }
    let first = models_dir().join(&arguments[0]);
    let text =
        std::fs::read_to_string(models_dir().join("wikitext2/wikitext-2-raw/wiki.test.raw"))?;
    let tokenizer = Tokenizer::from_tokenizer_json(first.join("tokenizer.json"))?;
    let mut tokens = tokenizer.prefix().to_vec();
    tokens.extend(tokenizer.encode(&text, false));
    let protocol = Protocol {
        context: CONTEXT,
        chunks,
        bos: tokenizer.prefix().first().copied(),
        kv: KvPrecision::F32,
    };
    let mut prompt = tokenizer.prefix().to_vec();
    prompt.extend(tokenizer.encode("Once upon a time, in a small village by the sea,", false));
    println!("WikiText-2 시험 글, 한 번에 {CONTEXT}토큰씩 {chunks}조각. 기준은 첫 줄이다.");
    println!(
        "{:<24} {:>4} {:>10} {:>18} {:>10} {:>18} {:>10} {:>9}",
        "모델",
        "블록",
        "f32 손실",
        "기준과의 차이",
        "Q4_0 손실",
        "기준과의 차이",
        "줄인 값",
        "토큰/초"
    );
    let mut base: Option<Losses> = None;
    for name in &arguments {
        let directory = models_dir().join(name);
        if only_rate {
            let small = LlamaModel::load_as(&directory, WeightFormat::Q4_0)?;
            let mut rates = Vec::new();
            for _ in 0..3 {
                let mut started = None;
                let mut cache = KvCache::new(&small.config);
                let written = generate(&small, &prompt, 128, &mut cache, |_| {
                    started.get_or_insert_with(Instant::now);
                })?;
                rates.push(
                    (written.len().max(2) - 1) as f64
                        / started.map_or(1.0, |at| at.elapsed().as_secs_f64()),
                );
            }
            rates.sort_by(f64::total_cmp);
            println!(
                "rate {name} 블록 {} 토큰/초 {:.0}",
                small.config.num_layers, rates[1]
            );
            continue;
        }
        let full = LlamaModel::load_as(&directory, WeightFormat::F32)?;
        let plain = evaluate(&full, &tokens, protocol)?;
        let base = base.get_or_insert_with(|| plain.clone());
        let small = LlamaModel::load_as(&directory, WeightFormat::Q4_0)?;
        let packed = evaluate(&small, &tokens, protocol)?;
        // 속도: 줄인 모델로 토큰 128개를 쓴다. 첫 토큰(프롬프트 처리)은 뺀다.
        let mut started = None;
        let mut cache = KvCache::new(&small.config);
        let written = generate(&small, &prompt, 128, &mut cache, |_| {
            started.get_or_insert_with(Instant::now);
        })?;
        let rate = (written.len().max(2) - 1) as f64
            / started.map_or(1.0, |at| at.elapsed().as_secs_f64());
        let (plain_gap, plain_error) = plain.difference_from(base);
        let (packed_gap, packed_error) = packed.difference_from(base);
        println!(
            "{:<24} {:>4} {:>10.4} {:>18} {:>10.4} {:>18} {:>+10.4} {:>9.0}",
            name,
            full.config.num_layers,
            plain.mean(),
            format!("{plain_gap:+.4} ± {plain_error:.4}"),
            packed.mean(),
            format!("{packed_gap:+.4} ± {packed_error:.4}"),
            packed.mean() - plain.mean(),
            rate
        );
    }
    Ok(())
}
