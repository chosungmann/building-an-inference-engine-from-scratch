//! 41장: LoRA 어댑터를 붙여 PEFT의 정답과 견준다.
//!
//! 실행: `cargo run --release -p badak_models --example ch41_lora`
//!
//! 어댑터와 정답은 `reference/ch41_train_lora.py`가 만든다
//! (`models/smollm2_135m_lora/`). 없으면 건너뛴다.

use std::time::Instant;

use badak_formats::safetensors::SafeTensors;
use badak_models::llama::cache::KvCache;
use badak_models::llama::generate::generate;
use badak_models::llama::lora::Adapter;
use badak_models::llama::model::LlamaModel;
use badak_models::models_dir;
use badak_models::tokenizer::{TokenId, Tokenizer};

type Outcome<T> = Result<T, Box<dyn std::error::Error>>;

/// 행렬 가중치가 메모리에서 차지하는 크기(MB).
fn megabytes(model: &LlamaModel) -> f64 {
    let bytes: usize = model
        .matrices()
        .iter()
        .map(|(_, _, matrix)| matrix.size_in_bytes())
        .sum();
    bytes as f64 / 1e6
}

/// 토큰 `count`개를 쓰는 속도(토큰/초). 세 번 잰 가운데.
fn speed(model: &LlamaModel, prompt: &[TokenId], count: usize) -> Outcome<f64> {
    let mut rates = Vec::new();
    for _ in 0..3 {
        let mut cache = KvCache::new(&model.config);
        model.forward_cached(&prompt[..prompt.len() - 1], &mut cache)?;
        let start = Instant::now();
        let written = generate(model, prompt, count, &mut cache, |_| {})?;
        rates.push(written.len() as f64 / start.elapsed().as_secs_f64());
    }
    rates.sort_by(f64::total_cmp);
    Ok(rates[1])
}

fn main() -> Outcome<()> {
    let (base, lora) = (
        models_dir().join("smollm2_135m"),
        models_dir().join("smollm2_135m_lora"),
    );
    let needed = [
        base.join("model.safetensors"),
        lora.join("adapter_model.safetensors"),
    ];
    if let Some(missing) = needed.iter().find(|path| !path.exists()) {
        println!("건너뜀: {}이 없습니다", missing.display());
        return Ok(());
    }
    let tokenizer = Tokenizer::from_tokenizer_json(base.join("tokenizer.json"))?;
    let reference = SafeTensors::open(lora.join("reference.safetensors"))?;
    let ids: Vec<TokenId> = reference
        .tensor("ids")?
        .to_vec::<i32>()?
        .into_iter()
        .map(|id| id as TokenId)
        .collect();
    let expected: Vec<TokenId> = reference
        .tensor("written")?
        .to_vec::<i32>()?
        .into_iter()
        .map(|id| id as TokenId)
        .collect();
    let wanted = reference.tensor("logits")?.to_vec::<f32>()?;

    let adapter = Adapter::load(&lora)?;
    let plain = LlamaModel::load(&base)?;
    println!(
        "어댑터: 층 {}개, r = {}, 곱하는 수 {}, 값 {}개 (모델의 {:.2}%)",
        adapter.len(),
        adapter.rank,
        adapter.scale,
        adapter.parameter_count(),
        100.0 * adapter.parameter_count() as f64 / plain.parameter_count() as f64
    );
    println!(
        "질문: {:?}\n",
        reference
            .metadata()
            .get("question")
            .map_or("", String::as_str)
    );

    let mut beside = LlamaModel::load(&base)?;
    adapter.attach(&mut beside)?;
    let mut merged = LlamaModel::load(&base)?;
    let start = Instant::now();
    adapter.merge(&mut merged)?;
    let merging = start.elapsed().as_secs_f64() * 1e3;

    let mut passed = true;
    println!("방법              답");
    for (name, model, compare) in [
        ("어댑터 없이", &plain, false),
        ("따로 둔 채로", &beside, true),
        ("합쳐서", &merged, true),
    ] {
        let written = generate(
            model,
            &ids,
            expected.len(),
            &mut KvCache::new(&model.config),
            |_| {},
        )?;
        println!("{name:<12} {:?}", tokenizer.decode(&written));
        if !compare {
            continue;
        }
        let logits = model.forward_cached(&ids, &mut KvCache::new(&model.config))?;
        let logits = logits.as_slice::<f32>()?;
        let worst = logits
            .iter()
            .zip(&wanted)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        let same = written == expected;
        println!(
            "             PEFT와 견줌: 점수의 가장 큰 차이 {worst:.2e}, 쓴 토큰 {}개 {}",
            expected.len(),
            if same { "같음" } else { "다름" }
        );
        passed &= same && worst < 2e-3;
    }

    println!("\n방법              행렬 가중치(MB)  토큰/초");
    for (name, model) in [
        ("어댑터 없이", &plain),
        ("따로 둔 채로", &beside),
        ("합쳐서", &merged),
    ] {
        println!(
            "{name:<12} {:>10.0}      {:>8.1}",
            megabytes(model),
            speed(model, &ids, 64)?
        );
    }
    println!("합치는 데 {merging:.0}ms");
    if !passed {
        return Err("PEFT의 정답과 다릅니다".into());
    }
    println!("\n모두 맞았습니다.");
    Ok(())
}
