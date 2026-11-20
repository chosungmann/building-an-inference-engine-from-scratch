//! 47장: 언어 모델 하나가 기기의 메모리를 얼마나 쓰는지 계산하고, 실제로
//! 돌려서 견준다.
//!
//! 실행: `cargo run --release -p badak_models --example ch47_budget -- predict [모델 ...]`
//!       `/usr/bin/time -l cargo run --release -p badak_models --example ch47_budget -- \
//!            run <모델> <stored|f32|q8_0|q4_0> <f32|f16> <토큰 수>`
//!
//! `predict`는 설정 파일과 가중치의 형상만 보고 계산한다. `run`은 그 설정으로
//! 모델을 올려 토큰을 그만큼 읽고, 계산한 값을 적는다. 프로세스의 메모리는
//! 밖에서 `/usr/bin/time -l`로 잰다(43장).

use badak_models::llama::WeightFormat;
use badak_models::llama::cache::{KvCache, KvPrecision};
use badak_models::llama::model::LlamaModel;
use badak_models::models_dir;
use badak_models::tokenizer::Tokenizer;

/// 모델의 행렬 가중치에 든 값의 수.
fn weights(model: &mut LlamaModel) -> Result<usize, Box<dyn std::error::Error>> {
    let mut count = 0;
    model.replace_matrices(|_, _, matrix| {
        count += matrix.shape().iter().product::<usize>();
        Ok(matrix.clone())
    })?;
    Ok(count)
}

// ANCHOR: budget
/// 가중치 한 벌이 차지하는 바이트. `Q4_0`은 값 32개에 18바이트다(20장).
fn weight_bytes(values: usize, format: WeightFormat) -> usize {
    match format {
        WeightFormat::F32 => values * 4,
        WeightFormat::Stored => values * 2,
        WeightFormat::Q8_0 => values / 32 * 34,
        WeightFormat::Q4_0 => values / 32 * 18,
    }
}

/// KV 캐시가 차지하는 바이트. 블록마다 키와 값, 토큰마다 `키·값 헤드의 수 ×
/// 헤드의 길이`개의 값이다(11장).
fn cache_bytes(
    layers: usize,
    kv_heads: usize,
    head_dim: usize,
    tokens: usize,
    precision: KvPrecision,
) -> usize {
    let element = match precision {
        KvPrecision::F32 => 4,
        KvPrecision::F16 => 2,
    };
    layers * 2 * kv_heads * head_dim * tokens * element
}
// ANCHOR_END: budget

fn megabytes(bytes: usize) -> f64 {
    bytes as f64 / 1e6
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let part = arguments
        .next()
        .ok_or("부분(predict, run, speed)을 주십시오")?;
    match part.as_str() {
        "predict" => {
            let mut names: Vec<String> = arguments.collect();
            if names.is_empty() {
                names = ["smollm2_135m", "smollm2_360m", "llama32_1b", "smollm2_1_7b"]
                    .map(String::from)
                    .to_vec();
            }
            for name in names {
                let directory = models_dir().join(&name);
                let mut model = LlamaModel::load_as(&directory, WeightFormat::Stored)?;
                let values = weights(&mut model)?;
                let config = &model.config;
                let per_token = cache_bytes(
                    config.num_layers,
                    config.num_kv_heads,
                    config.head_dim(),
                    1,
                    KvPrecision::F32,
                );
                println!(
                    "predict {name} 값 {values} 블록 {} 키값헤드 {} 헤드길이 {} 16비트 {:.0} Q8_0 {:.0} Q4_0 {:.0} 토큰당캐시 {}",
                    config.num_layers,
                    config.num_kv_heads,
                    config.head_dim(),
                    megabytes(weight_bytes(values, WeightFormat::Stored)),
                    megabytes(weight_bytes(values, WeightFormat::Q8_0)),
                    megabytes(weight_bytes(values, WeightFormat::Q4_0)),
                    per_token
                );
            }
        }
        "run" => {
            let name = arguments.next().ok_or("모델을 주십시오")?;
            let format = match arguments.next().as_deref() {
                Some("f32") => WeightFormat::F32,
                Some("q8_0") => WeightFormat::Q8_0,
                Some("q4_0") => WeightFormat::Q4_0,
                _ => WeightFormat::Stored,
            };
            let precision = match arguments.next().as_deref() {
                Some("f16") => KvPrecision::F16,
                _ => KvPrecision::F32,
            };
            let tokens: usize = arguments.next().map_or(Ok(512), |text| text.parse())?;
            let directory = models_dir().join(&name);
            let mut model = LlamaModel::load_as(&directory, format)?;
            let values = weights(&mut model)?;
            let tokenizer = Tokenizer::from_tokenizer_json(directory.join("tokenizer.json"))?;
            let mut prompt = tokenizer.prefix().to_vec();
            prompt.extend(
                tokenizer.encode("Once upon a time, in a small village by the sea,", false),
            );
            let long: Vec<_> = prompt.iter().copied().cycle().take(tokens).collect();
            let mut cache = KvCache::with_precision(&model.config, precision);
            // 한꺼번에 읽으면 중간 결과가 토큰 수만큼 커진다. 128개씩 끊어 읽는다.
            for piece in long.chunks(128) {
                model.forward_cached(piece, &mut cache)?;
            }
            let config = &model.config;
            let predicted = cache_bytes(
                config.num_layers,
                config.num_kv_heads,
                config.head_dim(),
                tokens,
                precision,
            );
            println!(
                "run {name} {} {} 토큰 {tokens} 가중치 {} 캐시계산 {} 캐시실제 {}",
                match format {
                    WeightFormat::F32 => "f32",
                    WeightFormat::Stored => "stored",
                    WeightFormat::Q8_0 => "q8_0",
                    WeightFormat::Q4_0 => "q4_0",
                },
                match precision {
                    KvPrecision::F32 => "f32",
                    KvPrecision::F16 => "f16",
                },
                weight_bytes(values, format),
                predicted,
                cache.size_in_bytes()
            );
        }
        "speed" => {
            // 다른 엔진들과 견줄 숫자: 프롬프트 256토큰을 읽는 속도와 토큰
            // 128개를 쓰는 속도. 세 번 재서 가운데 값을 적는다.
            let name = arguments.next().ok_or("모델을 주십시오")?;
            let (label, format) = match arguments.next().as_deref() {
                Some("f32") => ("f32", WeightFormat::F32),
                Some("q8_0") => ("q8_0", WeightFormat::Q8_0),
                Some("q4_0") => ("q4_0", WeightFormat::Q4_0),
                _ => ("stored", WeightFormat::Stored),
            };
            let directory = models_dir().join(&name);
            let started = std::time::Instant::now();
            let model = LlamaModel::load_as(&directory, format)?;
            let loaded = started.elapsed().as_secs_f64() * 1e3;
            let tokenizer = Tokenizer::from_tokenizer_json(directory.join("tokenizer.json"))?;
            let mut prompt = tokenizer.prefix().to_vec();
            prompt.extend(
                tokenizer.encode("Once upon a time, in a small village by the sea,", false),
            );
            let long: Vec<_> = prompt.iter().copied().cycle().take(256).collect();
            let (mut reading, mut writing) = (Vec::new(), Vec::new());
            for _ in 0..3 {
                let mut cache = KvCache::new(&model.config);
                let started = std::time::Instant::now();
                badak_models::llama::generate::generate(&model, &long, 1, &mut cache, |_| {})?;
                reading.push(long.len() as f64 / started.elapsed().as_secs_f64());
                let mut first = None;
                let mut cache = KvCache::new(&model.config);
                let written = badak_models::llama::generate::generate(
                    &model,
                    &prompt,
                    128,
                    &mut cache,
                    |_| {
                        first.get_or_insert_with(std::time::Instant::now);
                    },
                )?;
                writing.push(
                    (written.len().max(2) - 1) as f64
                        / first.map_or(1.0, |at| at.elapsed().as_secs_f64()),
                );
            }
            reading.sort_by(f64::total_cmp);
            writing.sort_by(f64::total_cmp);
            println!(
                "speed badak {name} {label} 올리기 {loaded:.0} 읽기 {:.0} 쓰기 {:.0}",
                reading[1], writing[1]
            );
        }
        other => return Err(format!("모르는 부분 {other}").into()),
    }
    Ok(())
}
