//! 19장: 가중치를 더 적은 비트로 나타내면 모델의 출력이 얼마나 달라지는지 본다.
//!
//! 실행: `cargo run --release -p badak_models --example ch19_effect -- [모델 디렉터리]`
//!
//! 가중치를 바꿨다가 `f32`로 되돌려 놓고 평소처럼 계산한다. 저장 공간도
//! 속도도 그대로다. 달라지는 것은 값뿐이다.

use badak_core::Tensor;
use badak_core::prob::{argmax, softmax};
use badak_core::quant::{Quantizer, bf16_from_f32, bf16_to_f32, f16_from_f32, f16_to_f32};
use badak_core::threads;
use badak_formats::safetensors::SafeTensors;
use badak_models::llama::cache::KvCache;
use badak_models::llama::generate::generate;
use badak_models::llama::model::LlamaModel;
use badak_models::models_dir;
use badak_models::tokenizer::{TokenId, Tokenizer};

const PROMPT: &str = "The capital of France is";
const NEW_TOKENS: usize = 40;
/// 스케일을 저장하는 데 드는 비트(16비트 실수 하나).
const SCALE_BITS: f64 = 16.0;

// ANCHOR: scheme
/// 가중치를 나타내는 방법.
#[derive(Clone, Copy)]
enum Scheme {
    /// 그대로 둔다.
    F32,
    /// 16비트 실수 두 가지.
    Bf16,
    F16,
    /// 정수 `bits`비트. 스케일을 텐서마다, 행마다, 또는 값 `n`개마다 정한다.
    PerTensor(u32),
    PerRow(u32),
    PerBlock(u32, usize),
}

impl Scheme {
    /// 값 하나에 드는 비트. 스케일을 저장하는 몫을 더한다.
    fn bits_per_weight(self, cols: usize) -> f64 {
        match self {
            Self::F32 => 32.0,
            Self::Bf16 | Self::F16 => 16.0,
            Self::PerTensor(bits) => f64::from(bits),
            Self::PerRow(bits) => f64::from(bits) + SCALE_BITS / cols as f64,
            Self::PerBlock(bits, size) => f64::from(bits) + SCALE_BITS / size as f64,
        }
    }

    /// 값들을 이 방법으로 바꿨다가 `f32`로 되돌려 놓는다.
    fn apply(self, values: &mut [f32], cols: usize) {
        let pool = threads::global().pool();
        let threads = pool.threads();
        // 조각의 경계가 행과 덩어리의 경계에 맞도록 열의 수의 배수로 자른다.
        let piece = cols * 32;
        let (bits, block) = match self {
            Self::F32 => return,
            Self::Bf16 | Self::F16 => {
                let convert: fn(f32) -> f32 = if matches!(self, Self::F16) {
                    |value| f16_to_f32(f16_from_f32(value))
                } else {
                    |value| bf16_to_f32(bf16_from_f32(value))
                };
                pool.run_chunks(values, piece, threads, &|_, chunk| {
                    chunk.iter_mut().for_each(|value| *value = convert(*value));
                });
                return;
            }
            Self::PerTensor(bits) => {
                // 스케일 하나를 먼저 정하고 모든 값에 쓴다.
                let quantizer = Quantizer::symmetric_for(values, bits);
                pool.run_chunks(values, piece, threads, &|_, chunk| {
                    for value in chunk {
                        *value = quantizer.round_trip(*value);
                    }
                });
                return;
            }
            Self::PerRow(bits) => (bits, cols),
            Self::PerBlock(bits, size) => (bits, size),
        };
        pool.run_chunks(values, piece, threads, &|_, chunk| {
            for block in chunk.chunks_mut(block) {
                let quantizer = Quantizer::symmetric_for(block, bits);
                for value in block {
                    *value = quantizer.round_trip(*value);
                }
            }
        });
    }
}
// ANCHOR_END: scheme

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let name = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "llama32_1b".to_string());
    let directory = models_dir().join(&name);
    let tokenizer = Tokenizer::from_tokenizer_json(directory.join("tokenizer.json"))?;
    let mut prompt = tokenizer.prefix().to_vec();
    prompt.extend(tokenizer.encode(PROMPT, false));

    // 기준: PyTorch가 f64로 계산한 점수(13.3절).
    let reference = SafeTensors::open(directory.join("logits.safetensors"))?;
    let (hi, lo) = (
        reference.tensor("truth_hi")?.to_vec::<f32>()?,
        reference.tensor("truth_lo")?.to_vec::<f32>()?,
    );
    let truth: Vec<f64> = hi
        .iter()
        .zip(&lo)
        .map(|(&hi, &lo)| f64::from(hi) + f64::from(lo))
        .collect();
    let largest = truth.iter().fold(0.0_f64, |max, v| max.max(v.abs()));

    let schemes = [
        ("f32 (그대로)", Scheme::F32),
        ("BF16", Scheme::Bf16),
        ("f16", Scheme::F16),
        ("8비트, 텐서마다", Scheme::PerTensor(8)),
        ("8비트, 행마다", Scheme::PerRow(8)),
        ("8비트, 32개마다", Scheme::PerBlock(8, 32)),
        ("6비트, 32개마다", Scheme::PerBlock(6, 32)),
        ("5비트, 32개마다", Scheme::PerBlock(5, 32)),
        ("4비트, 텐서마다", Scheme::PerTensor(4)),
        ("4비트, 행마다", Scheme::PerRow(4)),
        ("4비트, 32개마다", Scheme::PerBlock(4, 32)),
        ("3비트, 32개마다", Scheme::PerBlock(3, 32)),
        ("2비트, 32개마다", Scheme::PerBlock(2, 32)),
    ];

    println!("모델 {name}, 글 {PROMPT:?}, 이어 쓰는 토큰 {NEW_TOKENS}개");
    println!(
        "{:<20} {:>12} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "가중치", "값당 비트", "크기(GB)", "상대 오차", "1위 토큰", "그 확률", "같은 토큰"
    );
    let mut baseline: Option<(usize, Vec<TokenId>)> = None;
    let mut texts = Vec::new();
    for (label, scheme) in schemes {
        // ANCHOR: run
        let mut model = LlamaModel::load(&directory)?;
        let mut bits = 0.0;
        let mut count = 0;
        model.replace_matrices(|_, _, tensor| {
            let cols = tensor.shape()[1];
            // 20장부터 가중치가 16비트로 저장되어 있을 수 있어 `f32`로 꺼낸다.
            let mut values = tensor.to_f32()?.to_vec::<f32>()?;
            scheme.apply(&mut values, cols);
            bits += scheme.bits_per_weight(cols) * values.len() as f64;
            count += values.len();
            Ok(Tensor::from_vec(values, tensor.shape())?.into())
        })?;
        let logits = model.next_token_logits(&prompt)?.to_vec::<f32>()?;
        let mut cache = KvCache::new(&model.config);
        let generated = generate(&model, &prompt, NEW_TOKENS, &mut cache, |_| {})?;
        // ANCHOR_END: run

        let error = logits
            .iter()
            .zip(&truth)
            .map(|(&logit, &truth)| (f64::from(logit) - truth).abs())
            .fold(0.0_f64, f64::max);
        let best = argmax(&logits).unwrap_or(0);
        // 견줄 대상은 f32 그대로일 때의 1위 토큰과 이어 쓴 글이다.
        let (first, expected) = baseline.get_or_insert((best, generated.clone())).clone();
        let probability = softmax(&logits)[first];
        let same = generated
            .iter()
            .zip(&expected)
            .take_while(|(a, b)| a == b)
            .count();
        println!(
            "{:<20} {:>12.2} {:>10.2} {:>10.1e} {:>10} {:>9.2}% {:>10}",
            label,
            bits / count as f64,
            bits / 8.0 / 1e9,
            error / largest,
            format!("{:?}", tokenizer.decode(&[best as TokenId])),
            probability * 100.0,
            format!("{same} / {NEW_TOKENS}")
        );
        texts.push((label, tokenizer.decode(&generated)));
    }
    println!();
    println!(
        "\"그 확률\"은 f32일 때의 1위 토큰에 준 확률, \"같은 토큰\"은 f32일 때와 같은 글이 이어진 길이다."
    );
    println!();
    println!("이어 쓴 글");
    for (label, text) in texts {
        println!("{label:<20} {text:?}");
    }
    Ok(())
}
