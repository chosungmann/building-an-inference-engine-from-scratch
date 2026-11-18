//! 21장: 가중치를 줄이는 방법마다 퍼플렉서티를 잰다.
//!
//! 실행: `cargo run --release -p badak_models --example ch21_perplexity -- [모델] [조각 수] [묶음]`
//!
//! 묶음은 `formats`(저장 방식), `kinds`(가중치의 종류마다 하나씩만 줄이기),
//! `blocks`(블록마다 하나씩만 줄이기), `kv`(KV 캐시의 정밀도), `base`(원래
//! 모델만) 가운데 하나다.
//!
//! 먼저 준비할 것:
//!   sh scripts/download_wikitext2.sh
//!   sh scripts/convert_to_gguf.sh [모델]   (formats 묶음의 Q4_K_M 줄에만 필요)

use std::time::Instant;

use badak_core::Tensor;
use badak_core::packed::{
    Format, Matrix, Packed, Q4_K_BYTES, Q4_K_VALUES, Q6_K_BYTES, dequantize_q4_k, dequantize_q6_k,
};
use badak_core::quant::{Granularity, fake_quantize};
use badak_formats::gguf::{GgmlType, Gguf};
use badak_models::llama::WeightFormat;
use badak_models::llama::cache::KvPrecision;
use badak_models::llama::model::LlamaModel;
use badak_models::llama::perplexity::{Losses, Protocol, evaluate};
use badak_models::models_dir;
use badak_models::tokenizer::Tokenizer;

const CONTEXT: usize = 512;

/// 가중치를 두는 방법.
#[derive(Clone)]
enum Setup {
    /// 엔진이 그 방식으로 읽어 그 방식의 커널로 계산한다.
    Engine(WeightFormat),
    /// 묶었다가 `f32`로 되돌린 가중치. 입력은 줄이지 않는다.
    WeightsOnly(Format),
    /// 19장의 대칭 눈금. 정수 `bits`비트, 눈금은 값 `per`개마다(0이면 텐서마다,
    /// 1이면 행마다).
    Uniform(u32, usize),
    /// llama.cpp가 만든 K-양자화 파일의 가중치를 되돌린 것. 입력은 줄이지 않는다.
    Gguf(&'static str),
    /// 이름이 `kind`이고(비었으면 모두) 블록이 `layer`인(없으면 모두) 가중치만
    /// `Q4_0`으로 줄인다. 나머지는 그대로다.
    Only(&'static str, Option<usize>),
}

// ANCHOR: gguf
/// GGUF 파일의 K-양자화 텐서를 `f32`로 되돌려 우리 모델의 순서로 놓는다.
fn from_gguf(
    gguf: &Gguf,
    kind: &str,
    layer: Option<usize>,
    shape: &[usize],
    head_dim: usize,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let part = match kind {
        "embedding" => "token_embd.weight".to_string(),
        "q_proj" => "attn_q.weight".to_string(),
        "k_proj" => "attn_k.weight".to_string(),
        "v_proj" => "attn_v.weight".to_string(),
        "o_proj" => "attn_output.weight".to_string(),
        "gate_proj" => "ffn_gate.weight".to_string(),
        "up_proj" => "ffn_up.weight".to_string(),
        _ => "ffn_down.weight".to_string(),
    };
    let name = match layer {
        Some(layer) => format!("blk.{layer}.{part}"),
        None => part,
    };
    let info = gguf.info(&name).ok_or("파일에 없는 텐서")?;
    let (rows, cols) = (shape[0], shape[1]);
    let count = rows * cols;
    let bytes = gguf.bytes_from(&name)?;
    let mut values = vec![0.0_f32; count];
    match info.ggml_type {
        // 번호는 llama.cpp의 `ggml_type`: 12가 Q4_K, 14가 Q6_K.
        GgmlType::Other(12) => {
            dequantize_q4_k(&bytes[..count / Q4_K_VALUES * Q4_K_BYTES], &mut values);
        }
        GgmlType::Other(14) => {
            dequantize_q6_k(&bytes[..count / Q4_K_VALUES * Q6_K_BYTES], &mut values);
        }
        other => return Err(format!("{name}: 되돌릴 줄 모르는 형식 {other:?}").into()),
    }
    // 질의와 키는 파일에서 행의 순서가 다르다(13.3절). 원래 순서로 되돌린다.
    if kind == "q_proj" || kind == "k_proj" {
        let half = head_dim / 2;
        let mut original = vec![0.0_f32; count];
        for row in 0..rows {
            let (head, within) = (row / head_dim, row % head_dim);
            let source = head * head_dim + within / 2 + (within % 2) * half;
            original[source * cols..(source + 1) * cols]
                .copy_from_slice(&values[row * cols..(row + 1) * cols]);
        }
        values = original;
    }
    Ok(Tensor::from_vec(values, shape)?)
}
// ANCHOR_END: gguf

/// 방법대로 모델을 읽는다. 값 하나에 든 비트도 함께 돌려준다.
fn load(
    directory: &std::path::Path,
    setup: &Setup,
) -> Result<(LlamaModel, f64), Box<dyn std::error::Error>> {
    let mut model = match setup {
        Setup::Engine(format) => LlamaModel::load_as(directory, *format)?,
        _ => LlamaModel::load_as(directory, WeightFormat::F32)?,
    };
    let head_dim = model.config.head_dim();
    let gguf = match setup {
        Setup::Gguf(file) => Some(Gguf::open(directory.join(file))?),
        _ => None,
    };
    let (mut bits, mut count) = (0.0_f64, 0_usize);
    let mut failure = None;
    model.replace_matrices(|kind, layer, matrix| {
        let shape = matrix.shape().to_vec();
        let (rows, cols) = (shape[0], shape[1]);
        let values = || matrix.to_f32().and_then(|tensor| tensor.to_vec::<f32>());
        let (changed, per_weight): (Matrix, f64) = match setup {
            Setup::Engine(format) => {
                let per_weight = match format {
                    WeightFormat::F32 => 32.0,
                    WeightFormat::Stored => 16.0,
                    WeightFormat::Q8_0 => 8.5,
                    WeightFormat::Q4_0 => 4.5,
                };
                (matrix.clone(), per_weight)
            }
            Setup::WeightsOnly(format) => {
                let packed = Packed::quantize(*format, &values()?, rows, cols)?;
                let restored = Tensor::from_vec(packed.dequantize_rows(0..rows), &shape)?;
                (restored.into(), format.bits_per_weight())
            }
            Setup::Uniform(bits, per) => {
                let mut values = values()?;
                let (granularity, scale_bits) = match per {
                    0 => (Granularity::Tensor, 0.0),
                    1 => (Granularity::Blocks(cols), 16.0 / cols as f64),
                    per => (Granularity::Blocks(*per), 16.0 / *per as f64),
                };
                fake_quantize(&mut values, *bits, granularity);
                (
                    Tensor::from_vec(values, &shape)?.into(),
                    f64::from(*bits) + scale_bits,
                )
            }
            Setup::Gguf(_) => {
                let gguf = gguf.as_ref().expect("위에서 열었다");
                match from_gguf(gguf, kind, layer, &shape, head_dim) {
                    Ok(tensor) => {
                        let name_bits = gguf_bits(gguf, kind, layer).unwrap_or(32.0);
                        (tensor.into(), name_bits)
                    }
                    Err(error) => {
                        failure = Some(error.to_string());
                        (matrix.clone(), 32.0)
                    }
                }
            }
            Setup::Only(only_kind, only_layer) => {
                let chosen = (only_kind.is_empty() || *only_kind == kind)
                    && (only_layer.is_none() || *only_layer == layer);
                if chosen {
                    let packed = Packed::quantize(Format::Q4_0, &values()?, rows, cols)?;
                    (packed.into(), 4.5)
                } else {
                    (matrix.clone(), 16.0)
                }
            }
        };
        bits += per_weight * (rows * cols) as f64;
        count += rows * cols;
        Ok(changed)
    })?;
    if let Some(message) = failure {
        return Err(message.into());
    }
    Ok((model, bits / count as f64))
}

/// GGUF 파일에서 그 텐서의 값 하나에 든 비트.
fn gguf_bits(gguf: &Gguf, kind: &str, layer: Option<usize>) -> Option<f64> {
    let part = match kind {
        "embedding" => "token_embd.weight".to_string(),
        "q_proj" => "attn_q.weight".to_string(),
        "k_proj" => "attn_k.weight".to_string(),
        "v_proj" => "attn_v.weight".to_string(),
        "o_proj" => "attn_output.weight".to_string(),
        "gate_proj" => "ffn_gate.weight".to_string(),
        "up_proj" => "ffn_up.weight".to_string(),
        _ => "ffn_down.weight".to_string(),
    };
    let name = match layer {
        Some(layer) => format!("blk.{layer}.{part}"),
        None => part,
    };
    match gguf.info(&name)?.ggml_type {
        GgmlType::Other(12) => Some(Q4_K_BYTES as f64 * 8.0 / Q4_K_VALUES as f64),
        GgmlType::Other(14) => Some(Q6_K_BYTES as f64 * 8.0 / Q4_K_VALUES as f64),
        _ => None,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let name = arguments.next().unwrap_or_else(|| "llama32_1b".to_string());
    let chunks: usize = arguments.next().map_or(Ok(40), |text| text.parse())?;
    let suite = arguments.next().unwrap_or_else(|| "formats".to_string());
    let directory = models_dir().join(&name);

    // 평가할 글을 토큰으로 바꾼다. 글머리 토큰을 앞에 붙인다.
    let text =
        std::fs::read_to_string(models_dir().join("wikitext2/wikitext-2-raw/wiki.test.raw"))?;
    let tokenizer = Tokenizer::from_tokenizer_json(directory.join("tokenizer.json"))?;
    let start = Instant::now();
    let mut tokens = tokenizer.prefix().to_vec();
    tokens.extend(tokenizer.encode(&text, false));
    let protocol = Protocol {
        context: CONTEXT,
        chunks,
        // 글 앞에 붙이는 토큰이 있는 모델이면 조각마다 그것으로 시작한다.
        bos: tokenizer.prefix().first().copied(),
        kv: KvPrecision::F32,
    };
    println!(
        "모델 {name}, WikiText-2 시험 글 {}바이트, 토큰 {}개 (토큰화 {:.1}초)",
        text.len(),
        tokens.len(),
        start.elapsed().as_secs_f64()
    );
    println!(
        "한 번에 {CONTEXT}토큰씩 {}조각, 조각마다 뒤의 {}개를 채점",
        chunks.min(tokens.len() / CONTEXT),
        CONTEXT - 1 - CONTEXT / 2
    );
    println!();

    let half = KvPrecision::F16;
    let setups: Vec<(String, Setup, KvPrecision)> = match suite.as_str() {
        "formats" => [
            ("f32로 펼침", Setup::Engine(WeightFormat::F32)),
            ("BF16 그대로", Setup::Engine(WeightFormat::Stored)),
            ("Q8_0, 입력은 f32", Setup::WeightsOnly(Format::Q8_0)),
            ("Q8_0, 입력도 8비트", Setup::Engine(WeightFormat::Q8_0)),
            ("6비트, 32개마다", Setup::Uniform(6, 32)),
            ("5비트, 32개마다", Setup::Uniform(5, 32)),
            (
                "Q4_K_M 파일 (llama.cpp가 만듦)",
                Setup::Gguf("model_q4_k_m.gguf"),
            ),
            ("Q4_0, 입력은 f32", Setup::WeightsOnly(Format::Q4_0)),
            ("Q4_0, 입력도 8비트", Setup::Engine(WeightFormat::Q4_0)),
            ("4비트(정수 15가지), 32개마다", Setup::Uniform(4, 32)),
            ("4비트, 행마다", Setup::Uniform(4, 1)),
            ("3비트, 32개마다", Setup::Uniform(3, 32)),
            ("8비트, 텐서마다", Setup::Uniform(8, 0)),
            ("4비트, 텐서마다", Setup::Uniform(4, 0)),
        ]
        .into_iter()
        .map(|(label, setup)| (label.to_string(), setup, KvPrecision::F32))
        .collect(),
        "kinds" => std::iter::once(("BF16 그대로 (기준)", Setup::Engine(WeightFormat::Stored)))
            .chain(
                [
                    "embedding",
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ]
                .into_iter()
                .map(|kind| (kind, Setup::Only(kind, None))),
            )
            .chain(std::iter::once(("모두", Setup::Only("", None))))
            .map(|(label, setup)| (label.to_string(), setup, KvPrecision::F32))
            .collect(),
        "base" => vec![(
            "BF16 그대로".to_string(),
            Setup::Engine(WeightFormat::Stored),
            KvPrecision::F32,
        )],
        "blocks" => std::iter::once((
            "BF16 그대로 (기준)".to_string(),
            Setup::Engine(WeightFormat::Stored),
            KvPrecision::F32,
        ))
        .chain((0..16).map(|layer| {
            (
                format!("블록 {layer}"),
                Setup::Only("", Some(layer)),
                KvPrecision::F32,
            )
        }))
        .collect(),
        _ => vec![
            (
                "BF16, 캐시 f32".to_string(),
                Setup::Engine(WeightFormat::Stored),
                KvPrecision::F32,
            ),
            (
                "BF16, 캐시 f16".to_string(),
                Setup::Engine(WeightFormat::Stored),
                half,
            ),
            (
                "Q4_0, 캐시 f32".to_string(),
                Setup::Engine(WeightFormat::Q4_0),
                KvPrecision::F32,
            ),
            (
                "Q4_0, 캐시 f16".to_string(),
                Setup::Engine(WeightFormat::Q4_0),
                half,
            ),
        ],
    };

    println!(
        "{:<32} {:>9} {:>16} {:>9} {:>20} {:>7}",
        "가중치", "값당 비트", "퍼플렉서티", "평균 손실", "기준과의 손실 차이", "초"
    );
    let mut base: Option<Losses> = None;
    for (label, setup, kv) in setups {
        let start = Instant::now();
        let (model, bits) = load(&directory, &setup)?;
        let losses = evaluate(&model, &tokens, Protocol { kv, ..protocol })?;
        let seconds = start.elapsed().as_secs_f64();
        let base = base.get_or_insert_with(|| losses.clone());
        let (difference, error) = losses.difference_from(base);
        let perplexity = losses.perplexity();
        println!(
            "{:<32} {:>9.2} {:>16} {:>9.4} {:>20} {:>7.0}",
            label,
            bits,
            format!(
                "{:.3} ± {:.3}",
                perplexity,
                perplexity * losses.standard_error()
            ),
            losses.mean(),
            format!("{difference:+.4} ± {error:.4}"),
            seconds
        );
    }
    Ok(())
}
