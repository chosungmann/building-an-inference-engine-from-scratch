//! 20장: 양자화한 가중치로 계산한 점수가 얼마나 달라지는지, llama.cpp와는
//! 얼마나 가까운지 본다.
//!
//! 실행: `cargo run --release -p badak_models --example ch20_accuracy -- [모델 디렉터리]`
//!
//! 먼저 준비할 것:
//!   sh scripts/build_llama_cpp.sh
//!   sh scripts/build_ch13_tools.sh
//!   sh scripts/convert_to_gguf.sh [모델 디렉터리]
//!   sh scripts/quantize_gguf.sh [모델 디렉터리]

use std::path::PathBuf;
use std::process::Command;

use badak_core::Tensor;
use badak_core::packed::{Format, Packed};
use badak_core::prob::{argmax, softmax};
use badak_formats::safetensors::SafeTensors;
use badak_models::llama::WeightFormat;
use badak_models::llama::cache::KvCache;
use badak_models::llama::generate::generate;
use badak_models::llama::model::LlamaModel;
use badak_models::models_dir;
use badak_models::tokenizer::{TokenId, Tokenizer};

const PROMPT: &str = "The capital of France is";
const NEW_TOKENS: usize = 40;

/// 가중치를 두는 방법과 계산하는 방법.
#[derive(Clone, Copy)]
enum Setup {
    /// 엔진이 그 방식으로 읽어 그 방식의 커널로 계산한다.
    Engine(WeightFormat),
    /// 묶었다가 `f32`로 되돌린 가중치를 `f32` 커널로 계산한다. 가중치의
    /// 오차만 있고 입력의 오차는 없다(19.8절의 방법).
    WeightsOnly(Format),
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let name = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "llama32_1b".to_string());
    let directory = models_dir().join(&name);
    let tool = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../third_party/llama.cpp/build/bin/ch13_llama_logits");
    let tokenizer = Tokenizer::from_tokenizer_json(directory.join("tokenizer.json"))?;
    let mut prompt = tokenizer.prefix().to_vec();
    prompt.extend(tokenizer.encode(PROMPT, false));

    // 기준: 원래 모델을 PyTorch가 f64로 계산한 점수(13.3절).
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
    // 두 점수 목록이 가장 많이 벗어난 정도를 가장 큰 점수에 견준 것.
    let distance = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
            .fold(0.0_f64, f64::max)
            / largest
    };
    let from_truth = |logits: &[f32]| {
        logits
            .iter()
            .zip(&truth)
            .map(|(&logit, &truth)| (f64::from(logit) - truth).abs())
            .fold(0.0_f64, f64::max)
            / largest
    };

    // ANCHOR: ours
    let ours = |setup: Setup| -> Result<(Vec<f32>, Vec<TokenId>), Box<dyn std::error::Error>> {
        let model = match setup {
            Setup::Engine(format) => LlamaModel::load_as(&directory, format)?,
            Setup::WeightsOnly(format) => {
                let mut model = LlamaModel::load_as(&directory, WeightFormat::F32)?;
                model.replace_matrices(|_, _, matrix| {
                    let (rows, cols) = (matrix.shape()[0], matrix.shape()[1]);
                    let values = matrix.to_f32()?.to_vec::<f32>()?;
                    let packed = Packed::quantize(format, &values, rows, cols)?;
                    let restored = packed.dequantize_rows(0..rows);
                    Ok(Tensor::from_vec(restored, matrix.shape())?.into())
                })?;
                model
            }
        };
        let logits = model.next_token_logits(&prompt)?.to_vec::<f32>()?;
        let mut cache = KvCache::new(&model.config);
        let generated = generate(&model, &prompt, NEW_TOKENS, &mut cache, |_| {})?;
        Ok((logits, generated))
    };
    // ANCHOR_END: ours

    // llama.cpp에 같은 토큰을 넣어 점수와 이어 쓴 토큰을 받아 온다(13.3절의 도구).
    let theirs = |file: &str| -> Result<(Vec<f32>, Vec<TokenId>), Box<dyn std::error::Error>> {
        let logits_path = std::env::temp_dir().join(format!("badak_ch20_{name}.bin"));
        let mut command = Command::new(&tool);
        command.arg(directory.join(file)).arg(&logits_path);
        // 16비트 실수를 거치지 않는 설정으로 맞춘다.
        command.arg(NEW_TOKENS.to_string()).arg("exact");
        command.args(prompt.iter().map(|id| id.to_string()));
        let output = command.output()?;
        if !output.status.success() {
            let message = String::from_utf8_lossy(&output.stderr).into_owned();
            return Err(format!("llama.cpp 도구가 실패했습니다: {message}").into());
        }
        let logits = std::fs::read(&logits_path)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_le_bytes(*bytes))
            .collect();
        let generated = String::from_utf8(output.stdout)?
            .split_whitespace()
            .map(str::parse)
            .collect::<Result<_, _>>()?;
        Ok((logits, generated))
    };

    println!("모델 {name}, 글 {PROMPT:?}, 이어 쓰는 토큰 {NEW_TOKENS}개");
    println!(
        "{:<40} {:>10} {:>10} {:>9} {:>10}",
        "엔진과 가중치", "상대 오차", "1위 토큰", "그 확률", "같은 토큰"
    );
    let (base_logits, base_generated) = ours(Setup::Engine(WeightFormat::F32))?;
    let first = argmax(&base_logits).unwrap_or(0);
    let line = |label: &str, logits: &[f32], generated: &[TokenId]| {
        let best = argmax(logits).unwrap_or(0);
        let same = generated
            .iter()
            .zip(&base_generated)
            .take_while(|(a, b)| a == b)
            .count();
        println!(
            "{:<40} {:>10.1e} {:>10} {:>8.2}% {:>10}",
            label,
            from_truth(logits),
            format!("{:?}", tokenizer.decode(&[best as TokenId])),
            softmax(logits)[first] * 100.0,
            format!("{same} / {NEW_TOKENS}")
        );
    };
    line("badak, f32로 펼침", &base_logits, &base_generated);
    let (logits, generated) = ours(Setup::Engine(WeightFormat::Stored))?;
    line("badak, BF16 그대로", &logits, &generated);
    let same_bits = logits
        .iter()
        .zip(&base_logits)
        .all(|(a, b)| a.to_bits() == b.to_bits());

    let mut pairs = Vec::new();
    for (label, format, engine, file) in [
        ("Q8_0", Format::Q8_0, WeightFormat::Q8_0, "model_q8_0.gguf"),
        ("Q4_0", Format::Q4_0, WeightFormat::Q4_0, "model_q4_0.gguf"),
    ] {
        let (weights_only, generated) = ours(Setup::WeightsOnly(format))?;
        line(
            &format!("badak, {label}, 입력은 f32"),
            &weights_only,
            &generated,
        );
        let (integer, generated) = ours(Setup::Engine(engine))?;
        line(
            &format!("badak, {label}, 입력도 8비트"),
            &integer,
            &generated,
        );
        let (llama, generated) = theirs(file)?;
        line(&format!("llama.cpp, {label}"), &llama, &generated);
        pairs.push((label, weights_only, integer, llama));
    }
    let (llama, generated) = theirs("model_q4_k_m.gguf")?;
    line("llama.cpp, Q4_K_M", &llama, &generated);

    println!();
    println!(
        "BF16 그대로 계산한 점수와 f32로 펼쳐 계산한 점수: {}",
        if same_bits {
            "모든 비트가 같다"
        } else {
            "다른 비트가 있다"
        }
    );
    println!();
    println!("같은 가중치로 계산한 점수끼리의 차이 (가장 큰 점수에 견준 것)");
    println!(
        "{:<8} {:>26} {:>26}",
        "가중치", "입력을 8비트로 바꾼 몫", "badak과 llama.cpp"
    );
    for (label, weights_only, integer, llama) in &pairs {
        println!(
            "{:<8} {:>26.1e} {:>26.1e}",
            label,
            distance(weights_only, integer),
            distance(integer, llama)
        );
    }
    Ok(())
}
