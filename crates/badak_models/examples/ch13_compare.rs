//! 13장: 같은 모델을 llama.cpp로 계산한 결과와 대조한다.
//!
//! 실행: `cargo run --release -p badak_models --example ch13_compare -- [모델 디렉터리]`
//!
//! 먼저 준비할 것:
//!   sh scripts/build_llama_cpp.sh
//!   sh scripts/build_ch13_tools.sh
//!   sh scripts/convert_to_gguf.sh [모델 디렉터리]
//!
//! 기준은 PyTorch가 f64로 계산한 점수다(`logits.safetensors`). 내려받기
//! 스크립트가 함께 받아 온다.

use std::path::PathBuf;
use std::process::Command;

use badak_core::prob::argmax;
use badak_formats::gguf::Gguf;
use badak_formats::safetensors::SafeTensors;
use badak_models::llama::LlamaFiles;
use badak_models::llama::cache::KvCache;
use badak_models::llama::generate::generate;
use badak_models::llama::model::LlamaModel;
use badak_models::models_dir;
use badak_models::tokenizer::{TokenId, Tokenizer};

const PROMPT: &str = "The capital of France is";
const NEW_TOKENS: usize = 40;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let name = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "llama32_1b".to_string());
    let directory = models_dir().join(&name);
    let gguf_path = directory.join("model_f32.gguf");
    let tool = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../third_party/llama.cpp/build/bin/ch13_llama_logits");

    let model = LlamaModel::load(&directory)?;
    let tokenizer = Tokenizer::from_tokenizer_json(directory.join("tokenizer.json"))?;
    let mut prompt = tokenizer.prefix().to_vec();
    prompt.extend(tokenizer.encode(PROMPT, false));
    println!("모델 {name}, 글 {PROMPT:?}, 토큰 {}개", prompt.len());

    // 1. 변환된 파일이 무엇을 담고 있는가.
    let gguf = Gguf::open(&gguf_path)?;
    let metadata = gguf.metadata();
    println!();
    println!("변환된 파일: 텐서 {}개", gguf.names().count());
    for key in [
        "general.architecture",
        "general.file_type",
        "tokenizer.ggml.pre",
        "tokenizer.ggml.add_bos_token",
        "tokenizer.ggml.bos_token_id",
    ] {
        match metadata.get(key) {
            Some(value) => println!("  {key:<32} {value:?}"),
            None => println!("  {key:<32} (없음)"),
        }
    }

    // ANCHOR: layout
    // 2. 질의를 만드는 가중치의 행 순서. 원본은 헤드마다 벡터를 앞 절반과 뒤
    //    절반으로 나누어 짝을 짓고, 변환된 파일은 이웃한 둘을 짝짓는다.
    let files = LlamaFiles::open(&directory)?;
    let ours = files.block(0)?.q_proj.to_f32()?;
    let theirs = gguf.tensor("blk.0.attn_q.weight")?;
    let (dim, half) = (model.config.head_dim(), model.config.head_dim() / 2);
    let row = |tensor: &badak_core::Tensor, index: usize| {
        tensor
            .slice(0, index..index + 1)
            .and_then(|row| row.to_vec::<f32>())
    };
    let mut same_order = true;
    let mut interleaved = true;
    for index in 0..ours.shape()[0] {
        let (head, within) = (index / dim, index % dim);
        // 변환된 파일의 `within`번째 행이 원본의 몇 번째 행에서 왔는가.
        let source = head * dim + within / 2 + (within % 2) * half;
        same_order &= row(&theirs, index)? == row(&ours, index)?;
        interleaved &= row(&theirs, index)? == row(&ours, source)?;
    }
    println!();
    println!("첫 블록의 질의 가중치 {:?}", theirs.shape());
    println!("  원본과 행 순서가 같다:                 {same_order}");
    println!("  앞 절반과 뒤 절반을 번갈아 놓은 순서다: {interleaved}");
    // ANCHOR_END: layout

    // ANCHOR: truth
    // 3. 기준: PyTorch가 f64로 계산한 점수. f32 둘로 나누어 저장되어 있다.
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
    let scale = truth
        .iter()
        .fold(0.0_f64, |max, value| max.max(value.abs()));
    // 어떤 엔진의 점수가 기준에서 가장 많이 벗어난 정도.
    let error = |logits: &[f32]| {
        logits
            .iter()
            .zip(&truth)
            .map(|(&logit, &truth)| (f64::from(logit) - truth).abs())
            .fold(0.0_f64, f64::max)
    };
    // ANCHOR_END: truth

    // ANCHOR: logits
    // llama.cpp에 같은 토큰을 넣어 마지막 줄의 점수와, 이어 쓴 토큰을 받아 온다.
    let run =
        |file: &str, mode: &str| -> Result<(Vec<f32>, Vec<TokenId>), Box<dyn std::error::Error>> {
            let logits_path = std::env::temp_dir().join(format!("badak_ch13_{name}.bin"));
            let mut command = Command::new(&tool);
            command.arg(directory.join(file)).arg(&logits_path);
            command.arg(NEW_TOKENS.to_string()).arg(mode);
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
    // ANCHOR_END: logits

    let ours = model.next_token_logits(&prompt)?.to_vec::<f32>()?;
    let mut cache = KvCache::new(&model.config);
    let ours_generated = generate(&model, &prompt, NEW_TOKENS, &mut cache, |_| {})?;
    let same_prefix = |other: &[TokenId]| {
        ours_generated
            .iter()
            .zip(other)
            .take_while(|(a, b)| a == b)
            .count()
    };

    println!();
    println!(
        "기준(f64)과의 차이. 점수 {}개, 가장 큰 점수 {scale:.3}",
        truth.len()
    );
    println!(
        "{:<38} {:>10} {:>10} {:>10} {:>12}",
        "엔진과 설정", "최대 오차", "상대 오차", "1위 토큰", "같은 토큰"
    );
    let line = |label: &str, logits: &[f32], generated: Option<&[TokenId]>| {
        let best = argmax(logits).map(|id| tokenizer.decode(&[id as TokenId]));
        println!(
            "{:<38} {:>10.2e} {:>10.2e} {:>10} {:>12}",
            label,
            error(logits),
            error(logits) / scale,
            format!("{:?}", best.unwrap_or_default()),
            generated.map_or("-".to_string(), |ids| format!(
                "{} / {NEW_TOKENS}",
                same_prefix(ids)
            ))
        );
    };
    line("badak (f32)", &ours, Some(&ours_generated));
    line(
        "PyTorch (f32)",
        &reference.tensor("pytorch_f32")?.to_vec::<f32>()?,
        None,
    );
    let variants = [
        (
            "llama.cpp, F32 파일, 16비트 없이",
            "model_f32.gguf",
            "exact",
        ),
        (
            "llama.cpp, F32 파일, 기본 설정",
            "model_f32.gguf",
            "default",
        ),
        (
            "llama.cpp, BF16 파일, 기본 설정",
            "model_bf16.gguf",
            "default",
        ),
        (
            "llama.cpp, Q4_K_M 파일, 기본 설정",
            "model_q4_k_m.gguf",
            "default",
        ),
    ];
    let mut ok = interleaved;
    for (index, (label, file, mode)) in variants.into_iter().enumerate() {
        let (logits, generated) = run(file, mode)?;
        line(label, &logits, Some(&generated));
        if index == 0 {
            // 16비트를 거치지 않는 설정은 연산 여러 개를 지난 결과의 허용 오차
            // (10.7절) 안에 들어와야 한다.
            ok &= error(&ours) / scale < 1e-4 && error(&logits) / scale < 1e-4;
            ok &= same_prefix(&generated) == NEW_TOKENS && argmax(&logits) == argmax(&ours);
        }
    }
    println!();
    println!("  {:?}", tokenizer.decode(&ours_generated));
    println!();
    println!(
        "{}",
        if ok {
            "llama.cpp와 일치합니다."
        } else {
            "llama.cpp와 다릅니다."
        }
    );
    if !ok {
        std::process::exit(1);
    }
    Ok(())
}
