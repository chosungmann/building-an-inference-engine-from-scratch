//! 23장: 언어 모델의 선형 층을 GPU에 맡긴다.
//!
//! 실행: `cargo run --release -p badak_models --example ch23_llama -- [모델 디렉터리]`
//!
//! 순전파의 코드는 그대로 두고 가중치 행렬만 GPU에 올린 것으로 바꿔 끼운다.
//! 점수가 얼마나 달라지는지, 속도가 어떻게 되는지, 시간이 어디에 쓰이는지 본다.

use std::sync::Arc;
use std::time::Instant;

use badak_bench::stats::Summary;
use badak_core::prob::{argmax, softmax};
use badak_core::profile;
use badak_core::random::Pcg64;
use badak_formats::safetensors::SafeTensors;
use badak_gpu::Gpu;
use badak_gpu::offload::Offload;
use badak_models::llama::cache::KvCache;
use badak_models::llama::generate::generate;
use badak_models::llama::model::LlamaModel;
use badak_models::models_dir;
use badak_models::tokenizer::{TokenId, Tokenizer};

const PROMPT: &str = "The capital of France is";
const NEW_TOKENS: usize = 40;
const PROMPT_TOKENS: usize = 256;
const GENERATED: usize = 32;

// ANCHOR: offload
/// 모델을 읽고, 블록의 선형 층 일곱 개씩을 GPU에 맡긴다. 임베딩 표는 그대로
/// 둔다. 표에서 행을 뽑는 일과 마지막에 점수를 내는 일은 CPU가 한다.
///
/// `whole_mlp`가 참이면 MLP의 세 층은 하나씩 맡기지 않고 MLP를 통째로 맡긴다.
fn load_on_gpu(
    directory: &std::path::Path,
    offload: &Arc<Offload>,
    whole_mlp: bool,
) -> Result<(LlamaModel, f64, usize), Box<dyn std::error::Error>> {
    let mut model = LlamaModel::load(directory)?;
    let start = Instant::now();
    let mut bytes = 0;
    if whole_mlp {
        model.override_mlps(|_, gate, up, down| {
            bytes += (gate.len() + up.len() + down.len()) * size_of::<f32>();
            Ok(offload.mlp(gate, up, down)?)
        })?;
    }
    model.replace_matrices(|kind, _, matrix| {
        let in_mlp = matches!(kind, "gate_proj" | "up_proj" | "down_proj");
        if kind == "embedding" || (whole_mlp && in_mlp) {
            return Ok(matrix.clone());
        }
        bytes += matrix.len() * size_of::<f32>();
        Ok(offload.matrix(matrix.clone())?)
    })?;
    Ok((model, start.elapsed().as_secs_f64(), bytes))
}
// ANCHOR_END: offload

/// 프롬프트 256토큰을 한 번 처리하면서 시간이 어디에 쓰이는지 적는다(14.7절).
fn profile_prompt(
    label: &str,
    model: &LlamaModel,
    tokens: &[TokenId],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut cache = KvCache::new(&model.config);
    model.forward_cached(&tokens[..PROMPT_TOKENS], &mut cache)?;
    profile::reset();
    profile::enable(true);
    let mut cache = KvCache::new(&model.config);
    let start = Instant::now();
    model.forward_cached(&tokens[..PROMPT_TOKENS], &mut cache)?;
    let seconds = start.elapsed().as_secs_f64();
    profile::enable(false);
    println!();
    println!("프롬프트 256토큰을 한 번 처리하는 {seconds:.3}초가 쓰인 곳: {label}");
    let mut by_name: Vec<(&str, f64, u64)> = Vec::new();
    let records = profile::report();
    for (path, record) in &records {
        let name = path.rsplit('/').next().unwrap_or(path);
        match by_name
            .iter_mut()
            .find(|(existing, _, _)| *existing == name)
        {
            Some((_, seconds, calls)) => {
                *seconds += record.own.as_secs_f64();
                *calls += record.calls;
            }
            None => by_name.push((name, record.own.as_secs_f64(), record.calls)),
        }
    }
    by_name.sort_by(|a, b| b.1.total_cmp(&a.1));
    let measured: f64 = by_name.iter().map(|(_, seconds, _)| seconds).sum();
    println!(
        "{:<34} {:>8} {:>12} {:>8}",
        "일", "횟수", "시간 (ms)", "비중"
    );
    for (name, seconds, calls) in &by_name {
        let label = match *name {
            "gpu_compute" => "GPU: 명령을 보내고 기다리기",
            "gpu_upload" => "GPU: 입력을 버퍼에 적기",
            "gpu_download" => "GPU: 결과를 읽어 오기",
            "linear" => "선형 층의 나머지 (CPU)",
            "attention" => "어텐션의 점수와 섞기 (CPU)",
            "map" => "활성화 함수 (CPU)",
            "zip" => "더하기와 곱하기 (CPU)",
            "rms_norm" => "정규화 (CPU)",
            "rope" => "회전 (CPU)",
            "mlp" => "MLP의 나머지 (CPU)",
            "embedding" => "임베딩 (CPU)",
            "output_layer" => continue,
            other => other,
        };
        let width = 34 - label.chars().filter(|c| !c.is_ascii()).count();
        println!(
            "{:<width$} {:>8} {:>12.1} {:>7.1}%",
            label,
            calls,
            seconds * 1e3,
            seconds / measured * 100.0
        );
    }
    Ok(())
}

/// 프롬프트 256토큰을 한꺼번에 처리하는 속도와 토큰을 하나씩 처리하는 속도
/// (토큰/초, 되풀이의 중앙값).
fn speed(model: &LlamaModel, tokens: &[TokenId]) -> Result<(f64, f64), Box<dyn std::error::Error>> {
    let mut prompt = Vec::new();
    let mut generation = Vec::new();
    for repetition in 0..8 {
        let mut cache = KvCache::new(&model.config);
        cache.reserve(tokens.len());
        let start = Instant::now();
        model.forward_cached(&tokens[..PROMPT_TOKENS], &mut cache)?;
        let prompt_seconds = start.elapsed().as_secs_f64();
        let start = Instant::now();
        for token in &tokens[PROMPT_TOKENS..] {
            model.forward_cached(std::slice::from_ref(token), &mut cache)?;
        }
        let generation_seconds = start.elapsed().as_secs_f64();
        // 첫 번째는 준비 운동이다.
        if repetition > 0 {
            prompt.push(PROMPT_TOKENS as f64 / prompt_seconds);
            generation.push(GENERATED as f64 / generation_seconds);
        }
    }
    Ok((
        Summary::new(&prompt).median,
        Summary::new(&generation).median,
    ))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let name = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "llama32_1b".to_string());
    let directory = models_dir().join(&name);
    let Some(gpu) = Gpu::new() else {
        println!("이 기계에는 쓸 수 있는 GPU가 없습니다.");
        return Ok(());
    };
    let gpu = Arc::new(gpu);
    println!("모델 {name}, GPU {}", gpu.name());

    let cpu = LlamaModel::load(&directory)?;
    let fast = Offload::new(Arc::clone(&gpu));
    let (on_gpu, seconds, bytes) = load_on_gpu(&directory, &fast, false)?;
    println!(
        "가중치를 GPU에 올리는 데 {seconds:.2}초, 올린 양 {:.2}GB (f32로 펼쳐서)",
        bytes as f64 / 1e9
    );
    // 표준 WebGPU의 기능만 쓰는 장치. 브라우저에서 돌 때와 같은 조건이다.
    let standard_gpu = Arc::new(Gpu::standard().expect("위에서 열린 GPU다"));
    let standard = Offload::new(Arc::clone(&standard_gpu));
    let (on_gpu_standard, _, _) = load_on_gpu(&directory, &standard, false)?;
    let whole = Offload::new(Arc::clone(&gpu));
    let (on_gpu_whole, _, _) = load_on_gpu(&directory, &whole, true)?;

    // 1. 점수가 얼마나 달라지는가(20.6절과 같은 방법).
    let tokenizer = Tokenizer::from_tokenizer_json(directory.join("tokenizer.json"))?;
    let mut prompt = tokenizer.prefix().to_vec();
    prompt.extend(tokenizer.encode(PROMPT, false));
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
    let from_truth = |logits: &[f32]| {
        logits
            .iter()
            .zip(&truth)
            .map(|(&logit, &truth)| (f64::from(logit) - truth).abs())
            .fold(0.0_f64, f64::max)
            / largest
    };
    let run = |model: &LlamaModel| -> Result<(Vec<f32>, Vec<TokenId>), Box<dyn std::error::Error>> {
        let logits = model.next_token_logits(&prompt)?.to_vec::<f32>()?;
        let mut cache = KvCache::new(&model.config);
        let generated = generate(model, &prompt, NEW_TOKENS, &mut cache, |_| {})?;
        Ok((logits, generated))
    };
    println!();
    println!("글 {PROMPT:?}, 이어 쓰는 토큰 {NEW_TOKENS}개");
    println!(
        "{:<34} {:>10} {:>10} {:>9} {:>10} {:>10}",
        "선형 층을 계산하는 곳", "상대 오차", "1위 토큰", "그 확률", "같은 토큰", "같은 비트"
    );
    let (base_logits, base_generated) = run(&cpu)?;
    let first = argmax(&base_logits).unwrap_or(0);
    let line = |label: &str, logits: &[f32], generated: &[TokenId]| {
        let best = argmax(logits).unwrap_or(0);
        let same = generated
            .iter()
            .zip(&base_generated)
            .take_while(|(a, b)| a == b)
            .count();
        let same_bits = logits
            .iter()
            .zip(&base_logits)
            .filter(|(a, b)| a.to_bits() == b.to_bits())
            .count();
        println!(
            "{:<34} {:>10.1e} {:>10} {:>8.2}% {:>10} {:>9.1}%",
            label,
            from_truth(logits),
            format!("{:?}", tokenizer.decode(&[best as TokenId])),
            softmax(logits)[first] * 100.0,
            format!("{same} / {NEW_TOKENS}"),
            100.0 * same_bits as f64 / logits.len() as f64
        );
    };
    line("CPU", &base_logits, &base_generated);
    let (logits, generated) = run(&on_gpu)?;
    line("GPU, 표준 밖의 기능도", &logits, &generated);
    let (again, _) = run(&on_gpu)?;
    let (logits_standard, generated_standard) = run(&on_gpu_standard)?;
    line("GPU, 표준 WebGPU만", &logits_standard, &generated_standard);
    let (logits_whole, generated_whole) = run(&on_gpu_whole)?;
    line("GPU, MLP를 통째로", &logits_whole, &generated_whole);
    let repeatable = logits
        .iter()
        .zip(&again)
        .all(|(a, b)| a.to_bits() == b.to_bits());
    println!(
        "GPU에서 같은 계산을 두 번 했을 때: {}",
        if repeatable {
            "모든 비트가 같다"
        } else {
            "다른 비트가 있다"
        }
    );

    // 2. 속도.
    let mut rng = Pcg64::new(23);
    let vocab = cpu.config.vocab_size as u64;
    let tokens: Vec<TokenId> = (0..PROMPT_TOKENS + GENERATED)
        .map(|_| (rng.next_u64() % vocab) as TokenId)
        .collect();
    println!();
    println!(
        "{:<44} {:>16} {:>14}",
        "선형 층을 계산하는 곳", "프롬프트 처리", "토큰 생성"
    );
    let report = |label: &str, (prompt, generation): (f64, f64)| {
        println!("{label:<44} {prompt:>12.0} 토큰/초 {generation:>8.1} 토큰/초");
    };
    println!("(프롬프트 256토큰을 한꺼번에 처리한 뒤, 이어서 토큰 32개를 하나씩)");
    report("CPU (22장의 엔진)", speed(&cpu, &tokens)?);
    report(
        "GPU, 표준 WebGPU만 (층마다)",
        speed(&on_gpu_standard, &tokens)?,
    );
    report("GPU, 표준 밖의 기능도 (층마다)", speed(&on_gpu, &tokens)?);
    report("GPU, MLP를 통째로", speed(&on_gpu_whole, &tokens)?);
    // 벡터가 32개보다 적은 일은 CPU가 한다.
    whole.set_min_count(32);
    whole.statistics(true);
    report(
        "위와 같되 벡터 32개부터만 GPU",
        speed(&on_gpu_whole, &tokens)?,
    );
    let split = whole.statistics(true);
    println!(
        "나눠 맡겼을 때 선형 층을 계산한 곳: GPU {}번, CPU {}번 (되풀이 8번의 합)",
        split.calls, split.on_cpu
    );

    // 3. 프롬프트 256토큰을 처리하는 시간이 어디에 쓰였나.
    profile_prompt("표준 WebGPU만, 선형 층을 하나씩", &on_gpu_standard, &tokens)?;
    profile_prompt("표준 밖의 기능도, 선형 층을 하나씩", &on_gpu, &tokens)?;
    profile_prompt("MLP를 통째로 GPU에", &on_gpu_whole, &tokens)?;
    Ok(())
}
