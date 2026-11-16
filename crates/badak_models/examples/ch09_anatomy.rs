//! 9장: 언어 모델을 해부한다.
//!
//! 실행: `cargo run --release -p badak_models --example ch09_anatomy -- [모델 디렉터리]`
//!
//! 모델 디렉터리의 이름을 주지 않으면 `llama32_1b`를 쓴다. 승인 없이 받을 수
//! 있는 모델로 따라 하려면 `smollm2_135m`을 준다.

use std::hint::black_box;
use std::time::Instant;

use badak_core::approx::compare;
use badak_core::{Tensor, ops};
use badak_formats::safetensors::SafeTensors;
use badak_models::llama::attention::attention;
use badak_models::llama::{LlamaFiles, mlp, rope};
use badak_models::models_dir;

/// 5장에서 정한 기준: 많은 수를 더하는 계산의 상대 오차.
const RTOL: f32 = 1e-5;

fn report(
    name: &str,
    actual: &Tensor,
    expected: &Tensor,
) -> Result<bool, Box<dyn std::error::Error>> {
    let (actual, expected) = (actual.to_vec::<f32>()?, expected.to_vec::<f32>()?);
    let scale = expected
        .iter()
        .fold(0.0_f32, |max, value| max.max(value.abs()));
    let result = compare(&actual, &expected, RTOL, RTOL * scale);
    println!(
        "{:<22} {:>8} {:>10.3} {:>12.3e} {:>12.3e} {:>6}",
        name,
        result.len,
        scale,
        result.max_abs_diff,
        RTOL * scale,
        if result.is_close() {
            "통과"
        } else {
            "실패"
        }
    );
    Ok(result.is_close())
}

fn milliseconds<R>(mut f: impl FnMut() -> R) -> f64 {
    (0..3)
        .map(|_| {
            let start = Instant::now();
            black_box(f());
            start.elapsed().as_secs_f64() * 1e3
        })
        .fold(f64::INFINITY, f64::min)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let name = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "llama32_1b".to_string());
    let files = LlamaFiles::open(models_dir().join(&name))?;
    let c = &files.config;

    println!("== 설정 ({name}) ==");
    println!("단어 사전 크기        {}", c.vocab_size);
    println!("벡터 길이             {}", c.hidden_size);
    println!("블록 수               {}", c.num_layers);
    println!(
        "헤드 수               {} (키·값 헤드 {})",
        c.num_heads, c.num_kv_heads
    );
    println!("헤드의 벡터 길이      {}", c.head_dim());
    println!("MLP 가운데 층의 길이  {}", c.intermediate_size);
    println!("회전의 기준 주기      {}", c.rope_theta);

    // 파일에 실제로 들어 있는 텐서의 형상에서 파라미터 수를 센다.
    println!();
    println!("== 파라미터는 어디에 있는가 ==");
    let count = |pattern: &str| -> usize {
        files
            .weights()
            .names()
            .filter(|tensor| tensor.contains(pattern))
            .map(|tensor| {
                files
                    .weights()
                    .info(tensor)
                    .unwrap()
                    .shape
                    .iter()
                    .product::<usize>()
            })
            .sum()
    };
    let total = count("");
    let groups = [
        ("토큰 임베딩", count("embed_tokens")),
        ("어텐션 (모든 블록)", count("self_attn")),
        ("MLP (모든 블록)", count(".mlp.")),
        ("정규화", count("norm")),
    ];
    for (label, parameters) in groups {
        println!(
            "{:<22} {:>14} {:>7.1}%",
            label,
            parameters,
            100.0 * parameters as f64 / total as f64
        );
    }
    println!("{:<22} {:>14} {:>7.1}%", "합계", total, 100.0);
    println!(
        "파일의 텐서 {}개. f32로 펼치면 {:.2}GiB",
        files.weights().names().count(),
        (total * 4) as f64 / (1 << 30) as f64
    );

    // 첫 블록의 부품을 하나씩 PyTorch와 대조한다. 부품마다 PyTorch가 계산한
    // 입력을 넣어, 앞 부품의 오차가 뒤로 번지지 않게 한다.
    let reference = SafeTensors::open(files.directory().join("reference.safetensors"))?;
    let expected = |tensor: &str| reference.tensor(tensor);
    let ids = expected("ids")?;
    let tokens = ids.len();
    println!();
    println!("== 첫 블록의 부품별 대조 ==");
    println!("글: {:?}", reference.metadata()["prompt"]);
    println!("토큰 {tokens}개: {}", reference.metadata()["tokens"]);
    println!(
        "{:<22} {:>8} {:>10} {:>12} {:>12} {:>6}",
        "부품", "원소 수", "최대 크기", "최대 오차", "허용 오차", "결과"
    );
    let embedding = files.embedding()?;
    let block = files.block(0)?;
    let frequencies = rope::inverse_frequencies(c);
    let mut ok = true;

    let embedded = ops::embedding(&embedding, &ids)?;
    ok &= report("임베딩", &embedded, &expected("embed")?)?;

    let normed = ops::rms_norm(&expected("embed")?, &block.attention_norm, c.rms_norm_eps)?;
    ok &= report("정규화 1", &normed, &expected("l0.norm1")?)?;

    let normed = expected("l0.norm1")?;
    ok &= report(
        "질의 (선형 층)",
        &ops::linear(&normed, &block.q_proj, None)?,
        &expected("l0.q")?,
    )?;
    ok &= report(
        "키 (선형 층)",
        &ops::linear(&normed, &block.k_proj, None)?,
        &expected("l0.k")?,
    )?;
    ok &= report(
        "값 (선형 층)",
        &ops::linear(&normed, &block.v_proj, None)?,
        &expected("l0.v")?,
    )?;

    let attended = attention(&normed, &block, c, &frequencies)?;
    ok &= report(
        "어텐션 비중",
        &attended.weights,
        &expected("l0.attn_weights")?,
    )?;
    ok &= report("어텐션 출력", &attended.output, &expected("l0.attn")?)?;

    let after_attention = ops::add(&expected("embed")?, &expected("l0.attn")?)?;
    let normed = ops::rms_norm(&after_attention, &block.mlp_norm, c.rms_norm_eps)?;
    ok &= report("정규화 2", &normed, &expected("l0.norm2")?)?;

    let mlp_out = mlp(&expected("l0.norm2")?, &block)?;
    ok &= report("MLP", &mlp_out, &expected("l0.mlp")?)?;

    let block_out = ops::add(&after_attention, &expected("l0.mlp")?)?;
    ok &= report("블록 출력", &block_out, &expected("l0.out")?)?;
    println!(
        "{}",
        if ok {
            "모든 부품이 통과했습니다."
        } else {
            "실패한 부품이 있습니다."
        }
    );

    // 마지막 토큰이 앞의 토큰들에 준 비중을 헤드에 걸쳐 평균한 것.
    println!();
    println!("== 마지막 토큰의 어텐션 비중 (첫 블록, 헤드 평균) ==");
    let weights = attended.weights.to_vec::<f32>()?;
    let labels: Vec<&str> = reference.metadata()["tokens"].split(' ').collect();
    for (earlier, label) in labels.iter().enumerate() {
        let mean: f32 = (0..c.num_heads)
            .map(|head| weights[(head * tokens + tokens - 1) * tokens + earlier])
            .sum::<f32>()
            / c.num_heads as f32;
        println!(
            "{earlier:>3} {label:<12} {mean:.3} {}",
            "#".repeat((mean * 60.0).round() as usize)
        );
    }

    println!();
    println!("== 첫 블록의 부품별 시간 (토큰 {tokens}개, 스레드 1개) ==");
    let x = expected("l0.norm1")?;
    let y = expected("l0.norm2")?;
    let rows = [
        (
            "임베딩",
            milliseconds(|| ops::embedding(&embedding, &ids).unwrap()),
        ),
        (
            "정규화",
            milliseconds(|| ops::rms_norm(&x, &block.attention_norm, c.rms_norm_eps).unwrap()),
        ),
        (
            "어텐션",
            milliseconds(|| attention(&x, &block, c, &frequencies).unwrap().output),
        ),
        ("MLP", milliseconds(|| mlp(&y, &block).unwrap())),
    ];
    for (label, time) in rows {
        println!("{label:<10} {time:>10.3} ms");
    }
    if !ok {
        std::process::exit(1);
    }
    Ok(())
}
