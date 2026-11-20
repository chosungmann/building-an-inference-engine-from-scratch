//! 46장: 가중치를 0으로 만들면 품질, 크기, 속도가 어떻게 되는지 잰다.
//!
//! 실행: `cargo run --release -p badak_models --example ch46_prune -- <부분> [모델]`
//!
//! 부분:
//!   loss    블록의 행렬마다 절댓값이 작은 것부터 0으로 만들고 손실을 잰다
//!   size    그렇게 만든 가중치가 차지하는 크기 (그대로, gzip, 희소 형식)
//!   kernel  행렬 하나의 곱셈: 보통의 커널과 희소 커널
//!   speed   모델 전체: 보통의 커널과 희소 커널로 토큰을 쓰는 속도
//!   blocks  블록을 통째로 빼고 손실을 잰다. 하나씩, 그리고 덜 아픈 것부터 여럿
//!
//! 학습은 하지 않는다. 받은 가중치를 그대로 깎는다.

use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

use badak_core::Tensor;
use badak_core::ops::LinearWeight;
use badak_core::packed::{Delegated, Matrix};
use badak_core::simd;
use badak_core::sparse::{Sparse, prune_magnitude};
use badak_models::llama::WeightFormat;
use badak_models::llama::cache::{KvCache, KvPrecision};
use badak_models::llama::generate::generate;
use badak_models::llama::model::LlamaModel;
use badak_models::llama::perplexity::{Losses, Protocol, evaluate};
use badak_models::models_dir;
use badak_models::tokenizer::Tokenizer;

const CONTEXT: usize = 512;
const CHUNKS: usize = 64;

/// 블록의 행렬을 깎은 모델. 임베딩 표는 깎지 않는다. `sparse`이면 깎은
/// 행렬의 곱셈을 희소 커널에 맡긴다.
fn pruned(
    directory: &std::path::Path,
    fraction: f64,
    sparse: bool,
) -> Result<LlamaModel, Box<dyn std::error::Error>> {
    let mut model = LlamaModel::load_as(directory, WeightFormat::F32)?;
    model.replace_matrices(|kind, _, matrix| {
        if kind == "embedding" || fraction == 0.0 && !sparse {
            return Ok(matrix.clone());
        }
        let shape = matrix.shape().to_vec();
        let mut values = matrix.to_f32()?.to_vec::<f32>()?;
        prune_magnitude(&mut values, fraction);
        let dense: Matrix = Tensor::from_vec(values.clone(), &shape)?.into();
        if sparse {
            let kernel = Sparse::from_dense(&values, shape[0], shape[1])?;
            Ok(Matrix::Delegated(Delegated::new(dense, Arc::new(kernel))))
        } else {
            Ok(dense)
        }
    })?;
    Ok(model)
}

/// 바이트를 `gzip -9`로 눌렀을 때의 크기.
fn gzipped(bytes: &[u8]) -> std::io::Result<usize> {
    let mut child = Command::new("gzip")
        .arg("-9")
        .arg("-c")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    let mut input = child.stdin.take().expect("위에서 연결했다");
    let bytes = bytes.to_vec();
    let writer = std::thread::spawn(move || input.write_all(&bytes));
    let output = child.wait_with_output()?;
    writer.join().expect("쓰는 스레드가 죽지 않는다")?;
    Ok(output.stdout.len())
}

fn median(mut times: Vec<Duration>) -> f64 {
    times.sort();
    times[times.len() / 2].as_secs_f64() * 1e6
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let part = arguments
        .next()
        .ok_or("부분(loss, size, kernel, speed, blocks)을 주십시오")?;
    let name = arguments.next().unwrap_or_else(|| "smollm2_135m".into());
    let directory = models_dir().join(&name);
    let fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

    match part.as_str() {
        "loss" => {
            let text = std::fs::read_to_string(
                models_dir().join("wikitext2/wikitext-2-raw/wiki.test.raw"),
            )?;
            let tokenizer = Tokenizer::from_tokenizer_json(directory.join("tokenizer.json"))?;
            let mut tokens = tokenizer.prefix().to_vec();
            tokens.extend(tokenizer.encode(&text, false));
            let protocol = Protocol {
                context: CONTEXT,
                chunks: CHUNKS,
                bos: tokenizer.prefix().first().copied(),
                kv: KvPrecision::F32,
            };
            let mut base: Option<Losses> = None;
            for fraction in fractions {
                let model = pruned(&directory, fraction, false)?;
                let losses = evaluate(&model, &tokens, protocol)?;
                let base = base.get_or_insert_with(|| losses.clone());
                let (gap, error) = losses.difference_from(base);
                println!(
                    "loss {name} 깎은비율 {:.0} 손실 {:.4} 차이 {gap:+.4} 오차 {error:.4} 퍼플렉서티 {:.2}",
                    fraction * 100.0,
                    losses.mean(),
                    losses.perplexity()
                );
            }
        }
        "size" => {
            for fraction in fractions {
                let (mut raw, mut csr, mut count, mut zeros) = (Vec::new(), 0, 0, 0);
                let mut model = LlamaModel::load_as(&directory, WeightFormat::F32)?;
                model.replace_matrices(|kind, _, matrix| {
                    if kind != "embedding" {
                        let shape = matrix.shape().to_vec();
                        let mut values = matrix.to_f32()?.to_vec::<f32>()?;
                        zeros += prune_magnitude(&mut values, fraction);
                        count += values.len();
                        csr += Sparse::from_dense(&values, shape[0], shape[1])?.bytes();
                        // 이 모델은 값을 16비트(BF16)로 적어 둔다. `f32`의 위쪽
                        // 16비트가 그것이다. 파일에 적히는 그대로 센다.
                        for value in &values {
                            raw.extend(((value.to_bits() >> 16) as u16).to_le_bytes());
                        }
                    }
                    Ok(matrix.clone())
                })?;
                println!(
                    "size {name} 깎은비율 {:.0} 값 {count} 영 {zeros} 그대로 {} gzip {} 희소형식 {}",
                    fraction * 100.0,
                    raw.len(),
                    gzipped(&raw)?,
                    csr
                );
            }
        }
        "kernel" => {
            // SmolLM2-135M의 세 가지 행렬.
            for (rows, cols) in [(576, 576), (1536, 576), (576, 1536)] {
                for count in [1, 64] {
                    for fraction in [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98] {
                        let mut weight: Vec<f32> = (0..rows * cols)
                            .map(|index| ((index * 2_654_435_761) % 10_007) as f32 / 10_007.0 - 0.5)
                            .collect();
                        prune_magnitude(&mut weight, fraction);
                        let sparse = Sparse::from_dense(&weight, rows, cols)?;
                        let x: Vec<f32> = (0..count * cols)
                            .map(|index| (index % 17) as f32 * 0.1 - 0.8)
                            .collect();
                        let mut y = vec![0.0_f32; count * rows];
                        let (mut dense_times, mut sparse_times) = (Vec::new(), Vec::new());
                        // 번갈아 잰다.
                        for round in 0..205 {
                            let started = Instant::now();
                            simd::linear(&x, count, &weight, rows, cols, &mut y);
                            let dense = started.elapsed();
                            std::hint::black_box(&mut y);
                            let started = Instant::now();
                            sparse.apply(&x, count, &mut y)?;
                            let thin = started.elapsed();
                            std::hint::black_box(&mut y);
                            if round >= 5 {
                                dense_times.push(dense);
                                sparse_times.push(thin);
                            }
                        }
                        let (dense, thin) = (median(dense_times), median(sparse_times));
                        println!(
                            "kernel {rows}x{cols} 벡터 {count} 깎은비율 {:.0} 보통 {dense:.2} 희소 {thin:.2} 비 {:.2}",
                            fraction * 100.0,
                            dense / thin
                        );
                    }
                }
            }
        }
        "speed" => {
            let tokenizer = Tokenizer::from_tokenizer_json(directory.join("tokenizer.json"))?;
            let mut prompt = tokenizer.prefix().to_vec();
            prompt.extend(
                tokenizer.encode("Once upon a time, in a small village by the sea,", false),
            );
            // 프롬프트 처리: 같은 토큰을 이어 붙여 256개로 만든다.
            let long: Vec<_> = prompt.iter().copied().cycle().take(256).collect();
            // (깎은 비율, 커널). 마지막 줄은 깎지 않고 4비트로 줄인 모델이다.
            // 폰에 실릴 것과 견주려고 넣었다.
            for (fraction, kernel) in [
                (0.0, "보통"),
                (0.0, "희소"),
                (0.5, "희소"),
                (0.7, "희소"),
                (0.8, "희소"),
                (0.9, "희소"),
                (0.95, "희소"),
                (0.0, "Q4_0"),
            ] {
                std::thread::sleep(Duration::from_secs(3));
                let model = if kernel == "Q4_0" {
                    LlamaModel::load_as(&directory, WeightFormat::Q4_0)?
                } else {
                    pruned(&directory, fraction, kernel == "희소")?
                };
                let mut reading = Vec::new();
                let mut writing = Vec::new();
                for _ in 0..3 {
                    let mut cache = KvCache::new(&model.config);
                    let started = Instant::now();
                    generate(&model, &long, 1, &mut cache, |_| {})?;
                    reading.push(long.len() as f64 / started.elapsed().as_secs_f64());
                    let mut first = None;
                    let mut cache = KvCache::new(&model.config);
                    let written = generate(&model, &prompt, 128, &mut cache, |_| {
                        first.get_or_insert_with(Instant::now);
                    })?;
                    writing.push(
                        (written.len().max(2) - 1) as f64
                            / first.map_or(1.0, |at| at.elapsed().as_secs_f64()),
                    );
                }
                reading.sort_by(f64::total_cmp);
                writing.sort_by(f64::total_cmp);
                println!(
                    "speed {name} 깎은비율 {:.0} 커널 {} 읽기 {:.0} 쓰기 {:.0}",
                    fraction * 100.0,
                    kernel,
                    reading[1],
                    writing[1]
                );
            }
        }
        "blocks" => {
            let text = std::fs::read_to_string(
                models_dir().join("wikitext2/wikitext-2-raw/wiki.test.raw"),
            )?;
            let tokenizer = Tokenizer::from_tokenizer_json(directory.join("tokenizer.json"))?;
            let mut tokens = tokenizer.prefix().to_vec();
            tokens.extend(tokenizer.encode(&text, false));
            let protocol = Protocol {
                context: CONTEXT,
                chunks: CHUNKS,
                bos: tokenizer.prefix().first().copied(),
                kv: KvPrecision::F32,
            };
            let whole = LlamaModel::load_as(&directory, WeightFormat::F32)?;
            let count = whole.config.num_layers;
            let base = evaluate(&whole, &tokens, protocol)?;
            println!("blocks {name} 기준 손실 {:.4}", base.mean());
            // 1. 블록을 하나씩 빼 본다.
            let mut cost = Vec::new();
            for removed in 0..count {
                let mut model = LlamaModel::load_as(&directory, WeightFormat::F32)?;
                model.retain_blocks(|index| index != removed);
                let (gap, _) = evaluate(&model, &tokens, protocol)?.difference_from(&base);
                println!("blocks {name} 하나빼기 {removed} 차이 {gap:+.4}");
                cost.push((gap, removed));
            }
            // 2. 빼도 덜 아픈 것부터 차례로 더 뺀다.
            cost.sort_by(|a, b| a.0.total_cmp(&b.0));
            for dropped in 1..=12.min(count - 1) {
                let gone: Vec<usize> = cost[..dropped].iter().map(|&(_, index)| index).collect();
                let mut model = LlamaModel::load_as(&directory, WeightFormat::F32)?;
                model.retain_blocks(|index| !gone.contains(&index));
                let (gap, _) = evaluate(&model, &tokens, protocol)?.difference_from(&base);
                let mut listed = gone.clone();
                listed.sort_unstable();
                println!(
                    "blocks {name} 여럿빼기 {dropped} 차이 {gap:+.4} 뺀것 {}",
                    listed
                        .iter()
                        .map(usize::to_string)
                        .collect::<Vec<_>>()
                        .join(",")
                );
            }
        }
        other => return Err(format!("모르는 부분 {other}").into()),
    }
    Ok(())
}
