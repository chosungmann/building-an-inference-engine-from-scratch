//! 48장: 브라우저용 문의 함수들을 맥에서 그대로 불러, 견줄 답을 적는다.
//!
//! 실행: `cargo run --release -p badak_web --example ch48_native -- [llm [가중치]]`
//!
//! 적는 줄은 `crates/badak_web/js/run_node.mjs`가 적는 줄과 같은 꼴이다.
//! 지문과 비트가 같으면 WebAssembly에서 맥과 같은 계산을 한 것이다.

use std::ffi::CString;
use std::time::Instant;

use badak_core::simd::Backend;
use badak_graph::models_dir;
use badak_web::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("backend {:?}", Backend::detect());
    for (rows, cols, count) in [(13, 203, 9), (1536, 576, 1), (1536, 576, 64)] {
        let mut digest = 0_u64;
        // SAFETY: `digest`는 `u64` 하나를 쓸 수 있는 자리다.
        let seconds = unsafe { badak_web_bench_linear(rows, cols, count, 30, &mut digest) };
        println!(
            "linear {rows}x{cols}x{count} 지문 {digest:016x} 시간 {:.1}",
            seconds * 1e6
        );
    }

    let images = std::fs::read(models_dir().join("mnist/t10k-images-idx3-ubyte"))?;
    let path = models_dir().join("mnist_tflite/mlp.tflite");
    let path = path.to_str().ok_or("길이 UTF-8이 아닙니다")?;
    // SAFETY: 문의 약속대로 부른다. 포인터들은 해석기가 살아 있는 동안 쓴다.
    unsafe {
        let network = badak_web_graph_open(path.as_ptr(), path.len());
        assert!(!network.is_null());
        let input = std::slice::from_raw_parts_mut(
            badak_web_graph_input(network),
            badak_web_graph_input_len(network),
        );
        // 해석기는 입력의 자리를 계산 도중에 다시 쓴다(27장). 돌릴 때마다 채운다.
        let started = Instant::now();
        for _ in 0..5000 {
            for (value, &pixel) in input.iter_mut().zip(&images[16..16 + 784]) {
                *value = f32::from(pixel) / 255.0;
            }
            assert_eq!(badak_web_graph_run(network), 0);
        }
        println!(
            "mnist-time {:.1}",
            started.elapsed().as_secs_f64() * 1e6 / 5000.0
        );
        let scores = std::slice::from_raw_parts(
            badak_web_graph_output(network),
            badak_web_graph_output_len(network),
        );
        let best = (0..scores.len())
            .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
            .unwrap_or(0);
        let bits: Vec<String> = scores
            .iter()
            .map(|score| format!("{:08x}", score.to_bits()))
            .collect();
        println!("mnist 답 {best} 비트 {}", bits.join(" "));
    }

    if std::env::args().nth(1).as_deref() == Some("llm") {
        let directory = CString::new(models_dir().join("smollm2_135m").to_str().ok_or("길")?)?;
        let system = "You are a helpful AI assistant named SmolLM, trained by Hugging Face";
        let weights = std::env::args()
            .nth(2)
            .and_then(|text| text.parse().ok())
            .unwrap_or(0);
        // SAFETY: 문의 약속대로 부른다.
        unsafe {
            let llm = badak_web_llm_open(
                directory.as_ptr().cast(),
                directory.as_bytes().len(),
                weights,
            );
            assert!(!llm.is_null());
            let questions = [
                ("short", "What is the capital of France?", 32),
                (
                    "long",
                    "Write a short story about a robot learning to paint.",
                    96,
                ),
            ];
            for (label, question, most) in questions {
                // 브라우저의 측정(bench.html)과 같이 세 번 묻고 가장 빨랐던 것을 적는다.
                let mut fastest: Option<(f64, f64, usize, i32, Vec<u8>)> = None;
                for _ in 0..3 {
                    std::thread::sleep(std::time::Duration::from_secs(3));
                    let started = Instant::now();
                    let written = badak_web_llm_chat(
                        llm,
                        system.as_ptr(),
                        system.len(),
                        question.as_ptr(),
                        question.len(),
                        most,
                    );
                    let total = started.elapsed().as_secs_f64() * 1e3;
                    let pieces = take_pieces();
                    let first = pieces
                        .first()
                        .map_or(0.0, |(at, _)| (*at - started).as_secs_f64() * 1e3);
                    let answer: Vec<u8> = pieces
                        .iter()
                        .flat_map(|(_, bytes)| bytes.iter().copied())
                        .collect();
                    if fastest.as_ref().is_none_or(|best| total < best.1) {
                        fastest = Some((first, total, pieces.len(), written, answer));
                    }
                }
                let (first, total, pieces, written, answer) = fastest.ok_or("측정이 없습니다")?;
                let rate = (pieces as f64 - 1.0) / (total - first) * 1e3;
                let hex: String = answer.iter().map(|byte| format!("{byte:02x}")).collect();
                let shown = if label == "short" {
                    hex
                } else {
                    format!("{}bytes", answer.len())
                };
                println!(
                    "llm-chat native weights {weights} {label} tokens {written} first-ms {first:.0} total-ms {total:.0} per-s {rate:.1} text {shown}"
                );
            }
            badak_web_llm_close(llm);
        }
    }
    Ok(())
}
