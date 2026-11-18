//! 28장: 16비트 부동소수점으로 무엇을 얻고 무엇을 잃는지 본다.
//!
//! 실행: `cargo run --release -p badak_graph --example ch28_half`
//!
//! 셋을 잰다. 가중치를 16비트로 줄였을 때 모델의 점수가 얼마나 달라지는지,
//! 가중치만 16비트로 두고 `f32`로 계산하는 커널의 속도, 그리고 계산까지
//! 16비트로 하는 커널의 속도와 오차. 스레드는 하나만 쓴다.

use std::hint::black_box;

use badak_bench::stats::Summary;
use badak_bench::timing::{Plan, measure};
use badak_core::Tensor;
use badak_core::quant::F16;
use badak_core::random::Pcg64;
use badak_core::simd::{self, Backend, Walk};
use badak_formats::safetensors::SafeTensors;
use badak_formats::tflite::Model;
use badak_graph::{Graph, Interpreter, Registry, models_dir};

const IMAGES: [&str; 5] = ["cup", "golden_retriever", "guitar", "pelican", "tabby_cat"];

/// 계산까지 16비트로 하는 커널의 시간. 이 프로세서가 지원하지 않으면
/// `None`이다.
#[allow(clippy::too_many_arguments)]
fn native(
    plan: Plan,
    x: &[F16],
    w: &[F16],
    y: &mut [F16],
    vectors: usize,
    rows: usize,
    cols: usize,
) -> Option<f64> {
    #[cfg(target_arch = "aarch64")]
    if simd::half::available() {
        let samples = measure(plan, || {
            simd::half::linear(black_box(x), vectors, w, rows, cols, y);
        });
        return Some(Summary::new(&samples).median);
    }
    let _ = (plan, x, w, y, vectors, rows, cols);
    None
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. 가중치를 16비트로 줄였다가 되돌린 모델. 계산은 f32 그대로다.
    let directory = models_dir().join("vision_tflite");
    let reference = SafeTensors::open(directory.join("reference.safetensors"))?;
    let model = Model::open(directory.join("resnet50.tflite"))?;
    let graph = Graph::from_tflite(&model)?;
    let mut rounded = graph.clone();
    let (mut count, mut worst) = (0_usize, 0.0_f32);
    for value in &mut rounded.values {
        let Some(tensor) = &value.constant else {
            continue;
        };
        let Ok(original) = tensor.to_vec::<f32>() else {
            continue;
        };
        let halved: Vec<f32> = original
            .iter()
            .map(|&weight| F16::from_f32(weight).to_f32())
            .collect();
        for (a, b) in original.iter().zip(&halved) {
            worst = worst.max((a - b).abs() / a.abs().max(1e-3));
        }
        count += halved.len();
        value.constant = Some(Tensor::from_vec(halved, tensor.shape())?);
    }
    println!("ResNet-50의 가중치 {count}개를 16비트로 줄였다가 되돌림");
    println!(
        "  파일에서의 크기 {:.1}MB → {:.1}MB",
        count as f64 * 4.0 / 1e6,
        count as f64 * 2.0 / 1e6
    );
    println!("  가중치 하나의 가장 큰 상대 오차 {worst:.2e}");
    let registry = Registry::builtin();
    let mut exact = Interpreter::new(&graph, &registry)?;
    let mut half = Interpreter::new(&rounded, &registry)?;
    println!(
        "  {:<18} {:>12} {:>24}",
        "사진", "점수의 차이", "첫째 답의 확률"
    );
    for image in IMAGES {
        let input = reference
            .tensor(&format!("{image}.input"))?
            .to_vec::<f32>()?;
        exact.input_mut(0).copy_from_slice(&input);
        half.input_mut(0).copy_from_slice(&input);
        exact.run()?;
        half.run()?;
        let gap = exact
            .output(0)
            .iter()
            .zip(half.output(0))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        let top = |scores: &[f32]| {
            let probabilities = badak_core::prob::softmax(scores);
            let best = (0..scores.len())
                .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
                .expect("점수가 있다");
            (best, 100.0 * probabilities[best])
        };
        let (a, b) = (top(exact.output(0)), top(half.output(0)));
        println!(
            "  {image:<18} {gap:>12.4} {:>10.2}% → {:.2}%{}",
            a.1,
            b.1,
            if a.0 == b.0 { "" } else { "  (답이 바뀜)" }
        );
    }

    // 2. 커널의 속도. 합성곱 둘의 모양으로 잰다.
    println!("\n선형 층의 커널, 스레드 하나 (곱셈 억 번/초)");
    println!(
        "{:<26} {:>10} {:>16} {:>14} {:>12}",
        "모양 (벡터 × 행 × 열)", "f32", "가중치만 16비트", "모두 16비트", "상대 오차"
    );
    let mut rng = Pcg64::new(28);
    let plan = Plan {
        warmup: 3,
        repetitions: 21,
    };
    for (vectors, rows, cols) in [
        (3136, 64, 576),
        (784, 512, 128),
        (196, 256, 1024),
        (196, 256, 2304),
    ] {
        let mut values =
            |len: usize| -> Vec<f32> { (0..len).map(|_| rng.next_f64() as f32 - 0.5).collect() };
        let (x, w) = (values(vectors * cols), values(rows * cols));
        let to_half = |v: &[f32]| v.iter().map(|&f| F16::from_f32(f)).collect::<Vec<_>>();
        let (x16, w16) = (to_half(&x), to_half(&w));
        let (backend, walk) = (Backend::detect(), Walk::for_vectors(vectors));
        let mut y = vec![0.0_f32; vectors * rows];
        let mut y16 = vec![F16::from_f32(0.0); vectors * rows];
        let rate = |seconds: f64| (vectors * rows * cols) as f64 / seconds / 1e8;
        let full = Summary::new(&measure(plan, || {
            simd::linear_with(
                backend,
                walk,
                black_box(&x),
                vectors,
                &w,
                rows,
                cols,
                &mut y,
            );
        }))
        .median;
        let mixed = Summary::new(&measure(plan, || {
            simd::linear_with(
                backend,
                walk,
                black_box(&x),
                vectors,
                &w16,
                rows,
                cols,
                &mut y,
            );
        }))
        .median;
        simd::linear_with(backend, walk, &x, vectors, &w, rows, cols, &mut y);
        let (native, error) = match native(plan, &x16, &w16, &mut y16, vectors, rows, cols) {
            Some(seconds) => {
                let scale = y.iter().fold(0.0_f32, |max, value| max.max(value.abs()));
                let error = y
                    .iter()
                    .zip(&y16)
                    .map(|(exact, half)| (exact - half.to_f32()).abs())
                    .fold(0.0_f32, f32::max)
                    / scale;
                (format!("{:.0}", rate(seconds)), format!("{error:.1e}"))
            }
            None => ("없음".into(), String::new()),
        };
        println!(
            "{:<26} {:>10.0} {:>16.0} {:>14} {:>12}",
            format!("{vectors} × {rows} × {cols}"),
            rate(full),
            rate(mixed),
            native,
            error
        );
    }
    Ok(())
}
