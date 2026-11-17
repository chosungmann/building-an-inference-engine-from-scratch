//! 19장: 진짜 가중치를 더 적은 비트로 나타내면 오차가 얼마나 나는지 잰다.
//!
//! 실행: `cargo run --release -p badak_models --example ch19_weights -- [모델 디렉터리]`
//!
//! 가중치의 값은 찍지 않는다. 종류별로 모은 통계만 찍는다.

use std::collections::BTreeMap;
use std::sync::Mutex;

use badak_core::quant::{Quantizer, bf16_from_f32, bf16_to_f32, f16_from_f32, f16_to_f32};
use badak_core::threads;
use badak_models::llama::model::LlamaModel;
use badak_models::models_dir;

/// 견줄 방법들: (이름, 비트, 눈금 하나가 맡는 값의 수. 0이면 텐서 전체, 1이면 행 하나).
const SCHEMES: [(&str, u32, usize); 6] = [
    ("8비트/텐서", 8, 0),
    ("8비트/행", 8, 1),
    ("8비트/32개", 8, 32),
    ("4비트/텐서", 4, 0),
    ("4비트/행", 4, 1),
    ("4비트/32개", 4, 32),
];

/// 가중치 한 종류의 통계. 제곱의 합들은 `f64`로 모은다.
#[derive(Default, Clone)]
struct Stats {
    count: usize,
    signal: f64,
    /// 텐서마다의 "가장 큰 절댓값 ÷ 표준편차".
    peaks: Vec<f64>,
    /// 행마다의 같은 값을 텐서 안에서 평균한 것.
    row_peaks: Vec<f64>,
    f16_noise: f64,
    f16_changed: usize,
    bf16_changed: usize,
    noise: [f64; SCHEMES.len()],
}

fn square_sum(values: impl Iterator<Item = f64>) -> f64 {
    values.map(|value| value * value).sum()
}

/// 가장 큰 절댓값을 표준편차(평균을 0으로 본 것)로 나눈다.
fn peak(values: &[f32]) -> f64 {
    let largest = values.iter().fold(0.0_f32, |max, v| max.max(v.abs()));
    let power = square_sum(values.iter().map(|&v| f64::from(v))) / values.len() as f64;
    f64::from(largest) / power.sqrt()
}

// ANCHOR: noise
/// 대칭 눈금으로 바꿨다가 되돌릴 때 생기는 오차의 제곱의 합. `block`개마다
/// 눈금을 따로 정한다.
fn noise(values: &[f32], bits: u32, block: usize) -> f64 {
    values
        .chunks(block)
        .map(|chunk| {
            let quantizer = Quantizer::symmetric_for(chunk, bits);
            square_sum(
                chunk
                    .iter()
                    .map(|&v| f64::from(v) - f64::from(quantizer.round_trip(v))),
            )
        })
        .sum()
}
// ANCHOR_END: noise

fn measure(values: &[f32], cols: usize) -> Stats {
    let mut stats = Stats {
        count: values.len(),
        signal: square_sum(values.iter().map(|&v| f64::from(v))),
        peaks: vec![peak(values)],
        row_peaks: vec![values.chunks(cols).map(peak).sum::<f64>() / (values.len() / cols) as f64],
        ..Stats::default()
    };
    for &value in values {
        let half = f16_to_f32(f16_from_f32(value));
        stats.f16_changed += usize::from(half != value);
        stats.f16_noise += (f64::from(value) - f64::from(half)).powi(2);
        stats.bf16_changed += usize::from(bf16_to_f32(bf16_from_f32(value)) != value);
    }
    for (slot, (_, bits, block)) in stats.noise.iter_mut().zip(SCHEMES) {
        let block = match block {
            0 => values.len(),
            1 => cols,
            size => size,
        };
        *slot = noise(values, bits, block);
    }
    stats
}

fn db(signal: f64, noise: f64) -> String {
    if noise == 0.0 {
        "오차 없음".to_string()
    } else {
        format!("{:.1}", 10.0 * (signal / noise).log10())
    }
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let name = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "llama32_1b".to_string());
    let model = LlamaModel::load(models_dir().join(&name))?;
    let matrices = model.matrices();

    // 텐서마다 따로 재도 되므로 18장의 풀에 나눈다.
    let results: Vec<Mutex<Option<Stats>>> = matrices.iter().map(|_| Mutex::new(None)).collect();
    let pool = threads::global().pool();
    pool.run(matrices.len(), pool.threads(), &|index| {
        let (_, _, tensor) = matrices[index];
        let values = tensor.as_slice::<f32>().expect("가중치는 연속인 f32다");
        let stats = measure(values, tensor.shape()[1]);
        *results[index].lock().expect("한 번만 잠근다") = Some(stats);
    });

    // 종류별로 모은다.
    let mut kinds: BTreeMap<&str, Stats> = BTreeMap::new();
    let order = [
        "embedding",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "전체",
    ];
    for ((kind, _, _), result) in matrices.iter().zip(&results) {
        let stats = result.lock().expect("끝났다").take().expect("모두 쟀다");
        for key in [*kind, "전체"] {
            let total = kinds.entry(key).or_default();
            total.count += stats.count;
            total.signal += stats.signal;
            total.peaks.extend(&stats.peaks);
            total.row_peaks.extend(&stats.row_peaks);
            total.f16_noise += stats.f16_noise;
            total.f16_changed += stats.f16_changed;
            total.bf16_changed += stats.bf16_changed;
            for (sum, value) in total.noise.iter_mut().zip(stats.noise) {
                *sum += value;
            }
        }
    }

    println!("모델 {name}: 행렬 모양의 가중치 {}개", matrices.len());
    println!();
    println!("1. 값이 얼마나 퍼져 있는가 (가장 큰 절댓값 ÷ 표준편차, 가운데 값)");
    println!(
        "{:<12} {:>8} {:>14} {:>12} {:>12}",
        "종류", "텐서", "값 (백만)", "텐서 전체", "행 하나"
    );
    for kind in order {
        let stats = kinds.get_mut(kind).expect("모든 종류가 있다");
        println!(
            "{:<12} {:>8} {:>14.1} {:>12.1} {:>12.1}",
            kind,
            stats.peaks.len(),
            stats.count as f64 / 1e6,
            median(&mut stats.peaks),
            median(&mut stats.row_peaks),
        );
    }

    println!();
    println!("2. 16비트 실수로 바꾸면 달라지는 값");
    println!(
        "{:<12} {:>14} {:>14} {:>14}",
        "종류", "BF16", "f16", "f16의 SQNR"
    );
    for kind in order {
        let stats = &kinds[kind];
        println!(
            "{:<12} {:>13}개 {:>13}개 {:>14}",
            kind,
            stats.bf16_changed,
            stats.f16_changed,
            db(stats.signal, stats.f16_noise)
        );
    }

    println!();
    println!("3. 정수 눈금으로 바꿀 때의 SQNR (dB)");
    print!("{:<12}", "종류");
    for (label, _, _) in SCHEMES {
        print!(" {label:>12}");
    }
    println!();
    for kind in order {
        let stats = &kinds[kind];
        print!("{kind:<12}");
        for noise in stats.noise {
            print!(" {:>12}", db(stats.signal, noise));
        }
        println!();
    }
    Ok(())
}
