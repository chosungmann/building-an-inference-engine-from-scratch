//! 19장: 정수 눈금에 맞출 때 생기는 오차를 잰다.
//!
//! 실행: `cargo run --release -p badak_core --example ch19_uniform`
//!
//! 만든 값으로 네 가지를 본다. 비트의 수와 오차, 눈금의 끝을 어디에 둘지,
//! 유난히 큰 값 하나의 영향, 한쪽으로 치우친 값.

use badak_core::quant::{Granularity, Quantizer, fake_quantize, sqnr_db};
use badak_core::random::Pcg64;

const COUNT: usize = 1 << 20;

/// -1과 1 사이에 고르게 퍼진 값.
fn uniform(rng: &mut Pcg64, count: usize) -> Vec<f32> {
    (0..count)
        .map(|_| (rng.next_f64() * 2.0 - 1.0) as f32)
        .collect()
}

/// 평균 0, 표준편차 1인 종 모양 분포의 값(박스-뮬러 변환).
fn gaussian(rng: &mut Pcg64, count: usize) -> Vec<f32> {
    (0..count)
        .map(|_| {
            let (u, v) = (1.0 - rng.next_f64(), rng.next_f64());
            ((-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()) as f32
        })
        .collect()
}

fn round_trip(values: &[f32], quantizer: Quantizer) -> Vec<f32> {
    values
        .iter()
        .map(|&value| quantizer.round_trip(value))
        .collect()
}

// ANCHOR: predicted
/// 스케일 한 칸이 `scale`일 때 예상되는 SQNR. 오차가 칸 안에 고르게 퍼진다고
/// 보면 오차의 제곱의 평균은 `scale² ÷ 12`다.
fn predicted_sqnr(values: &[f32], scale: f32) -> f64 {
    let power = values
        .iter()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum::<f64>()
        / values.len() as f64;
    10.0 * (power / (f64::from(scale).powi(2) / 12.0)).log10()
}
// ANCHOR_END: predicted

fn main() {
    let mut rng = Pcg64::new(19);
    let flat = uniform(&mut rng, COUNT);
    let bell = gaussian(&mut rng, COUNT);
    let largest = bell.iter().fold(0.0_f32, |max, v| max.max(v.abs()));

    println!("1. 비트의 수와 SQNR (dB). 눈금의 끝은 가장 큰 절댓값");
    println!(
        "{:>6} {:>8} {:>12} {:>12} {:>12} {:>12}",
        "비트", "단계", "고른 분포", "예상", "종 모양", "예상"
    );
    for bits in [2, 3, 4, 5, 6, 8, 10, 12, 16] {
        let a = Quantizer::symmetric_for(&flat, bits);
        let b = Quantizer::symmetric_for(&bell, bits);
        println!(
            "{:>6} {:>8} {:>12.2} {:>12.2} {:>12.2} {:>12.2}",
            bits,
            2 * a.high + 1,
            sqnr_db(&flat, &round_trip(&flat, a)),
            predicted_sqnr(&flat, a.scale),
            sqnr_db(&bell, &round_trip(&bell, b)),
            predicted_sqnr(&bell, b.scale),
        );
    }

    println!();
    println!("2. 눈금의 끝을 어디에 둘까: 종 모양 분포의 SQNR (dB). 가장 큰 절댓값은 {largest:.2}");
    println!(
        "{:>14} {:>10} {:>10} {:>10} {:>10}",
        "눈금의 끝", "3비트", "4비트", "6비트", "8비트"
    );
    let limits = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, largest];
    for limit in limits {
        let label = if limit == largest {
            "가장 큰 값".to_string()
        } else {
            format!("표준편차 × {limit:.1}")
        };
        print!("{label:>14}");
        for bits in [3, 4, 6, 8] {
            let quantizer = Quantizer::symmetric(limit, bits);
            print!(" {:>10.2}", sqnr_db(&bell, &round_trip(&bell, quantizer)));
        }
        println!();
    }

    println!();
    println!("3. 유난히 큰 값 하나: 종 모양의 값 4096개, 4비트, SQNR (dB)");
    println!(
        "{:>22} {:>14} {:>14} {:>14}",
        "섞인 값", "전체에 스케일 하나", "256개마다", "32개마다"
    );
    // 큰 값 자신은 빼고, 나머지 값들이 얼마나 보존되는지 본다.
    let others = |values: &[f32]| [&values[..1000], &values[1001..]].concat();
    for outlier in [0.0_f32, 10.0, 30.0, 100.0] {
        let mut values = bell[..4096].to_vec();
        let label = if outlier > 0.0 {
            values[1000] = outlier;
            format!("표준편차 × {outlier:.0}")
        } else {
            "없음".to_string()
        };
        print!("{label:>22}");
        for granularity in [
            Granularity::Tensor,
            Granularity::Blocks(256),
            Granularity::Blocks(32),
        ] {
            let mut restored = values.clone();
            fake_quantize(&mut restored, 4, granularity);
            print!(" {:>14.2}", sqnr_db(&others(&values), &others(&restored)));
        }
        println!();
    }

    println!();
    println!("4. 한쪽으로 치우친 값: 종 모양 분포의 절댓값, SQNR (dB)");
    println!("{:>6} {:>12} {:>12}", "비트", "대칭", "비대칭");
    let one_sided: Vec<f32> = bell.iter().map(|v| v.abs()).collect();
    for bits in [4, 8] {
        let symmetric = Quantizer::symmetric_for(&one_sided, bits);
        let asymmetric = Quantizer::asymmetric_for(&one_sided, bits);
        println!(
            "{:>6} {:>12.2} {:>12.2}",
            bits,
            sqnr_db(&one_sided, &round_trip(&one_sided, symmetric)),
            sqnr_db(&one_sided, &round_trip(&one_sided, asymmetric)),
        );
    }
}
