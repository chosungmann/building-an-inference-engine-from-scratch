//! 19장: 실수를 담는 세 형식(f32, f16, BF16)을 견준다.
//!
//! 실행: `cargo run --release -p badak_core --example ch19_formats`

use badak_core::quant::{bf16_from_f32, bf16_to_f32, f16_from_f32, f16_to_f32};

/// 값을 16비트로 바꿨다가 되돌린다.
type RoundTrip = fn(f32) -> f32;

fn via_f16(value: f32) -> f32 {
    f16_to_f32(f16_from_f32(value))
}

fn via_bf16(value: f32) -> f32 {
    bf16_to_f32(bf16_from_f32(value))
}

fn main() {
    println!("1. 형식의 생김새");
    println!(
        "{:<8} {:>6} {:>6} {:>6} {:>14} {:>14} {:>16} {:>14}",
        "형식",
        "부호",
        "지수",
        "가수",
        "가장 큰 값",
        "가장 작은 값",
        "1 다음 수와의 간격",
        "1과 2 사이"
    );
    let rows: [(&str, u32, u32, f32, f32, f32); 3] = [
        ("f32", 8, 23, f32::MAX, f32::from_bits(1), f32::EPSILON),
        (
            "f16",
            5,
            10,
            f16_to_f32(0x7bff),
            f16_to_f32(0x0001),
            f16_to_f32(0x3c01) - 1.0,
        ),
        (
            "BF16",
            8,
            7,
            bf16_to_f32(0x7f7f),
            bf16_to_f32(0x0001),
            bf16_to_f32(0x3f81) - 1.0,
        ),
    ];
    for (name, exponent, mantissa, max, min, gap) in rows {
        println!(
            "{:<8} {:>6} {:>6} {:>6} {:>14.4e} {:>14.4e} {:>16.3e} {:>12}개",
            name,
            1,
            exponent,
            mantissa,
            max,
            min,
            gap,
            1_u32 << mantissa
        );
    }

    println!();
    println!("2. 같은 값을 세 형식에 담으면");
    println!(
        "{:>14} {:>16} {:>11} {:>16} {:>11}",
        "f32", "f16", "상대 오차", "BF16", "상대 오차"
    );
    let show = |value: f32, convert: RoundTrip| {
        let stored = convert(value);
        let error = ((f64::from(stored) - f64::from(value)) / f64::from(value)).abs();
        let error = if stored.is_infinite() {
            "무한대".to_string()
        } else if stored == 0.0 && value != 0.0 {
            "0이 됨".to_string()
        } else {
            format!("{error:.1e}")
        };
        format!("{stored:>16.8e} {error:>11}")
    };
    for value in [
        1.0_f32,
        0.1,
        std::f32::consts::PI,
        0.02,
        1.0e-3,
        1.0e-5,
        1.0e-8,
        1000.5,
        70000.0,
        1.0e30,
    ] {
        println!(
            "{:>14.8e} {} {}",
            value,
            show(value, via_f16),
            show(value, via_bf16)
        );
    }

    println!();
    println!("3. 이웃한 두 값 사이의 간격 (값의 크기에 따라)");
    println!("{:>10} {:>14} {:>14} {:>14}", "값", "f32", "f16", "BF16");
    for exponent in [-20_i32, -14, -10, -5, 0, 5, 10, 15] {
        let value = 2.0_f32.powi(exponent);
        let f32_gap = f32::from_bits(value.to_bits() + 1) - value;
        let half = f16_from_f32(value);
        let f16_gap = if half == 0 || half >= 0x7bff {
            "(담지 못함)".to_string()
        } else {
            format!("{:.3e}", f16_to_f32(half + 1) - f16_to_f32(half))
        };
        let bf16_gap = bf16_to_f32(bf16_from_f32(value) + 1) - value;
        println!(
            "{:>10} {:>14.3e} {:>14} {:>14.3e}",
            format!("2^{exponent}"),
            f32_gap,
            f16_gap,
            bf16_gap
        );
    }
}
