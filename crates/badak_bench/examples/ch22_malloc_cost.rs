//! 22장: 메모리를 한 번 얻고 돌려주는 데 드는 시간을 크기별로 잰다.
//!
//! 실행: `cargo run --release -p badak_bench --example ch22_malloc_cost`
//!
//! 두 가지를 잰다. 매번 새로 얻어 값을 채우고 돌려주는 경우와, 한 번 얻은
//! 것을 계속 다시 쓰는 경우다. 채우는 값은 둘 다 같다.

use std::hint::black_box;
use std::time::Instant;

// ANCHOR: work
/// `f32` `len`개짜리 벡터를 매번 새로 얻어 값을 채운다.
fn fresh(len: usize, rounds: usize) -> f64 {
    let start = Instant::now();
    for round in 0..rounds {
        let mut values = vec![0.0_f32; len];
        fill(&mut values, round);
        black_box(&values);
    }
    start.elapsed().as_secs_f64() / rounds as f64
}

/// 벡터 하나를 한 번만 얻어 계속 다시 쓴다.
fn reused(len: usize, rounds: usize) -> f64 {
    let mut values = vec![0.0_f32; len];
    let start = Instant::now();
    for round in 0..rounds {
        fill(&mut values, round);
        black_box(&values);
    }
    start.elapsed().as_secs_f64() / rounds as f64
}

fn fill(values: &mut [f32], round: usize) {
    for (index, value) in values.iter_mut().enumerate() {
        *value = (index + round) as f32;
    }
}
// ANCHOR_END: work

fn main() {
    println!("f32 벡터 하나를 채우는 데 걸리는 시간 (마이크로초)");
    println!(
        "{:>12} {:>10} {:>14} {:>14} {:>14}",
        "원소", "바이트", "매번 새로", "다시 쓰기", "차이"
    );
    for len in [16, 512, 8192, 131_072, 2_097_152, 16_777_216] {
        let rounds = (1 << 26) / len.max(4096);
        let rounds = rounds.clamp(8, 100_000);
        // 한 번씩 돌려 준비 운동을 한다.
        fresh(len, 2);
        let (a, b) = (fresh(len, rounds), reused(len, rounds));
        let bytes = len * 4;
        let size = if bytes >= 1 << 20 {
            format!("{}MB", bytes >> 20)
        } else if bytes >= 1 << 10 {
            format!("{}KB", bytes >> 10)
        } else {
            format!("{bytes}B")
        };
        println!(
            "{:>12} {:>10} {:>14.3} {:>14.3} {:>14.3}",
            len,
            size,
            a * 1e6,
            b * 1e6,
            (a - b) * 1e6
        );
    }
}
