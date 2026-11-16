//! 5장: 더하는 방법에 따라 오차가 어떻게 쌓이는지 잰다.
//!
//! 실행: `cargo run --release -p badak_core --example ch05_sum_error`
//!
//! 0 이상 1 미만의 의사 난수 n개를 f32로 더하고, 같은 수들을 f64로 더한
//! 값을 정답으로 삼아 상대 오차를 구한다.

use badak_core::linalg::{sum_naive, sum_pairwise};

/// 재현 가능한 의사 난수. 값은 [0, 1) 범위다.
fn pseudo_random(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 40) as f32 / (1_u64 << 24) as f32
        })
        .collect()
}

fn main() {
    println!(
        "{:>12} {:>14} {:>14} {:>14}",
        "n", "정답(f64)", "단순 합산", "쌍별 합산"
    );
    for exponent in 2..=8 {
        let n = 10_usize.pow(exponent);
        let values = pseudo_random(n, 42);
        let exact: f64 = values.iter().map(|&v| f64::from(v)).sum();
        let relative_error = |sum: f32| ((f64::from(sum) - exact) / exact).abs();
        println!(
            "{:>12} {:>14.3} {:>14.3e} {:>14.3e}",
            n,
            exact,
            relative_error(sum_naive(&values)),
            relative_error(sum_pairwise(&values))
        );
    }
}
