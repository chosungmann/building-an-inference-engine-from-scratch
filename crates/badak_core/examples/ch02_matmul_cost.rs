//! 2장: 행렬 곱의 계산량을 세고, 단순한 구현의 시간을 잰다.
//!
//! 실행: `cargo run --release -p badak_core --example ch02_matmul_cost`
//!
//! 여기서 쓰는 방법은 초시계 수준의 거친 측정이다. 제대로 된 측정 방법은
//! 14장에서 다룬다.

use std::hint::black_box;
use std::time::Instant;

use badak_core::linalg::matmul;

/// 재현 가능한 의사 난수로 채운 벡터. 값은 [-1, 1) 범위다.
fn pseudo_random(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            // 선형 합동 생성기. 품질은 낮지만 재현 가능한 입력으로는 충분하다.
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f32 / (1_u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn main() {
    const REPEATS: usize = 5;
    println!(
        "{:>6} {:>16} {:>12} {:>12} {:>10}",
        "n", "곱셈 횟수", "메모리(MiB)", "시간(ms)", "GFLOP/s"
    );
    for n in [64, 128, 256, 512, 1024] {
        let a = pseudo_random(n * n, 1);
        let b = pseudo_random(n * n, 2);

        // 여러 번 실행해 가장 빠른 시간을 취한다.
        let mut best = f64::INFINITY;
        for _ in 0..REPEATS {
            let start = Instant::now();
            let c = matmul(black_box(&a), black_box(&b), n, n, n);
            best = best.min(start.elapsed().as_secs_f64());
            black_box(c);
        }

        let multiplications = (n * n * n) as f64;
        // 곱셈 한 번과 덧셈 한 번을 부동소수점 연산 2회로 센다.
        let gflops = 2.0 * multiplications / best / 1e9;
        // 행렬 세 개(A, B, C), 원소 하나에 4바이트.
        let mebibytes = (3 * n * n * 4) as f64 / (1024.0 * 1024.0);
        println!(
            "{:>6} {:>16} {:>12.2} {:>12.2} {:>10.2}",
            n,
            multiplications,
            mebibytes,
            best * 1e3,
            gflops
        );
    }
}
