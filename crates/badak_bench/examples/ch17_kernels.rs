//! 17장: 벡터 명령을 직접 쓴 커널을 16장의 커널, 기준 구현과 견준다.
//!
//! 실행: `cargo run --release -p badak_bench --example ch17_kernels`
//!
//! 가중치는 `(8192, 2048)`, 64MB다. 입력 벡터의 수를 바꿔 가며 곱셈 하나에
//! 드는 사이클을 잰다.

use std::hint::black_box;

use badak_bench::{Counters, Plan, measure};
use badak_core::kernel;
use badak_core::simd::{self, Backend, Walk};

const ROWS: usize = 8192;
const COLS: usize = 2048;

fn values(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|i| ((i * 31 + seed) % 23) as f32 * 0.0625 - 0.5)
        .collect()
}

/// 일을 여러 번 실행해 곱셈 하나에 든 사이클을 돌려준다.
fn cycles(multiplies: usize, repetitions: usize, mut work: impl FnMut() -> f32) -> f64 {
    let plan = Plan {
        warmup: 1,
        repetitions,
    };
    let before = Counters::now();
    measure(plan, &mut work);
    let runs = ((plan.warmup + plan.repetitions) * multiplies) as f64;
    before
        .zip(Counters::now())
        .map_or(f64::NAN, |(before, after)| {
            after.since(before).cycles as f64 / runs
        })
}

fn main() {
    let fast = Backend::detect();
    let w = values(ROWS * COLS, 1);
    println!("이 기계의 구현: {fast:?}. 가중치 (8192, 2048), 곱셈당 사이클");
    println!(
        "{:>10} {:>8} {:>10} {:>10} {:>10} {:>10}  고른 방법",
        "벡터의 수", "16장", "기준 구현", "네 줄기", "두 행씩", "네 행씩"
    );
    for count in [1, 2, 3, 4, 6, 8, 12, 16, 64, 256] {
        let x = values(count * COLS, 2);
        let mut y = vec![0.0_f32; count * ROWS];
        let multiplies = count * ROWS * COLS;
        let repetitions = (256 / count).clamp(3, 20);
        let old = cycles(multiplies, repetitions, || {
            kernel::linear(black_box(&x), count, black_box(&w), ROWS, COLS, &mut y);
            y[0]
        });
        let mut with = |backend: Backend, walk: Walk| {
            cycles(multiplies, repetitions, || {
                simd::linear_with(
                    backend,
                    walk,
                    black_box(&x),
                    count,
                    black_box(&w),
                    ROWS,
                    COLS,
                    &mut y,
                );
                y[0]
            })
        };
        let portable = with(Backend::Portable, Walk::TwoRows);
        let streams = with(fast, Walk::FourStreams);
        let two = with(fast, Walk::TwoRows);
        let four = with(fast, Walk::FourRows);
        println!(
            "{:>10} {:>8.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3}  {:?}",
            count,
            old,
            portable,
            streams,
            two,
            four,
            Walk::for_vectors(count)
        );
    }
}
