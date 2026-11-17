//! 16장: 커널의 모양을 측정으로 고른다.
//!
//! 실행: `cargo run --release -p badak_bench --example ch16_shapes`
//!
//! `kernel::dots`는 내적 `M × N`개를 한꺼번에 계산하고, 내적 하나의 합을
//! `L`개의 변수에 나눈다. 세 숫자를 바꿔 가며 곱셈 하나에 드는 명령과
//! 사이클을 잰다. 벡터는 모두 가장 빠른 캐시에 들어가는 크기다.

use std::hint::black_box;

use badak_bench::{Counters, Plan, Summary, measure};
use badak_core::kernel::dots;
use badak_core::linalg::dot;

const LEN: usize = 2048;

fn vectors(count: usize, seed: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|v| {
            (0..LEN)
                .map(|i| ((i * 31 + (v + seed) * 17) % 19) as f32 * 0.125)
                .collect()
        })
        .collect()
}

fn report(label: &str, multiplies: usize, mut work: impl FnMut() -> f32) {
    let plan = Plan {
        warmup: 2000,
        repetitions: 20000,
    };
    let before = Counters::now();
    let samples = measure(plan, &mut work);
    let spent = before
        .zip(Counters::now())
        .map(|(before, after)| after.since(before));
    let seconds = Summary::new(&samples).median;
    let runs = ((plan.warmup + plan.repetitions) * multiplies) as f64;
    let (instructions, cycles) = spent.map_or((f64::NAN, f64::NAN), |c| {
        (c.instructions as f64 / runs, c.cycles as f64 / runs)
    });
    println!(
        "{:<26} {:>10.2} {:>10.3} {:>14.1}",
        label,
        instructions,
        cycles,
        multiplies as f64 / seconds / 1e9
    );
}

/// `M × N × L` 모양의 커널을 잰다.
fn shape<const M: usize, const N: usize, const L: usize>() {
    let (xs, ws) = (vectors(M, 0), vectors(N, 100));
    let x: [&[f32]; M] = std::array::from_fn(|r| xs[r].as_slice());
    let w: [&[f32]; N] = std::array::from_fn(|c| ws[c].as_slice());
    report(&format!("{M} × {N}, 합 {L}개"), M * N * LEN, || {
        let out = dots::<M, N, L>(black_box(x), black_box(w));
        out[M - 1][N - 1]
    });
}

fn main() {
    println!("길이 {LEN}인 벡터들의 내적, 곱셈 하나당");
    println!(
        "{:<26} {:>10} {:>10} {:>14}",
        "커널의 모양", "명령", "사이클", "십억 곱셈/초"
    );
    let (a, b) = (vectors(1, 0), vectors(1, 100));
    report("2장의 dot", LEN, || {
        dot(black_box(&a[0]), black_box(&b[0]))
    });
    shape::<1, 1, 4>();
    shape::<1, 1, 8>();
    shape::<1, 1, 16>();
    shape::<1, 1, 32>();
    println!();
    shape::<1, 2, 16>();
    shape::<1, 3, 16>();
    shape::<1, 4, 16>();
    shape::<1, 6, 16>();
    shape::<1, 8, 16>();
    shape::<1, 2, 32>();
    shape::<1, 4, 32>();
    shape::<1, 4, 8>();
    shape::<1, 8, 8>();
    println!();
    shape::<2, 2, 16>();
    shape::<2, 4, 16>();
    shape::<4, 4, 16>();
    shape::<2, 4, 8>();
    shape::<3, 4, 8>();
    shape::<4, 4, 8>();
    shape::<4, 8, 4>();
}
