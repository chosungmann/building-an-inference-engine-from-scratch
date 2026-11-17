//! 20장: 가중치를 저장한 방식마다 커널의 속도를 잰다.
//!
//! 실행: `cargo run --release -p badak_bench --example ch20_kernels`
//!
//! 가중치는 `(8192, 2048)`이고 스레드는 하나다. 곱셈 하나에 드는 사이클과
//! 명령을 잰다. 메모리에서 읽는 양이 방식마다 다르다.

use std::hint::black_box;

use badak_bench::{Counters, Plan, measure};
use badak_core::packed::{Format, Kernel, Packed};
use badak_core::quant::Bf16;
use badak_core::random::Pcg64;
use badak_core::simd::{self, Backend, Walk};

const ROWS: usize = 8192;
const COLS: usize = 2048;

fn values(count: usize, seed: u64) -> Vec<f32> {
    let mut rng = Pcg64::new(seed);
    (0..count)
        .map(|_| ((rng.next_f64() - 0.5) * 0.1) as f32)
        .collect()
}

/// 일을 여러 번 실행해 곱셈 하나에 든 (사이클, 명령)을 돌려준다.
fn cost(multiplies: usize, repetitions: usize, mut work: impl FnMut() -> f32) -> (f64, f64) {
    let plan = Plan {
        warmup: 1,
        repetitions,
    };
    let before = Counters::now();
    measure(plan, &mut work);
    let runs = ((plan.warmup + plan.repetitions) * multiplies) as f64;
    before
        .zip(Counters::now())
        .map_or((f64::NAN, f64::NAN), |(before, after)| {
            let used = after.since(before);
            (used.cycles as f64 / runs, used.instructions as f64 / runs)
        })
}

fn main() {
    let backend = Backend::detect();
    // BF16으로 정확히 나타낼 수 있는 값으로 맞춰 둔다. 진짜 가중치가 그렇다.
    let halves: Vec<Bf16> = values(ROWS * COLS, 1)
        .into_iter()
        .map(Bf16::from_f32)
        .collect();
    let w: Vec<f32> = halves.iter().map(|half| half.to_f32()).collect();
    let q8 = Packed::quantize(Format::Q8_0, &w, ROWS, COLS).expect("32의 배수다");
    let q4 = Packed::quantize(Format::Q4_0, &w, ROWS, COLS).expect("32의 배수다");

    println!("가중치 (8192, 2048), 스레드 1개. 곱셈 하나에 드는 사이클 (괄호 안은 명령)");
    println!(
        "{:<26} {:>10} {:>18} {:>18}",
        "가중치", "크기 (MB)", "벡터 1개", "벡터 64개"
    );
    let megabytes = |bytes: usize| bytes as f64 / 1e6;
    #[allow(clippy::type_complexity)]
    let row = |label: &str, size: f64, run: &mut dyn FnMut(&[f32], usize, &mut [f32])| {
        print!("{label:<26} {size:>10.1}");
        for count in [1, 64] {
            let x = values(count * COLS, 2);
            let mut y = vec![0.0_f32; count * ROWS];
            let repetitions = if count == 1 { 30 } else { 3 };
            let (cycles, instructions) = cost(count * ROWS * COLS, repetitions, || {
                run(black_box(&x), count, &mut y);
                y[0]
            });
            print!(" {:>18}", format!("{cycles:.3} ({instructions:.2})"));
        }
        println!();
    };
    row("f32", megabytes(w.len() * 4), &mut |x, count, y| {
        simd::linear_with(
            backend,
            Walk::for_vectors(count),
            x,
            count,
            &w,
            ROWS,
            COLS,
            y,
        );
    });
    row("BF16", megabytes(halves.len() * 2), &mut |x, count, y| {
        simd::linear_with(
            backend,
            Walk::for_vectors(count),
            x,
            count,
            &halves,
            ROWS,
            COLS,
            y,
        );
    });
    for (label, packed) in [("Q8_0", &q8), ("Q4_0", &q4)] {
        let size = megabytes(packed.bytes().len());
        row(
            &format!("{label}, 기준 구현"),
            size,
            &mut |x, count, y| {
                packed.linear_with(Kernel::Portable, 1, x, count, y);
            },
        );
        row(
            &format!("{label}, 벡터 명령"),
            size,
            &mut |x, count, y| {
                packed.linear_with(Kernel::Fast, 1, x, count, y);
            },
        );
    }
}
