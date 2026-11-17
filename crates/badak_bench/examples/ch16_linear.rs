//! 16장: 선형 층 하나를 여러 방법으로 계산해, 방법마다의 효과를 잰다.
//!
//! 실행: `cargo run --release -p badak_bench --example ch16_linear -- [실험]`
//!
//! 실험은 `steps`(개선을 한 단계씩), `rows`(가중치를 몇 행씩 읽을지),
//! `blocks`(입력을 몇 개씩 끊을지), `matmul`(일반적인 행렬 곱) 가운데
//! 하나다. 주지 않으면 모두 한다.
//!
//! 가중치는 `(8192, 2048)`, 64MB다. Llama 3.2 1B의 MLP가 벡터를 펼치는 층과
//! 같은 크기이고, 어느 캐시에도 들어가지 않는다.

use std::hint::black_box;

use badak_bench::{Counters, Plan, Summary, measure};
use badak_core::kernel::{self, dots};
use badak_core::linalg;

const ROWS: usize = 8192;
const COLS: usize = 2048;

fn values(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|i| ((i * 31 + seed) % 23) as f32 * 0.0625 - 0.5)
        .collect()
}

fn report(label: &str, multiplies: usize, repetitions: usize, mut work: impl FnMut() -> f32) {
    let plan = Plan {
        warmup: 1,
        repetitions,
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
        "{:<40} {:>8.2} {:>8.3} {:>12.2}",
        label,
        instructions,
        cycles,
        multiplies as f64 / seconds / 1e9
    );
}

fn header(first: &str) {
    println!(
        "{:<40} {:>8} {:>8} {:>12}",
        first, "명령", "사이클", "십억 곱셈/초"
    );
}

// ANCHOR: steps
/// 1단계: 2장의 `matvec`을 벡터마다 부른다. 15장까지의 `linear`가 하던 일.
fn per_vector_naive(x: &[f32], count: usize, w: &[f32], y: &mut [f32]) {
    for t in 0..count {
        let out = linalg::matvec(w, ROWS, COLS, &x[t * COLS..(t + 1) * COLS]);
        y[t * ROWS..(t + 1) * ROWS].copy_from_slice(&out);
    }
}

/// 2단계: 내적만 새 커널로 바꾼다. 반복문의 순서는 그대로다. 벡터마다
/// 가중치 전체를 처음부터 끝까지 읽는다.
fn per_vector_lanes(x: &[f32], count: usize, w: &[f32], y: &mut [f32]) {
    for t in 0..count {
        for o in 0..ROWS {
            y[t * ROWS + o] =
                kernel::dot(&x[t * COLS..(t + 1) * COLS], &w[o * COLS..(o + 1) * COLS]);
        }
    }
}

/// 3단계: 반복문의 순서를 바꾼다. 가중치 한 행을 읽어 온 김에 모든 벡터와
/// 내적한다. 가중치를 한 번만 읽는다.
fn per_row(x: &[f32], count: usize, w: &[f32], y: &mut [f32]) {
    for o in 0..ROWS {
        for t in 0..count {
            y[t * ROWS + o] =
                kernel::dot(&x[t * COLS..(t + 1) * COLS], &w[o * COLS..(o + 1) * COLS]);
        }
    }
}
// ANCHOR_END: steps

fn steps() {
    let w = values(ROWS * COLS, 1);
    for count in [1, 256] {
        let x = values(count * COLS, 2);
        let mut y = vec![0.0_f32; count * ROWS];
        let multiplies = count * ROWS * COLS;
        let repetitions = if count == 1 { 20 } else { 3 };
        println!("벡터 {count}개 × 가중치 (8192, 2048), 곱셈 하나당");
        header("방법");
        report(
            "1. 2장의 matvec을 벡터마다",
            multiplies,
            repetitions,
            || {
                per_vector_naive(black_box(&x), count, black_box(&w), &mut y);
                y[0]
            },
        );
        report(
            "2. 내적의 합을 16개로 나눈다",
            multiplies,
            repetitions,
            || {
                per_vector_lanes(black_box(&x), count, black_box(&w), &mut y);
                y[0]
            },
        );
        report(
            "3. 가중치 한 행으로 모든 벡터를",
            multiplies,
            repetitions,
            || {
                per_row(black_box(&x), count, black_box(&w), &mut y);
                y[0]
            },
        );
        report(
            "4. kernel::linear (16장의 결과)",
            multiplies,
            repetitions,
            || {
                kernel::linear(black_box(&x), count, black_box(&w), ROWS, COLS, &mut y);
                y[0]
            },
        );
        println!();
    }
}

/// 가중치를 `N`행씩 읽으며 벡터 `count`개와 곱한다.
fn rows_at_once<const N: usize>(x: &[f32], count: usize, w: &[f32], y: &mut [f32]) {
    for (block, o) in w.chunks_exact(N * COLS).zip((0..).step_by(N)) {
        let weights: [&[f32]; N] = std::array::from_fn(|r| &block[r * COLS..(r + 1) * COLS]);
        for t in 0..count {
            let [results] = dots::<1, N, 16>([&x[t * COLS..(t + 1) * COLS]], weights);
            y[t * ROWS + o..t * ROWS + o + N].copy_from_slice(&results);
        }
    }
}

fn rows() {
    let w = values(ROWS * COLS, 1);
    println!("가중치를 몇 행씩 읽을 것인가 (곱셈당 사이클)");
    println!("{:>10} {:>12} {:>12}", "벡터의 수", "한 행씩", "네 행씩");
    for count in [1, 2, 3, 4, 5, 6, 7, 8, 16, 64] {
        let x = values(count * COLS, 2);
        let mut y = vec![0.0_f32; count * ROWS];
        let multiplies = (count * ROWS * COLS) as f64;
        let cycles = |work: &mut dyn FnMut()| {
            let plan = Plan {
                warmup: 1,
                repetitions: 10,
            };
            let before = Counters::now();
            measure(plan, &mut *work);
            before
                .zip(Counters::now())
                .map_or(f64::NAN, |(before, after)| {
                    after.since(before).cycles as f64 / (11.0 * multiplies)
                })
        };
        let one = cycles(&mut || rows_at_once::<1>(black_box(&x), count, black_box(&w), &mut y));
        let four = cycles(&mut || rows_at_once::<4>(black_box(&x), count, black_box(&w), &mut y));
        println!("{:>10} {:>12.3} {:>12.3}", count, one, four);
    }
}

fn blocks() {
    // 입력이 캐시보다 훨씬 큰 경우: 벡터 4096개, 32MB.
    let (count, rows) = (4096, 256);
    let w = values(rows * COLS, 1);
    let x = values(count * COLS, 2);
    let mut y = vec![0.0_f32; count * rows];
    println!("벡터 {count}개(32MB) × 가중치 ({rows}, 2048), 곱셈 하나당");
    header("한 번에 다루는 벡터");
    for block in [16, 64, 256, 512, 1024, 2048, 4096] {
        let bytes = block * COLS * 4;
        let label = if bytes >= 1 << 20 {
            format!("{block}개 ({}MB)", bytes >> 20)
        } else {
            format!("{block}개 ({}KB)", bytes >> 10)
        };
        report(&label, count * rows * COLS, 3, || {
            kernel::linear_blocked(
                black_box(&x),
                count,
                black_box(&w),
                rows,
                COLS,
                &mut y,
                block,
            );
            y[0]
        });
    }
}

// ANCHOR: matmul
/// 일반적인 행렬 곱 `A × B`. 2장의 구현과, `B`를 뒤집어 옮겨 적은 뒤 선형
/// 층의 커널을 쓰는 구현을 견준다.
fn matmuls() {
    let size = 1024;
    let a = values(size * size, 1);
    let b = values(size * size, 2);
    println!("{size} × {size} 행렬 둘의 곱");
    println!("{:<34} {:>12}", "방법", "밀리초");
    let time = |work: &mut dyn FnMut() -> f32| {
        let plan = Plan {
            warmup: 1,
            repetitions: 5,
        };
        Summary::new(&measure(plan, work)).median * 1e3
    };
    let naive = time(&mut || linalg::matmul(black_box(&a), black_box(&b), size, size, size)[0]);
    println!("{:<34} {:>12.1}", "2장의 matmul", naive);

    // 옮겨 적기만: B의 열이 행이 되도록 뒤집는다.
    let transpose = |b: &[f32]| {
        let mut out = vec![0.0_f32; size * size];
        for row in 0..size {
            for col in 0..size {
                out[col * size + row] = b[row * size + col];
            }
        }
        out
    };
    let packing = time(&mut || transpose(black_box(&b))[1]);
    println!("{:<34} {:>12.1}", "B를 뒤집어 옮겨 적기만", packing);

    let mut c = vec![0.0_f32; size * size];
    let packed = time(&mut || {
        let b_transposed = transpose(black_box(&b));
        kernel::linear(black_box(&a), size, &b_transposed, size, size, &mut c);
        c[0]
    });
    println!("{:<34} {:>12.1}", "옮겨 적고 kernel::linear", packed);
}
// ANCHOR_END: matmul

fn main() {
    let experiment = std::env::args().nth(1);
    let wants = |name: &str| experiment.is_none() || experiment.as_deref() == Some(name);
    if wants("steps") {
        steps();
    }
    if wants("rows") {
        rows();
        println!();
    }
    if wants("blocks") {
        blocks();
        println!();
    }
    if wants("matmul") {
        matmuls();
    }
}
