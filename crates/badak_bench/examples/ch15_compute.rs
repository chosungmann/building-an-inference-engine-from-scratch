//! 15장: 계산 쪽의 한계와, 지금의 행렬 곱이 메모리에 얼마나 매여 있는지 잰다.
//!
//! 실행: `cargo run --release -p badak_bench --example ch15_compute -- [실험]`
//!
//! 실험은 `dot`(내적을 짜는 방법에 따른 속도), `matvec`(행렬의 크기에 따른
//! 속도), `columns`(한 행의 길이에 따른 속도) 가운데 하나다. 주지 않으면
//! 모두 한다.

use std::hint::black_box;

use badak_bench::{Counters, Plan, Summary, measure};
use badak_core::linalg::{dot, matvec};

// ANCHOR: lanes
/// 합을 `N`개의 변수에 나누어 담는 내적.
///
/// 2장의 `dot`은 합 하나에 차례로 더한다. 덧셈 하나가 끝나야 다음 덧셈을
/// 시작할 수 있다. 합을 여러 개로 나누면 서로 기다리지 않는 덧셈이 `N`개
/// 생긴다. 더하는 순서가 달라지므로 결과의 마지막 자리는 `dot`과 다를 수
/// 있다.
fn dot_lanes<const N: usize>(a: &[f32], b: &[f32]) -> f32 {
    let mut sums = [0.0_f32; N];
    let (a_chunks, a_rest) = a.as_chunks::<N>();
    let (b_chunks, b_rest) = b.as_chunks::<N>();
    for (x, y) in a_chunks.iter().zip(b_chunks) {
        for lane in 0..N {
            sums[lane] += x[lane] * y[lane];
        }
    }
    let rest: f32 = a_rest.iter().zip(b_rest).map(|(x, y)| x * y).sum();
    sums.iter().sum::<f32>() + rest
}
// ANCHOR_END: lanes

/// 일을 재서 (곱셈 하나의 나노초, 명령, 사이클, 1초에 하는 곱셈)을 출력한다.
fn report(label: &str, multiplies: usize, plan: Plan, mut work: impl FnMut() -> f32) {
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
        "{:<22} {:>10.3} {:>10.2} {:>10.2} {:>12.2}",
        label,
        seconds / multiplies as f64 * 1e9,
        instructions,
        cycles,
        multiplies as f64 / seconds / 1e9
    );
}

fn header(first: &str) {
    println!(
        "{:<22} {:>10} {:>10} {:>10} {:>12}",
        first, "나노초", "명령", "사이클", "십억 곱셈/초"
    );
}

fn dots() {
    // 길이 4096인 벡터 둘. 합쳐 32KB라 가장 빠른 캐시에 다 들어간다.
    let len = 4096;
    let a: Vec<f32> = (0..len).map(|i| (i % 13) as f32 * 0.125).collect();
    let b: Vec<f32> = (0..len).map(|i| (i % 7) as f32 * 0.25).collect();
    let plan = Plan {
        warmup: 1000,
        repetitions: 20000,
    };
    println!("내적을 짜는 방법에 따른 속도 (길이 {len}, 곱셈 하나당)");
    header("합을 담는 변수의 수");
    report("1 (2장의 dot)", len, plan, || {
        dot(black_box(&a), black_box(&b))
    });
    report("4", len, plan, || {
        dot_lanes::<4>(black_box(&a), black_box(&b))
    });
    report("8", len, plan, || {
        dot_lanes::<8>(black_box(&a), black_box(&b))
    });
    report("16", len, plan, || {
        dot_lanes::<16>(black_box(&a), black_box(&b))
    });
    report("32", len, plan, || {
        dot_lanes::<32>(black_box(&a), black_box(&b))
    });
    report("64", len, plan, || {
        dot_lanes::<64>(black_box(&a), black_box(&b))
    });
}

fn matvecs() {
    // 열의 수는 Llama 3.2 1B의 hidden_size와 같게 두고 행의 수를 늘려 간다.
    let cols = 2048;
    let x: Vec<f32> = (0..cols).map(|i| (i % 7) as f32 * 0.25).collect();
    println!("행렬의 크기에 따른 행렬 × 벡터의 속도 (열 {cols}개, 곱셈 하나당)");
    header("행렬의 크기");
    let mut rows = 4;
    while rows * cols * 4 <= 2 << 30 {
        let bytes = rows * cols * 4;
        let matrix: Vec<f32> = (0..rows * cols).map(|i| (i % 11) as f32 * 0.125).collect();
        // 작은 행렬은 여러 번 되풀이해 한 번의 측정이 너무 짧지 않게 한다.
        let repetitions = ((64 << 20) / bytes).clamp(3, 2000);
        let plan = Plan {
            warmup: 1,
            repetitions,
        };
        let label = if bytes >= 1 << 20 {
            format!("{}MB", bytes >> 20)
        } else {
            format!("{}KB", bytes >> 10)
        };
        report(&label, rows * cols, plan, || {
            matvec(black_box(&matrix), rows, cols, black_box(&x))[0]
        });
        rows *= 4;
    }
}

fn columns() {
    // 행렬 전체의 크기는 128MB로 두고 한 행의 길이만 바꾼다.
    let total = 32 << 20;
    println!("한 행의 길이에 따른 행렬 × 벡터의 속도 (행렬 128MB, 곱셈 하나당)");
    header("한 행의 길이");
    for cols in [64, 256, 576, 1536, 2048, 8192, 32768] {
        let rows = total / cols;
        let matrix: Vec<f32> = (0..rows * cols).map(|i| (i % 11) as f32 * 0.125).collect();
        let x: Vec<f32> = (0..cols).map(|i| (i % 7) as f32 * 0.25).collect();
        let plan = Plan {
            warmup: 1,
            repetitions: 5,
        };
        report(&cols.to_string(), rows * cols, plan, || {
            matvec(black_box(&matrix), rows, cols, black_box(&x))[0]
        });
    }
}

fn main() {
    let experiment = std::env::args().nth(1);
    if experiment.is_none() || experiment.as_deref() == Some("dot") {
        dots();
        println!();
    }
    if experiment.is_none() || experiment.as_deref() == Some("matvec") {
        matvecs();
        println!();
    }
    if experiment.is_none() || experiment.as_deref() == Some("columns") {
        columns();
    }
}
