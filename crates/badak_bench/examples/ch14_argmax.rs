//! 14장: `argmax`가 왜 느린지 재서 알아낸다 (12.8절의 숙제).
//!
//! 실행: `cargo run --release -p badak_bench --example ch14_argmax`
//!
//! 같은 일을 하는 구현 넷을 같은 입력으로 재고, 값 하나를 처리하는 데 든
//! 시간, 명령, 사이클을 견준다.

use std::hint::black_box;

use badak_bench::{Counters, Plan, Summary, measure};
use badak_core::prob::argmax;
use badak_core::random::Pcg64;

// ANCHOR: variants
/// 12장까지의 구현. 가장 좋은 후보를 `Option`에 담고, `NaN`을 따로 거른다.
fn argmax_option(values: &[f32]) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (index, &value) in values.iter().enumerate() {
        if value.is_nan() {
            continue;
        }
        if best.is_none_or(|(_, best_value)| value > best_value) {
            best = Some((index, value));
        }
    }
    best.map(|(index, _)| index)
}

/// 후보를 `Option` 없이 두 변수에 담는다. `NaN > x`는 늘 거짓이므로 `NaN`은
/// 따로 거르지 않아도 뽑히지 않는다.
fn argmax_plain(values: &[f32]) -> Option<usize> {
    let (mut best_index, mut best_value) = (usize::MAX, f32::NEG_INFINITY);
    for (index, &value) in values.iter().enumerate() {
        if value > best_value {
            (best_index, best_value) = (index, value);
        }
    }
    (best_index != usize::MAX).then_some(best_index)
}

/// 두 번에 나눈다. 먼저 가장 큰 값만 찾고, 그다음 그 값이 있는 자리를 찾는다.
/// 첫 반복문에는 "어디였는지"를 기억하는 일이 없다.
fn argmax_two_pass(values: &[f32]) -> Option<usize> {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    values.iter().position(|&value| value == max)
}
// ANCHOR_END: variants

fn main() {
    // 12장의 점수와 같은 크기의 입력. 값은 아무 수다.
    let count = 128_256;
    let mut rng = Pcg64::new(14);
    let values: Vec<f32> = (0..count)
        .map(|_| (rng.next_f64() * 20.0 - 10.0) as f32)
        .collect();
    let expected = argmax_option(&values);

    println!("값 {count}개에서 가장 큰 값의 자리 찾기");
    println!(
        "{:<26} {:>10} {:>12} {:>12} {:>8}",
        "구현", "마이크로초", "값당 명령", "값당 사이클", "IPC"
    );
    let plan = Plan {
        warmup: 50,
        repetitions: 500,
    };
    type Variant = (&'static str, fn(&[f32]) -> Option<usize>);
    let variants: [Variant; 4] = [
        ("12장까지의 구현", argmax_option),
        ("Option 없이", argmax_plain),
        ("두 번에 나누어", argmax_two_pass),
        ("지금의 prob::argmax", argmax),
    ];
    for (label, function) in variants {
        assert_eq!(function(&values), expected, "{label}");
        let before = Counters::now();
        let samples = measure(plan, || function(black_box(&values)));
        let spent = before
            .zip(Counters::now())
            .map(|(before, after)| after.since(before));
        let runs = (plan.warmup + plan.repetitions) as f64 * count as f64;
        let (instructions, cycles, ipc) = spent.map_or((f64::NAN, f64::NAN, f64::NAN), |c| {
            (
                c.instructions as f64 / runs,
                c.cycles as f64 / runs,
                c.instructions_per_cycle(),
            )
        });
        println!(
            "{:<26} {:>10.1} {:>12.2} {:>12.2} {:>8.2}",
            label,
            Summary::new(&samples).median * 1e6,
            instructions,
            cycles,
            ipc
        );
    }
}
