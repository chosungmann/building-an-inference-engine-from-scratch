//! 18장: 서로 다른 변수인데도 스레드끼리 방해하는 경우를 본다.
//!
//! 실행: `cargo run --release -p badak_bench --example ch18_false_sharing`
//!
//! 스레드마다 자기 카운터만 올린다. 카운터들을 메모리에 붙여 놓았을 때와
//! 캐시 줄 하나씩 떼어 놓았을 때를 견준다.

use std::sync::atomic::{AtomicU64, Ordering};

use badak_bench::{Plan, Summary, measure};
use badak_core::threads::{DEFAULT_WAITING, ThreadPool};

const INCREMENTS: u64 = 20_000_000;

// ANCHOR: layout
/// 붙여 놓은 카운터. 8바이트씩 나란히 놓인다.
struct Packed(AtomicU64);

/// 떼어 놓은 카운터. 하나가 128바이트를 차지해서 캐시 줄을 혼자 쓴다.
#[repr(align(128))]
struct Padded(AtomicU64);
// ANCHOR_END: layout

// ANCHOR: count
/// 스레드마다 자기 번호의 카운터를 `INCREMENTS`번 올린다.
fn count_up(pool: &ThreadPool, threads: usize, counters: &[&AtomicU64]) {
    pool.run(threads, threads, &|index| {
        let counter = counters[index];
        for _ in 0..INCREMENTS {
            counter.fetch_add(1, Ordering::Relaxed);
        }
    });
}
// ANCHOR_END: count

fn main() {
    let pool = ThreadPool::new(12, DEFAULT_WAITING);
    let packed: Vec<Packed> = (0..12).map(|_| Packed(AtomicU64::new(0))).collect();
    let padded: Vec<Padded> = (0..12).map(|_| Padded(AtomicU64::new(0))).collect();
    let near: Vec<&AtomicU64> = packed.iter().map(|counter| &counter.0).collect();
    let far: Vec<&AtomicU64> = padded.iter().map(|counter| &counter.0).collect();
    let plan = Plan {
        warmup: 1,
        repetitions: 7,
    };

    println!("스레드마다 자기 카운터를 {INCREMENTS}번 올리는 데 걸리는 시간 (밀리초)");
    println!(
        "{:>8} {:>14} {:>14} {:>10}",
        "스레드", "붙여 놓음", "떼어 놓음", "배수"
    );
    for threads in [1, 2, 4, 8, 12] {
        let near = Summary::new(&measure(plan, || count_up(&pool, threads, &near))).median;
        let far = Summary::new(&measure(plan, || count_up(&pool, threads, &far))).median;
        println!(
            "{:>8} {:>14.1} {:>14.1} {:>9.1}배",
            threads,
            near * 1e3,
            far * 1e3,
            near / far
        );
    }
}
