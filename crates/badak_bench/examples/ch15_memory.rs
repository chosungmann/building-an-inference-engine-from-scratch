//! 15장: 이 기계가 메모리에서 1초에 얼마나 읽을 수 있는지 잰다.
//!
//! 실행: `cargo run --release -p badak_bench --example ch15_memory -- [실험]`
//!
//! 실험은 `sizes`(배열의 크기를 바꿔 가며)와 `threads`(스레드의 수를 바꿔
//! 가며) 가운데 하나다. 주지 않으면 둘 다 한다.

use std::hint::black_box;
use std::sync::Barrier;
use std::time::Instant;

// ANCHOR: read
/// 배열을 처음부터 끝까지 읽는다. 읽은 값을 모두 더해 돌려준다.
///
/// 정수의 덧셈은 순서를 바꿔도 결과가 같아서, 컴파일러가 값 여러 개를 한꺼번에
/// 더하는 코드를 만든다. 계산은 거의 공짜가 되고 읽는 속도만 남는다.
fn read_all(data: &[u64]) -> u64 {
    data.iter().fold(0, |sum, &value| sum.wrapping_add(value))
}

/// 배열을 `passes`번 읽고, 1초에 읽은 양을 GB로 돌려준다. 여러 번 재서 가장
/// 빠른 값을 고른다.
fn read_speed(data: &[u64], passes: usize) -> f64 {
    let bytes = (size_of_val(data) * passes) as f64;
    (0..5)
        .map(|_| {
            let start = Instant::now();
            for _ in 0..passes {
                black_box(read_all(black_box(data)));
            }
            bytes / start.elapsed().as_secs_f64() / 1e9
        })
        .fold(0.0, f64::max)
}
// ANCHOR_END: read

/// 값이 들어 있는 배열을 만든다. 모든 쪽에 한 번씩 닿게 해서, 측정 중에
/// 운영체제가 메모리를 붙이는 일이 없게 한다(14.3절).
fn filled(bytes: usize) -> Vec<u64> {
    (0..bytes / 8).map(|i| i as u64).collect()
}

fn sizes() {
    println!("배열의 크기에 따른 읽기 속도 (스레드 1개)");
    println!("{:>12} {:>12}", "크기", "GB/초");
    let mut bytes = 16 << 10;
    while bytes <= 1 << 30 {
        let data = filled(bytes);
        // 작은 배열은 여러 번 읽어 한 번의 측정이 256MB쯤 되게 한다.
        let passes = ((256 << 20) / bytes).max(1);
        let label = if bytes >= 1 << 20 {
            format!("{}MB", bytes >> 20)
        } else {
            format!("{}KB", bytes >> 10)
        };
        println!("{:>12} {:>12.1}", label, read_speed(&data, passes));
        bytes *= 2;
    }
}

// ANCHOR: threads
/// 스레드 `count`개가 저마다의 배열을 동시에 읽을 때, 모두 합쳐 1초에 읽는 양.
fn parallel_read_speed(count: usize, bytes_each: usize) -> f64 {
    let arrays: Vec<Vec<u64>> = (0..count).map(|_| filled(bytes_each)).collect();
    let passes = 8;
    (0..3)
        .map(|_| {
            // 모든 스레드가 준비된 뒤 한꺼번에 출발하게 한다.
            let barrier = Barrier::new(count + 1);
            std::thread::scope(|scope| {
                for data in &arrays {
                    let barrier = &barrier;
                    scope.spawn(move || {
                        barrier.wait();
                        for _ in 0..passes {
                            black_box(read_all(black_box(data)));
                        }
                    });
                }
                barrier.wait();
                let start = Instant::now();
                // 범위가 끝날 때 모든 스레드가 끝나기를 기다린다.
                start
            })
            .elapsed()
            .as_secs_f64()
        })
        .map(|seconds| (count * bytes_each * passes) as f64 / seconds / 1e9)
        .fold(0.0, f64::max)
}
// ANCHOR_END: threads

fn threads() {
    println!("스레드의 수에 따른 읽기 속도 (스레드마다 256MB)");
    println!("{:>8} {:>12} {:>14}", "스레드", "GB/초", "스레드당 GB/초");
    for count in [1, 2, 4, 6, 8, 10, 12, 14, 16] {
        let speed = parallel_read_speed(count, 256 << 20);
        println!(
            "{:>8} {:>12.1} {:>14.1}",
            count,
            speed,
            speed / count as f64
        );
    }
}

fn main() {
    let experiment = std::env::args().nth(1);
    if experiment.as_deref() != Some("threads") {
        sizes();
        println!();
    }
    if experiment.as_deref() != Some("sizes") {
        threads();
    }
}
