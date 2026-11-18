//! 26장: 선형 층의 커널이 입력 벡터 하나일 때와 여럿일 때 내는 속도.
//!
//! 실행: `cargo run --release -p badak_graph --example ch26_single_vector`
//!
//! 사진을 한 장씩 넣으면 커널은 벡터 하나를 받는다. 그때의 속도를 벡터를
//! 여럿 받을 때와 견준다. 스레드는 하나만 쓴다.

use std::hint::black_box;

use badak_bench::stats::Summary;
use badak_bench::timing::{Plan, measure};
use badak_core::simd::{self, Backend, Walk};

fn main() {
    println!(
        "{:<12} {:>6} {:>14} {:>16}",
        "가중치", "벡터", "벡터 하나 (µs)", "곱셈 (억 번/초)"
    );
    for (rows, cols) in [(128, 784), (512, 256), (2048, 2048)] {
        let w: Vec<f32> = (0..rows * cols).map(|i| (i % 13) as f32 * 0.01).collect();
        for count in [1_usize, 2, 8] {
            let x: Vec<f32> = (0..count * cols).map(|i| (i % 7) as f32 * 0.1).collect();
            let mut y = vec![0.0_f32; count * rows];
            let repeats = (50_000_000 / (rows * cols * count)).max(1);
            let plan = Plan {
                warmup: 3,
                repetitions: 31,
            };
            let seconds = Summary::new(&measure(plan, || {
                for _ in 0..repeats {
                    simd::linear_with(
                        Backend::detect(),
                        Walk::for_vectors(count),
                        black_box(&x),
                        count,
                        &w,
                        rows,
                        cols,
                        &mut y,
                    );
                }
            }))
            .median
                / repeats as f64;
            println!(
                "{:<12} {:>6} {:>14.2} {:>16.0}",
                format!("{rows}×{cols}"),
                count,
                seconds / count as f64 * 1e6,
                (rows * cols * count) as f64 / seconds / 1e8
            );
        }
    }
}
