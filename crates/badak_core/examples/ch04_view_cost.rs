//! 4장: 뷰와 복사의 비용, 그리고 읽는 순서가 속도에 미치는 영향을 잰다.
//!
//! 실행: `cargo run --release -p badak_core --example ch04_view_cost`
//!
//! 초시계 수준의 거친 측정이다. 제대로 된 측정 방법은 14장에서 다룬다.

use std::hint::black_box;
use std::time::Instant;

use badak_core::{Result, Tensor};

/// 여러 번 실행해 가장 빠른 시간을 초 단위로 돌려준다.
fn best_of<R>(repeats: usize, mut f: impl FnMut() -> R) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..repeats {
        let start = Instant::now();
        black_box(f());
        best = best.min(start.elapsed().as_secs_f64());
    }
    best
}

fn main() -> Result<()> {
    const N: usize = 4096;
    let t = Tensor::from_vec((0..N * N).map(|i| (i % 7) as f32).collect(), &[N, N])?;
    println!(
        "{N} × {N} f32 텐서, {} MiB",
        t.size_in_bytes() / (1024 * 1024)
    );

    println!();
    println!("== 전치: 뷰와 복사 ==");
    let view = best_of(1000, || t.transpose(0, 1).unwrap());
    let copy = best_of(5, || t.transpose(0, 1).unwrap().contiguous());
    println!("{:<34} {:>12.1} ns", "뷰만 만들기", view * 1e9);
    println!("{:<34} {:>12.1} ms", "새 저장소로 복사하기", copy * 1e3);

    println!();
    println!("== 모든 원소 더하기: 읽는 순서의 영향 ==");
    let transposed = t.transpose(0, 1)?;
    let copied = transposed.contiguous();
    let rows = best_of(5, || t.as_slice::<f32>().unwrap().iter().sum::<f32>());
    let rows_iter = best_of(5, || t.iter::<f32>().unwrap().sum::<f32>());
    let columns_iter = best_of(5, || transposed.iter::<f32>().unwrap().sum::<f32>());
    let copied_iter = best_of(5, || copied.iter::<f32>().unwrap().sum::<f32>());
    println!("{:<34} {:>12.1} ms", "원본, 슬라이스로", rows * 1e3);
    println!("{:<34} {:>12.1} ms", "원본, 반복자로", rows_iter * 1e3);
    println!(
        "{:<34} {:>12.1} ms",
        "전치 뷰, 반복자로",
        columns_iter * 1e3
    );
    println!(
        "{:<34} {:>12.1} ms",
        "전치 후 복사본, 반복자로",
        copied_iter * 1e3
    );
    Ok(())
}
