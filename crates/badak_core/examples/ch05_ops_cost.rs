//! 5장: 같은 연산이 어느 길로 처리되느냐에 따라 걸리는 시간을 잰다.
//!
//! 실행: `cargo run --release -p badak_core --example ch05_ops_cost`
//!
//! 초시계 수준의 거친 측정이다. 제대로 된 측정 방법은 14장에서 다룬다.

use std::hint::black_box;
use std::time::Instant;

use badak_core::{Result, Tensor, linalg, ops};

/// 여러 번 실행해 가장 빠른 시간을 밀리초 단위로 돌려준다.
fn best_of<R>(repeats: usize, mut f: impl FnMut() -> R) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..repeats {
        let start = Instant::now();
        black_box(f());
        best = best.min(start.elapsed().as_secs_f64());
    }
    best * 1e3
}

fn main() -> Result<()> {
    const N: usize = 4096;
    let a = Tensor::from_vec((0..N * N).map(|i| (i % 7) as f32).collect(), &[N, N])?;
    let b = Tensor::from_vec((0..N * N).map(|i| (i % 5) as f32).collect(), &[N, N])?;
    let row = Tensor::from_vec((0..N).map(|i| i as f32).collect(), &[N])?;
    let a_view = a.transpose(0, 1)?;
    let b_view = b.transpose(0, 1)?;

    println!("== 덧셈, {N} × {N} ==");
    let slices = best_of(5, || {
        let (x, y) = (a.as_slice::<f32>().unwrap(), b.as_slice::<f32>().unwrap());
        x.iter().zip(y).map(|(x, y)| x + y).collect::<Vec<f32>>()
    });
    let fast = best_of(5, || ops::add(&a, &b).unwrap());
    let broadcast = best_of(5, || ops::add(&a, &row).unwrap());
    let views = best_of(5, || ops::add(&a_view, &b_view).unwrap());
    println!("{:<40} {:>9.1} ms", "슬라이스를 직접 더하기 (기준)", slices);
    println!("{:<40} {:>9.1} ms", "add: 형상이 같고 연속 (빠른 길)", fast);
    println!(
        "{:<40} {:>9.1} ms",
        "add: 행 하나를 브로드캐스팅 (일반)", broadcast
    );
    println!("{:<40} {:>9.1} ms", "add: 전치 뷰 두 개 (일반)", views);

    println!();
    const M: usize = 512;
    let p = Tensor::from_vec((0..M * M).map(|i| (i % 7) as f32).collect(), &[M, M])?;
    let q = Tensor::from_vec((0..M * M).map(|i| (i % 5) as f32).collect(), &[M, M])?;
    let q_view = q.transpose(0, 1)?;
    println!("== 행렬 곱, {M} × {M} ==");
    let kernel = best_of(5, || {
        linalg::matmul(p.as_slice().unwrap(), q.as_slice().unwrap(), M, M, M)
    });
    let contiguous = best_of(5, || ops::matmul(&p, &q).unwrap());
    let with_view = best_of(5, || ops::matmul(&p, &q_view).unwrap());
    println!("{:<40} {:>9.1} ms", "커널을 직접 부르기 (기준)", kernel);
    println!("{:<40} {:>9.1} ms", "matmul: 둘 다 연속", contiguous);
    println!(
        "{:<40} {:>9.1} ms",
        "matmul: 한쪽이 전치 뷰 (복사 포함)", with_view
    );
    Ok(())
}
