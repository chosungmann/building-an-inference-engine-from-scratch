//! 17장: 이 기계에서 어떤 커널 구현이 선택되는지, 그리고 그 구현이 기준 구현과
//! 같은 비트를 내는지 보인다.
//!
//! 실행: `cargo run --release -p badak_bench --example ch17_backend`
//!
//! CI가 이 예제를 실행한다. 로그에서 x86 기계가 어떤 구현을 썼는지 볼 수 있다.

use badak_core::simd::{self, Backend, Walk};

fn main() {
    let backend = Backend::detect();
    println!("프로세서의 종류: {}", std::env::consts::ARCH);
    println!("선택된 구현: {backend:?}");

    // 크기가 어중간한 선형 층 하나를 두 구현으로 계산해 비트를 견준다.
    let (count, rows, cols) = (9, 13, 203);
    let values = |count: usize, seed: usize| -> Vec<f32> {
        (0..count)
            .map(|i| ((i * 37 + seed * 101) % 29) as f32 * 0.37 - 5.0)
            .collect()
    };
    let (x, w) = (values(count * cols, 3), values(rows * cols, 4));
    let run = |backend: Backend, walk: Walk| {
        let mut y = vec![0.0_f32; count * rows];
        simd::linear_with(backend, walk, &x, count, &w, rows, cols, &mut y);
        y.iter().map(|value| value.to_bits()).collect::<Vec<u32>>()
    };
    let reference = run(Backend::Portable, Walk::TwoRows);
    let same = [Walk::FourStreams, Walk::TwoRows, Walk::FourRows]
        .into_iter()
        .all(|walk| run(backend, walk) == reference);
    // 다른 기계의 결과와 견줄 수 있도록 결과 전체를 숫자 하나로 줄여 보인다.
    let digest = reference
        .iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, bits| {
            (hash ^ u64::from(*bits)).wrapping_mul(0x0100_0000_01b3)
        });
    println!("기준 구현과 같은 비트: {same}");
    println!("결과의 지문: {digest:016x}");
    assert!(same, "선택된 구현이 기준 구현과 다른 값을 냈습니다");
}
