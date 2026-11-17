//! 17장: 벡터 명령으로 직접 짠 커널의 모양에 따른 속도.
//!
//! 실행: `cargo run --release -p badak_bench --example ch17_shapes`
//!
//! 내적 `M × N`개를 한꺼번에 계산하는 커널을 모양을 바꿔 가며 잰다. 모든
//! 벡터는 가장 빠른 캐시에 들어가는 크기다. 16장의 `ch16_shapes`와 같은
//! 측정을 내장 함수로 짠 커널에 대해 한다.

use std::hint::black_box;

use badak_bench::{Counters, Plan, measure};
use badak_core::simd::{Backend, LANES};

const LEN: usize = 2048;

/// 이 기계의 빠른 구현으로 내적 `M × N`개를 계산한다.
fn fast_dots<const M: usize, const N: usize>(xs: [&[f32]; M], ws: [&[f32]; N]) -> [[f32; N]; M] {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON은 64비트 ARM의 모든 프로세서에 있다.
        unsafe { badak_core::simd::neon::dots(xs, ws) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        assert_eq!(
            Backend::detect(),
            Backend::Avx2,
            "이 예제는 AVX2가 있어야 합니다"
        );
        // SAFETY: 바로 위에서 AVX2와 FMA가 있음을 확인했다.
        unsafe { badak_core::simd::avx2::dots(xs, ws) }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        badak_core::simd::dots_portable(xs, ws)
    }
}

fn vectors(count: usize, seed: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|v| {
            (0..LEN)
                .map(|i| ((i * 31 + (v + seed) * 17) % 19) as f32 * 0.125)
                .collect()
        })
        .collect()
}

fn shape<const M: usize, const N: usize>(register_width: usize) {
    let (xs, ws) = (vectors(M, 0), vectors(N, 100));
    let x: [&[f32]; M] = std::array::from_fn(|r| xs[r].as_slice());
    let w: [&[f32]; N] = std::array::from_fn(|c| ws[c].as_slice());
    let plan = Plan {
        warmup: 2000,
        repetitions: 30000,
    };
    let before = Counters::now();
    // 결과 전체를 가린다. 일부만 쓰면 컴파일러가 나머지 내적을 지운다(14.3절).
    measure(plan, || {
        black_box(fast_dots::<M, N>(black_box(x), black_box(w)))
    });
    let spent = before
        .zip(Counters::now())
        .map(|(before, after)| after.since(before));
    let runs = ((plan.warmup + plan.repetitions) * M * N * LEN) as f64;
    let (instructions, cycles) = spent.map_or((f64::NAN, f64::NAN), |c| {
        (c.instructions as f64 / runs, c.cycles as f64 / runs)
    });
    // 합을 담는 레지스터의 수와, 곱셈 하나에 값을 읽어 오는 횟수.
    let registers = M * N * LANES / register_width;
    let loads = (M + N) as f64 / (M * N * register_width) as f64;
    println!(
        "{:<8} {:>10} {:>12.3} {:>10.2} {:>10.3}",
        format!("{M} × {N}"),
        registers,
        loads,
        instructions,
        cycles
    );
}

fn main() {
    let backend = Backend::detect();
    // 레지스터 하나에 담기는 f32의 수.
    let register_width = if backend == Backend::Avx2 { 8 } else { 4 };
    println!("구현: {backend:?}. 길이 {LEN}인 벡터들의 내적, 곱셈 하나당");
    println!(
        "{:<8} {:>10} {:>12} {:>10} {:>10}",
        "모양", "합 레지스터", "읽기 횟수", "명령", "사이클"
    );
    shape::<1, 1>(register_width);
    shape::<1, 2>(register_width);
    shape::<1, 4>(register_width);
    shape::<2, 2>(register_width);
    shape::<2, 4>(register_width);
    shape::<4, 2>(register_width);
    shape::<2, 6>(register_width);
    shape::<4, 4>(register_width);
}
