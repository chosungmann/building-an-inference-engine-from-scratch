//! 31장: SME 커널과 NEON 커널이 같은 선형 층을 계산하는 속도를 견준다.
//!
//! 실행: `cargo run --release -p badak_core --example ch31_sme`
//!
//! 스레드 하나로 잰다. 형상은 ResNet-50과 MobileNetV2의 합성곱에서 골랐다
//! (입력 벡터의 수 × 출력 채널 × 창 하나의 값의 수).

#[cfg(target_arch = "aarch64")]
use std::time::Instant;

#[cfg(target_arch = "aarch64")]
use badak_core::simd::{self, Backend, Walk};

#[cfg(target_arch = "aarch64")]
fn values(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|i| ((i * 37 + seed * 101) % 29) as f32 * 0.07 - 1.0)
        .collect()
}

/// `work`를 되풀이해 한 번에 걸린 시간의 가장 작은 값을 초로 돌려준다.
#[cfg(target_arch = "aarch64")]
fn best(mut work: impl FnMut()) -> f64 {
    let mut best = f64::INFINITY;
    let started = Instant::now();
    while started.elapsed().as_secs_f64() < 0.4 {
        let start = Instant::now();
        work();
        best = best.min(start.elapsed().as_secs_f64());
    }
    best
}

fn main() {
    #[cfg(target_arch = "aarch64")]
    {
        use badak_core::simd::sme;
        if !sme::available() {
            println!("이 기계에는 SME가 없습니다.");
            return;
        }
        println!("스트리밍 모드의 벡터 하나: f32 {}개", sme::lanes());
        println!(
            "{:>6} {:>6} {:>6} {:>12} {:>12} {:>6} {:>12}",
            "벡터", "행", "열", "NEON 억/초", "SME 억/초", "배", "가장 큰 차이"
        );
        let shapes = [
            (12544, 64, 147),
            (3136, 64, 576),
            (3136, 256, 64),
            (784, 128, 1152),
            (784, 512, 128),
            (196, 256, 2304),
            (196, 1024, 256),
            (49, 512, 4608),
            (49, 2048, 512),
            (12544, 96, 16),
            (3136, 24, 96),
            (196, 96, 384),
            (16, 1000, 2048),
            (1, 1000, 2048),
        ];
        for (count, rows, cols) in shapes {
            let (x, w) = (values(count * cols, 1), values(rows * cols, 2));
            let (mut plain, mut outer) = (vec![0.0; count * rows], vec![0.0; count * rows]);
            let walk = Walk::for_vectors(count);
            let neon = best(|| {
                simd::linear_with(Backend::Neon, walk, &x, count, &w, rows, cols, &mut plain);
            });
            let packed = sme::Packed::new(&w, rows, cols).expect("SME가 있다");
            let fast = best(|| sme::linear(&x, count, &packed, &mut outer));
            let gap = plain
                .iter()
                .zip(&outer)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            let rate = |seconds: f64| (count * rows * cols) as f64 / seconds / 1e8;
            println!(
                "{count:>6} {rows:>6} {cols:>6} {:>12.0} {:>12.0} {:>6.1} {gap:>12.1e}",
                rate(neon),
                rate(fast),
                neon / fast
            );
        }

        // 스레드를 늘리면 어떻게 되는가. 스레드마다 같은 일을 따로 하고,
        // 모두 합쳐 1초에 곱셈을 몇 번 했는지를 본다.
        println!("\n스레드마다 같은 일을 할 때, 모두 합친 속도 (784 × 512 × 128, 억/초)");
        println!(
            "{:>6} {:>10} {:>6} {:>10} {:>6}",
            "스레드", "NEON", "배", "SME", "배"
        );
        let (count, rows, cols) = (784, 512, 128);
        let mut single = [0.0_f64; 2];
        for threads in [1, 2, 4, 6, 8, 12] {
            let mut rates = [0.0_f64; 2];
            for (kind, rate) in rates.iter_mut().enumerate() {
                let rounds = if kind == 0 { 20 } else { 200 };
                let started = Instant::now();
                std::thread::scope(|scope| {
                    for _ in 0..threads {
                        scope.spawn(move || {
                            let (x, w) = (values(count * cols, 1), values(rows * cols, 2));
                            let mut y = vec![0.0; count * rows];
                            let packed = sme::Packed::new(&w, rows, cols).expect("SME가 있다");
                            let walk = Walk::for_vectors(count);
                            for _ in 0..rounds {
                                if kind == 0 {
                                    let backend = Backend::Neon;
                                    simd::linear_with(
                                        backend, walk, &x, count, &w, rows, cols, &mut y,
                                    );
                                } else {
                                    sme::linear(&x, count, &packed, &mut y);
                                }
                            }
                        });
                    }
                });
                let work = (threads * rounds * count * rows * cols) as f64;
                *rate = work / started.elapsed().as_secs_f64() / 1e8;
            }
            if threads == 1 {
                single = rates;
            }
            println!(
                "{threads:>6} {:>10.0} {:>6.1} {:>10.0} {:>6.1}",
                rates[0],
                rates[0] / single[0],
                rates[1],
                rates[1] / single[1]
            );
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    println!("SME는 64비트 ARM의 명령입니다.");
}
