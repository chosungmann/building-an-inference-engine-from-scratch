//! 28장: 채널을 어디에 두느냐에 따라 같은 `1 × 1` 합성곱의 반복문이 어떻게
//! 달라지는지 본다.
//!
//! 실행: `cargo run --release -p badak_graph --example ch28_layout`
//!
//! 세 가지 레이아웃으로 같은 계산을 한다. 커널은 모두 컴파일러가 벡터 명령으로
//! 바꿔 주는 평범한 반복문이다(17장의 커널을 쓰지 않는다). 레이아웃이 반복문의
//! 모양을 어떻게 정하는지만 보려는 것이다. 스레드는 하나만 쓴다.

use std::hint::black_box;

use badak_bench::stats::Summary;
use badak_bench::timing::{Plan, measure};
use badak_core::random::Pcg64;

/// 채널 묶음 하나의 채널 수.
const BLOCK: usize = 8;

// ANCHOR: nhwc
/// NHWC: 픽셀마다, 출력 채널마다, 입력 채널을 따라 내적한다.
fn nhwc(x: &[f32], w: &[f32], y: &mut [f32], cin: usize, cout: usize) {
    for (pixel, out) in x.chunks_exact(cin).zip(y.chunks_exact_mut(cout)) {
        for (value, filter) in out.iter_mut().zip(w.chunks_exact(cin)) {
            *value = pixel.iter().zip(filter).map(|(a, b)| a * b).sum();
        }
    }
}
// ANCHOR_END: nhwc

// ANCHOR: nchw
/// NCHW: 출력 채널마다, 입력 채널마다, 한 장(평면) 전체에 가중치 하나를 곱해
/// 더한다. 가장 안쪽 반복문이 픽셀을 따라 돈다.
fn nchw(x: &[f32], w: &[f32], y: &mut [f32], cin: usize) {
    let plane = x.len() / cin;
    y.fill(0.0);
    for (o, out) in y.chunks_exact_mut(plane).enumerate() {
        for (i, source) in x.chunks_exact(plane).enumerate() {
            let weight = w[o * cin + i];
            for (value, &pixel) in out.iter_mut().zip(source) {
                *value += weight * pixel;
            }
        }
    }
}
// ANCHOR_END: nchw

// ANCHOR: blocked
/// 채널 묶음(NCHW8c): 채널을 여덟 개씩 묶어, 묶음 안에서는 픽셀마다 여덟
/// 채널이 나란히 놓인다. 픽셀마다, 입력 채널마다, 출력 채널 여덟 개에
/// 한꺼번에 더한다. 가중치도 출력 채널 여덟 개가 나란히 놓이게 바꿔 둔다.
fn blocked(x: &[f32], w: &[f32], y: &mut [f32], cin: usize) {
    let plane = x.len() / cin;
    y.fill(0.0);
    for (ob, out) in y.chunks_exact_mut(plane * BLOCK).enumerate() {
        for (ib, source) in x.chunks_exact(plane * BLOCK).enumerate() {
            // 이 묶음 쌍의 가중치: (입력 채널 8) × (출력 채널 8).
            let weights = &w[(ob * (cin / BLOCK) + ib) * BLOCK * BLOCK..][..BLOCK * BLOCK];
            for (sums, pixel) in out
                .as_chunks_mut::<BLOCK>()
                .0
                .iter_mut()
                .zip(source.as_chunks::<BLOCK>().0)
            {
                for (i, &value) in pixel.iter().enumerate() {
                    let column: &[f32; BLOCK] =
                        weights[i * BLOCK..][..BLOCK].try_into().expect("여덟 개다");
                    for o in 0..BLOCK {
                        sums[o] += value * column[o];
                    }
                }
            }
        }
    }
}
// ANCHOR_END: blocked

fn main() {
    let mut rng = Pcg64::new(28);
    let plan = Plan {
        warmup: 3,
        repetitions: 21,
    };
    println!(
        "{:<22} {:>12} {:>12} {:>16}",
        "모양", "NHWC", "NCHW", "채널 묶음 (8)"
    );
    for (side, cin, cout) in [(56, 64, 64), (28, 128, 128), (14, 256, 256), (7, 512, 512)] {
        let plane = side * side;
        let mut values =
            |len: usize| -> Vec<f32> { (0..len).map(|_| rng.next_f64() as f32 - 0.5).collect() };
        // 같은 값을 세 가지 레이아웃으로 놓는다. x[픽셀][채널]이 기준이다.
        let (x_nhwc, w_oi) = (values(plane * cin), values(cout * cin));
        let mut x_nchw = vec![0.0; plane * cin];
        let mut x_blocked = vec![0.0; plane * cin];
        for pixel in 0..plane {
            for channel in 0..cin {
                let value = x_nhwc[pixel * cin + channel];
                x_nchw[channel * plane + pixel] = value;
                x_blocked[((channel / BLOCK) * plane + pixel) * BLOCK + channel % BLOCK] = value;
            }
        }
        let mut w_blocked = vec![0.0; cout * cin];
        for o in 0..cout {
            for i in 0..cin {
                let index = ((o / BLOCK) * (cin / BLOCK) + i / BLOCK) * BLOCK * BLOCK;
                w_blocked[index + (i % BLOCK) * BLOCK + o % BLOCK] = w_oi[o * cin + i];
            }
        }
        let (mut a, mut b, mut c) = (
            vec![0.0; plane * cout],
            vec![0.0; plane * cout],
            vec![0.0; plane * cout],
        );
        let rate = |seconds: f64| (plane * cin * cout) as f64 / seconds / 1e8;
        let first = Summary::new(&measure(plan, || {
            nhwc(black_box(&x_nhwc), &w_oi, &mut a, cin, cout)
        }))
        .median;
        let second = Summary::new(&measure(plan, || {
            nchw(black_box(&x_nchw), &w_oi, &mut b, cin)
        }))
        .median;
        let third = Summary::new(&measure(plan, || {
            blocked(black_box(&x_blocked), &w_blocked, &mut c, cin)
        }))
        .median;
        // 세 결과가 같은 값인지 확인한다(놓인 순서만 다르다).
        let mut worst = 0.0_f32;
        for pixel in 0..plane {
            for channel in 0..cout {
                let expected = a[pixel * cout + channel];
                let from_nchw = b[channel * plane + pixel];
                let from_blocked = c[((channel / BLOCK) * plane + pixel) * BLOCK + channel % BLOCK];
                worst = worst
                    .max((expected - from_nchw).abs())
                    .max((expected - from_blocked).abs());
            }
        }
        assert!(worst < 1e-3, "세 레이아웃의 결과가 다릅니다: {worst}");
        println!(
            "{:<22} {:>12.0} {:>12.0} {:>16.0}",
            format!("{side}×{side}, {cin} → {cout}"),
            rate(first),
            rate(second),
            rate(third)
        );
    }
    println!("(곱셈 억 번/초. 17장의 커널은 같은 일에 430~510억 번을 낸다)");
}
