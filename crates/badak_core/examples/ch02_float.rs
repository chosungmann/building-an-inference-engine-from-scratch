//! 2장: 부동소수점의 성질을 직접 확인한다.
//!
//! 실행: `cargo run --release -p badak_core --example ch02_float`

fn main() {
    println!("== f32가 표현하는 값 ==");
    println!("0.1을 f32로 저장하면     : {:.20}", 0.1_f32);
    println!("f32::EPSILON             : {:e}", f32::EPSILON);
    println!("16777216 + 1 (f32)       : {}", 16_777_216.0_f32 + 1.0);

    println!();
    println!("== 0.1을 천만 번 더하기 (정답 1000000) ==");
    let mut sum32 = 0.0_f32;
    let mut sum64 = 0.0_f64;
    for _ in 0..10_000_000 {
        sum32 += 0.1;
        sum64 += 0.1;
    }
    println!("f32 누적 합              : {sum32}");
    println!("f64 누적 합              : {sum64}");

    println!();
    println!("== 더하는 순서에 따른 차이: 1/1 + 1/2 + ... + 1/n, n = 1000만 ==");
    let n = 10_000_000_u32;
    let forward: f32 = (1..=n).map(|i| 1.0 / i as f32).sum();
    let backward: f32 = (1..=n).rev().map(|i| 1.0 / i as f32).sum();
    let reference: f64 = (1..=n).rev().map(|i| 1.0 / f64::from(i)).sum();
    println!("큰 수부터 (f32)          : {forward}");
    println!("작은 수부터 (f32)        : {backward}");
    println!("기준값 (f64)             : {reference:.6}");

    println!();
    println!("== 결합법칙이 성립하지 않는다 ==");
    let (a, b, c) = (1.0e8_f32, -1.0e8_f32, 1.0_f32);
    println!("(a + b) + c              : {}", (a + b) + c);
    println!("a + (b + c)              : {}", a + (b + c));
}
