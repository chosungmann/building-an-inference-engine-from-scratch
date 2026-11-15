//! 2장: 소프트맥스와 그 수치 안정성.
//!
//! 실행: `cargo run --release -p badak_core --example ch02_softmax`

use badak_core::prob::{argmax, log_softmax, softmax, softmax_naive};

fn show(label: &str, values: &[f32]) {
    let formatted: Vec<String> = values.iter().map(|v| format!("{v:.4}")).collect();
    println!("{label:<24}: [{}]", formatted.join(", "));
}

fn main() {
    let scores = [2.0, 1.0, 0.1, -1.0];
    println!("== 점수를 확률로 ==");
    show("점수", &scores);
    let probabilities = softmax(&scores);
    show("소프트맥스", &probabilities);
    println!(
        "{:<24}: {:.6}",
        "확률의 합",
        probabilities.iter().sum::<f32>()
    );
    println!("{:<24}: {:?}", "가장 큰 값의 위치", argmax(&probabilities));

    println!();
    println!("== 점수 차이가 벌어지면 ==");
    show("점수 × 1", &softmax(&scores));
    let doubled: Vec<f32> = scores.iter().map(|s| s * 2.0).collect();
    show("점수 × 2", &softmax(&doubled));
    let halved: Vec<f32> = scores.iter().map(|s| s * 0.5).collect();
    show("점수 × 0.5", &softmax(&halved));

    println!();
    println!("== 큰 점수에서의 넘침 ==");
    let large = [1000.0, 1001.0, 1002.0];
    show("점수", &large);
    println!("{:<24}: {:?}", "단순한 구현", softmax_naive(&large));
    show("안정한 구현", &softmax(&large));
    println!("{:<24}: {}", "exp(1000) (f32)", 1000.0_f32.exp());

    println!();
    println!("== 아주 작은 확률의 로그 ==");
    let extreme = [0.0, 200.0];
    println!("{:<24}: {}", "ln(softmax)[0]", softmax(&extreme)[0].ln());
    println!("{:<24}: {}", "log_softmax[0]", log_softmax(&extreme)[0]);
}
