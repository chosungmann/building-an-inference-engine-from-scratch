//! 3장: 경사 하강법으로 `y = w x + b`의 가중치를 찾는다.
//!
//! 실행: `cargo run --release -p badak_core --example ch03_training`
//!
//! 이 책은 학습을 다루지 않는다. 이 예제는 "가중치가 어디서 오는가"를
//! 한 번 눈으로 보기 위한 것이다.

// ANCHOR: training
fn main() {
    // 정답 함수 y = 2x + 1에서 뽑은 데이터. 모델은 이 식을 모른다.
    let data: Vec<(f32, f32)> = (-5..=5).map(|i| (i as f32, 2.0 * i as f32 + 1.0)).collect();
    let count = data.len() as f32;

    // 아무 값에서나 시작한다.
    let (mut w, mut b) = (0.0_f32, 0.0_f32);
    let learning_rate = 0.02;

    println!("{:>6} {:>10} {:>10} {:>12}", "단계", "w", "b", "손실");
    for step in 0..=1000 {
        // 순전파: 지금의 w, b로 예측하고, 정답과의 차이를 제곱해 평균한다.
        let loss: f32 = data
            .iter()
            .map(|&(x, y)| (w * x + b - y).powi(2))
            .sum::<f32>()
            / count;

        if [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000].contains(&step) {
            println!("{step:>6} {w:>10.5} {b:>10.5} {loss:>12.6}");
        }

        // 기울기: w와 b를 조금 바꾸면 손실이 얼마나 변하는가.
        let grad_w: f32 = data
            .iter()
            .map(|&(x, y)| 2.0 * (w * x + b - y) * x)
            .sum::<f32>()
            / count;
        let grad_b: f32 = data
            .iter()
            .map(|&(x, y)| 2.0 * (w * x + b - y))
            .sum::<f32>()
            / count;

        // 손실이 줄어드는 방향으로 조금 움직인다.
        w -= learning_rate * grad_w;
        b -= learning_rate * grad_b;
    }
}
// ANCHOR_END: training
