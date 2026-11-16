//! 7장: 손글씨 숫자를 분류한다.
//!
//! 실행: `cargo run --release -p badak_models --example ch07_classify`
//!
//! 먼저 `sh scripts/download_mnist.sh`와 `sh scripts/download_ch07.sh`로
//! 데이터와 모델을 받아 두어야 한다.

use badak_formats::idx;
use badak_models::mnist::{MnistMlp, art_from_image, image_from_art};
use badak_models::models_dir;

// ANCHOR: art
/// 직접 그린 숫자. `#`은 밝은 픽셀, `+`는 중간, `.`은 검은 픽셀이다.
/// 28줄, 줄마다 28칸이다. 고쳐서 다른 숫자를 그려 보자.
const SEVEN: &str = "\
............................
............................
............................
............................
............................
............................
......+##############+......
......################......
......+##############+......
.................+###+......
.................####.......
................+###+.......
................####........
...............+###+........
...............####.........
..............+###+.........
..............####..........
.............+###+..........
.............####...........
............+###+...........
............####............
............###+............
............................
............................
............................
............................
............................
............................";
// ANCHOR_END: art

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let models = models_dir();
    let model = MnistMlp::load(models.join("mnist_mlp/model.safetensors"))?;
    let images = idx::open(models.join("mnist/t10k-images-idx3-ubyte"))?;
    let labels = idx::open(models.join("mnist/t10k-labels-idx1-ubyte"))?;
    println!("모델: 파라미터 {}개", model.parameter_count());
    println!("데이터: {:?}, 정답 {:?}", images.shape(), labels.shape());

    // ANCHOR: accuracy
    let predictions = model.classify(&images)?;
    let labels = labels.to_vec::<u8>()?;
    let correct = predictions
        .iter()
        .zip(&labels)
        .filter(|&(&predicted, &label)| predicted == i32::from(label))
        .count();
    println!(
        "정확도: {}장 중 {}장 ({:.2}%)",
        labels.len(),
        correct,
        100.0 * correct as f64 / labels.len() as f64
    );
    // ANCHOR_END: accuracy

    // 전처리에서 정규화를 빼먹으면 어떻게 될까. 평균 0, 표준편차 1은
    // 아무것도 하지 않는 것과 같다.
    let careless =
        MnistMlp::load(models.join("mnist_mlp/model.safetensors"))?.with_normalization(0.0, 1.0);
    let wrong = careless
        .classify(&images)?
        .iter()
        .zip(&labels)
        .filter(|&(&predicted, &label)| predicted == i32::from(label))
        .count();
    println!(
        "정규화를 빼먹으면: {}장 중 {}장 ({:.2}%)",
        labels.len(),
        wrong,
        100.0 * wrong as f64 / labels.len() as f64
    );

    println!();
    println!("앞의 열 장");
    println!(
        "  정답  {}",
        labels[..10]
            .iter()
            .map(|l| format!("{l} "))
            .collect::<String>()
    );
    println!(
        "  예측  {}",
        predictions[..10]
            .iter()
            .map(|p| format!("{p} "))
            .collect::<String>()
    );

    println!();
    println!("직접 그린 숫자");
    let drawn = image_from_art(SEVEN);
    println!("{}", art_from_image(&drawn.squeeze(0)?)?);
    println!("예측: {}", model.classify(&drawn)?[0]);

    // 같은 그림을 왼쪽 위로 6칸 옮기면 어떻게 될까.
    let shifted: String = SEVEN
        .lines()
        .skip(6)
        .map(|line| format!("{}\n", &line[6..]))
        .collect();
    let moved = image_from_art(&shifted);
    println!();
    println!("왼쪽 위로 6칸 옮긴 같은 숫자");
    println!("{}", art_from_image(&moved.squeeze(0)?)?);
    println!("예측: {}", model.classify(&moved)?[0]);
    Ok(())
}
