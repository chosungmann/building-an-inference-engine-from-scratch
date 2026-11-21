//! 26장: 그래프 해석기의 결과를 LiteRT의 결과와 견준다.
//!
//! 실행: `cargo run --release -p badak_graph --example ch26_verify`
//!
//! 시험 데이터 만 장을 두 모델로 분류한다. 앞의 16장은 확률을, 전체는
//! 고른 숫자를 LiteRT와 견준다. 7장의 엔진과도 견준다.

use badak_formats::idx;
use badak_formats::safetensors::SafeTensors;
use badak_formats::tflite::Model;
use badak_graph::{Graph, Interpreter, Registry, Strategy, models_dir};
use badak_models::mnist::MnistMlp;

/// 줄마다 가장 큰 값의 위치.
fn argmax(rows: &[f32], width: usize) -> Vec<i32> {
    rows.chunks_exact(width)
        .map(|row| {
            let mut best = 0;
            for (index, &value) in row.iter().enumerate() {
                if value > row[best] {
                    best = index;
                }
            }
            best as i32
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let models = models_dir();
    let images = idx::open(models.join("mnist/t10k-images-idx3-ubyte"))?;
    let labels = idx::open(models.join("mnist/t10k-labels-idx1-ubyte"))?.to_vec::<u8>()?;
    let count = images.shape()[0];
    // 모델의 입력은 0과 1 사이의 픽셀 값이다. 정규화는 그래프 안에 있다.
    let pixels: Vec<f32> = images
        .to_vec::<u8>()?
        .iter()
        .map(|&pixel| f32::from(pixel) / 255.0)
        .collect();
    let reference = SafeTensors::open(models.join("mnist_tflite/reference.safetensors"))?;
    let registry = Registry::builtin();

    for name in ["mlp", "residual"] {
        let model = Model::open(models.join(format!("mnist_tflite/{name}.tflite")))?;
        let graph = Graph::from_tflite(&model)?;

        // 파일에 적힌 대로 한 장씩.
        let mut single = Interpreter::new(&graph, &registry)?;
        let mut probabilities = Vec::with_capacity(count * 10);
        for image in pixels.as_chunks::<784>().0 {
            single.input_mut(0).copy_from_slice(image);
            single.run()?;
            probabilities.extend_from_slice(single.output(0));
        }
        // 만 장을 한 번에.
        let mut batch = Interpreter::with(&graph, &registry, &[&[count, 784]], Strategy::BySize)?;
        batch.input_mut(0).copy_from_slice(&pixels);
        batch.run()?;

        let expected = reference
            .tensor(&format!("{name}.probabilities"))?
            .to_vec::<f32>()?;
        let worst = probabilities
            .iter()
            .zip(&expected)
            .map(|(ours, theirs)| (ours - theirs).abs())
            .fold(0.0_f32, f32::max);
        let predictions = argmax(&probabilities, 10);
        let theirs = reference
            .tensor(&format!("{name}.predictions"))?
            .to_vec::<i32>()?;
        let agree = predictions
            .iter()
            .zip(&theirs)
            .filter(|(a, b)| a == b)
            .count();
        let correct = predictions
            .iter()
            .zip(&labels)
            .filter(|&(&prediction, &label)| prediction == i32::from(label))
            .count();
        let same_bits = batch.output(0) == probabilities;

        println!("{name}.tflite");
        println!(
            "  확률의 가장 큰 차이 (앞의 {}장)   {worst:.2e}",
            expected.len() / 10
        );
        println!("  LiteRT와 같은 숫자를 고른 사진     {agree} / {count}");
        println!("  정답을 맞힌 사진                   {correct} / {count}");
        println!(
            "  한 장씩과 만 장을 한 번에          {}",
            if same_bits {
                "비트까지 같음"
            } else {
                "다름"
            }
        );
        if name == "mlp" {
            // 7장의 엔진은 전처리를 코드로 한다. 같은 가중치다.
            let hand_written = MnistMlp::load(models.join("mnist_mlp/model.safetensors"))?;
            let theirs = hand_written.classify(&images)?;
            let agree = predictions
                .iter()
                .zip(&theirs)
                .filter(|(a, b)| a == b)
                .count();
            println!("  7장의 엔진과 같은 숫자를 고른 사진 {agree} / {count}");

            // 변환기가 정규화 (x - 평균) / 표준편차를 첫 층에 접어 넣었는지 본다.
            // W(x - m)/s + b = (W/s)x + (b - m/s ΣW) 이다.
            let original = SafeTensors::open(models.join("mnist_mlp/model.safetensors"))?;
            let (mean, std) = (0.1307_f32, 0.3081_f32);
            let weight = original.tensor("fc1.weight")?.to_vec::<f32>()?;
            let bias = original.tensor("fc1.bias")?.to_vec::<f32>()?;
            let find = |shape: &[usize]| {
                let value = graph.values.iter().find(|value| value.shape == shape);
                value
                    .and_then(|value| value.constant.as_ref())
                    .unwrap()
                    .to_vec::<f32>()
            };
            let (folded_weight, folded_bias) = (find(&[128, 784])?, find(&[128])?);
            let weight_gap = folded_weight
                .iter()
                .zip(&weight)
                .map(|(folded, original)| (folded - original / std).abs())
                .fold(0.0_f32, f32::max);
            let bias_gap = folded_bias
                .iter()
                .zip(bias.iter().zip(weight.as_chunks::<784>().0))
                .map(|(folded, (bias, row))| {
                    (folded - (bias - mean / std * row.iter().sum::<f32>())).abs()
                })
                .fold(0.0_f32, f32::max);
            println!("  파일의 첫 층과 W/s의 가장 큰 차이   {weight_gap:.2e}");
            println!("  파일의 첫 편향과 b - (m/s)ΣW의 차이 {bias_gap:.2e}");
        }
    }
    Ok(())
}
