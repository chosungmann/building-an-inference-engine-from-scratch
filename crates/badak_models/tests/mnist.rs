//! 내려받은 모델과 데이터로 MNIST 분류기 전체를 검증한다.
//!
//! 파일이 없으면 건너뛴다. `sh scripts/download_mnist.sh`와
//! `sh scripts/download_ch07.sh`를 실행하면 준비된다.

use badak_core::approx::compare;
use badak_core::ops;
use badak_formats::idx;
use badak_formats::safetensors::SafeTensors;
use badak_models::mnist::MnistMlp;
use badak_models::models_dir;

#[test]
fn matches_pytorch_on_the_whole_test_set() {
    let models = models_dir();
    let model_path = models.join("mnist_mlp/model.safetensors");
    let images_path = models.join("mnist/t10k-images-idx3-ubyte");
    if !model_path.exists() || !images_path.exists() {
        eprintln!(
            "건너뜀: {} 아래에 모델과 데이터가 없습니다",
            models.display()
        );
        return;
    }

    let model = MnistMlp::load(model_path).unwrap();
    assert_eq!(model.parameter_count(), 101_770);
    let reference = SafeTensors::open(models.join("mnist_mlp/reference.safetensors")).unwrap();
    let images = idx::open(images_path).unwrap();
    let labels = idx::open(models.join("mnist/t10k-labels-idx1-ubyte")).unwrap();

    // 점수가 PyTorch와 허용 오차 안에서 같다.
    let logits = model.forward(&model.preprocess(&images).unwrap()).unwrap();
    let expected = reference
        .tensor("all_logits")
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    let scale = expected
        .iter()
        .fold(0.0_f32, |max, value| max.max(value.abs()));
    let report = compare(
        &logits.to_vec::<f32>().unwrap(),
        &expected,
        1e-5,
        1e-5 * scale,
    );
    assert!(report.is_close(), "{report}");

    // 예측이 한 장도 다르지 않다.
    let predictions = ops::argmax(&logits, 1).unwrap().to_vec::<i32>().unwrap();
    assert_eq!(
        predictions,
        reference
            .tensor("predictions")
            .unwrap()
            .to_vec::<i32>()
            .unwrap()
    );

    // 정확도가 PyTorch가 기록한 값과 같다.
    let correct = predictions
        .iter()
        .zip(labels.to_vec::<u8>().unwrap())
        .filter(|&(&predicted, label)| predicted == i32::from(label))
        .count();
    assert_eq!(correct.to_string(), reference.metadata()["correct"]);
}
