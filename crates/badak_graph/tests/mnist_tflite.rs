//! 내려받은 `.tflite` 모델로 그래프 해석기 전체를 검증한다.
//!
//! 파일이 없으면 건너뛴다. `sh scripts/download_mnist.sh`와
//! `sh scripts/download_ch26.sh`를 실행하면 준비된다.

use std::sync::Arc;

use badak_core::random::Pcg64;
use badak_formats::idx;
use badak_formats::safetensors::SafeTensors;
use badak_formats::tflite::Model;
use badak_graph::{Graph, Interpreter, Registry, Strategy, memory, models_dir};

fn available() -> bool {
    let models = models_dir();
    let ready = models.join("mnist_tflite/residual.tflite").exists()
        && models.join("mnist/t10k-images-idx3-ubyte").exists();
    if !ready {
        eprintln!(
            "건너뜀: {} 아래에 모델과 데이터가 없습니다",
            models.display()
        );
    }
    ready
}

#[test]
fn matches_litert_on_the_whole_test_set() {
    if !available() {
        return;
    }
    let models = models_dir();
    let images = idx::open(models.join("mnist/t10k-images-idx3-ubyte")).unwrap();
    let count = images.shape()[0];
    let pixels: Vec<f32> = images
        .to_vec::<u8>()
        .unwrap()
        .iter()
        .map(|&pixel| f32::from(pixel) / 255.0)
        .collect();
    let reference = SafeTensors::open(models.join("mnist_tflite/reference.safetensors")).unwrap();
    let registry = Registry::builtin();

    for (name, steps) in [("mlp", 3), ("residual", 12)] {
        let model = Model::open(models.join(format!("mnist_tflite/{name}.tflite"))).unwrap();
        let graph = Graph::from_tflite(&model).unwrap();
        // 가중치는 복사하지 않고 파일을 그대로 가리킨다.
        let constants = graph
            .values
            .iter()
            .filter_map(|value| value.constant.as_ref());
        assert!(
            constants.clone().count() >= 4 && constants.clone().all(|tensor| tensor.is_borrowed())
        );

        let shape: &[usize] = &[count, 784];
        let mut batch = Interpreter::with(&graph, &registry, &[shape], Strategy::BySize).unwrap();
        assert_eq!(batch.step_count(), steps);
        batch.input_mut(0).copy_from_slice(&pixels);
        batch.run().unwrap();
        let probabilities = batch.output(0).to_vec();

        // 앞의 16장은 확률이 LiteRT와 같다.
        let expected = reference
            .tensor(&format!("{name}.probabilities"))
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        for (ours, theirs) in probabilities.iter().zip(&expected) {
            assert!((ours - theirs).abs() < 2e-6, "{name}: {ours} != {theirs}");
        }
        // 만 장 모두 같은 숫자를 고른다.
        let theirs = reference
            .tensor(&format!("{name}.predictions"))
            .unwrap()
            .to_vec::<i32>()
            .unwrap();
        for (row, &theirs) in probabilities.as_chunks::<10>().0.iter().zip(&theirs) {
            let ours = (0..10).max_by(|&a, &b| row[a].total_cmp(&row[b])).unwrap();
            assert_eq!(ours as i32, theirs);
        }

        // 자리를 어떻게 나누든, 한 장씩 넣든 결과의 비트가 같다.
        for strategy in [Strategy::Separate, Strategy::InOrder, Strategy::BySize] {
            let mut single = Interpreter::with(&graph, &registry, &[], strategy).unwrap();
            assert!(single.arena_len() >= memory::lower_bound(&single.lifetimes()));
            for (image, expected) in pixels
                .as_chunks::<784>()
                .0
                .iter()
                .zip(probabilities.as_chunks::<10>().0)
                .take(64)
            {
                single.scribble(f32::NAN);
                single.input_mut(0).copy_from_slice(image);
                single.run().unwrap();
                assert_eq!(single.output(0), expected);
            }
        }
    }
}

#[test]
fn damaged_files_never_panic() {
    if !available() {
        return;
    }
    let bytes = std::fs::read(models_dir().join("mnist_tflite/mlp.tflite")).unwrap();
    let registry = Registry::builtin();
    let attempt = |bytes: Vec<u8>| {
        let Ok(model) = Model::parse(Arc::new(bytes)) else {
            return;
        };
        let Ok(graph) = Graph::from_tflite(&model) else {
            return;
        };
        if let Ok(mut interpreter) = Interpreter::new(&graph, &registry) {
            let _ = interpreter.run();
        }
    };
    let mut rng = Pcg64::new(26);
    // 잘린 파일.
    for _ in 0..200 {
        let len = (rng.next_u64() % bytes.len() as u64) as usize;
        attempt(bytes[..len].to_vec());
    }
    // 한 바이트가 바뀐 파일. 가중치가 아닌 부분은 파일의 앞과 뒤에 있다.
    let edges: Vec<usize> = (0..2048).chain(bytes.len() - 2048..bytes.len()).collect();
    for &position in &edges {
        let mut broken = bytes.clone();
        broken[position] ^= 1 << (rng.next_u64() % 8);
        attempt(broken);
        let mut broken = bytes.clone();
        broken[position] = 0xff;
        attempt(broken);
    }
}
