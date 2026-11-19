//! 정수 모델이 LiteRT의 자체 커널과 같은 비트를 내는지 확인한다 (29장).
//!
//! 파일이 없으면 건너뛴다. `sh scripts/download_ch29.sh`로 받는다.

use badak_formats::safetensors::SafeTensors;
use badak_formats::tflite::Model;
use badak_graph::quantized::QuantizedInterpreter;
use badak_graph::{Graph, models_dir};

fn signed(bytes: Vec<u8>) -> Vec<i8> {
    bytes.into_iter().map(|byte| byte as i8).collect()
}

#[test]
fn every_layer_matches_litert_bit_for_bit() {
    let directory = models_dir().join("vision_tflite");
    if !directory.join("mobilenetv2_int8.tflite").exists() {
        eprintln!("건너뜀: {}에 정수 모델이 없습니다", directory.display());
        return;
    }
    let reference = SafeTensors::open(directory.join("reference_int8.safetensors")).unwrap();
    let model = Model::open(directory.join("mobilenetv2_int8.tflite")).unwrap();
    let graph = Graph::from_tflite(&model).unwrap();
    assert!(graph.is_quantized());
    let mut interpreter = QuantizedInterpreter::new(&graph).unwrap();
    assert_eq!(interpreter.step_count(), 70);
    let read = |name: &str| {
        signed(
            reference
                .tensor_as_stored(name)
                .unwrap()
                .to_vec::<u8>()
                .unwrap(),
        )
    };

    // 사진 다섯 장의 출력이 모두 같다.
    for image in ["cup", "golden_retriever", "guitar", "pelican", "tabby_cat"] {
        interpreter
            .input_mut(0)
            .copy_from_slice(&read(&format!("{image}.input")));
        interpreter.run().unwrap();
        assert_eq!(
            interpreter.output(0),
            read(&format!("{image}.builtin")),
            "{image}"
        );
    }

    // 첫 사진은 단계마다의 출력도 모두 같다.
    interpreter.input_mut(0).copy_from_slice(&read("cup.input"));
    let outputs = interpreter.step_outputs();
    let mut compared = 0;
    interpreter
        .run_visiting(|index, ours| {
            let expected = read(&format!("cup.tensor.{}", outputs[index]));
            assert_eq!(ours, expected, "단계 {index}");
            compared += 1;
        })
        .unwrap();
    assert_eq!(compared, 70);
}
