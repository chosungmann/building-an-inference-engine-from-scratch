//! 공식 라이브러리가 쓴 파일을 우리 파서가 읽는지 확인한다.
//!
//! `tests/data`의 파일은 `reference/ch06_fixtures.py`가 Python의
//! `safetensors`와 `gguf` 패키지로 만든 것이다.

use std::path::PathBuf;

use badak_core::DType;
use badak_formats::Error;
use badak_formats::gguf::{GgmlType, Gguf, MetadataValue};
use badak_formats::safetensors::SafeTensors;

fn data(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

// ANCHOR: safetensors
#[test]
fn reads_a_file_written_by_the_safetensors_library() {
    let file = SafeTensors::open(data("tiny.safetensors")).unwrap();
    assert_eq!(
        file.names().collect::<Vec<_>>(),
        ["bias", "half", "ids", "pixels", "weight"]
    );
    assert_eq!(file.metadata()["format"], "pt");
    assert_eq!(file.metadata()["note"], "밑바닥 badak");

    let weight = file.tensor("weight").unwrap();
    assert_eq!((weight.dtype(), weight.shape()), (DType::F32, &[2, 3][..]));
    assert_eq!(
        weight.to_vec::<f32>().unwrap(),
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    );
    assert!(weight.is_borrowed());

    assert_eq!(
        file.tensor("bias").unwrap().to_vec::<f32>().unwrap(),
        [0.5, -1.5, 2.25]
    );
    assert_eq!(
        file.tensor("ids").unwrap().to_vec::<i32>().unwrap(),
        [3, 1, 4, 1]
    );
    let pixels = file.tensor("pixels").unwrap();
    assert_eq!(pixels.shape(), [2, 2]);
    assert_eq!(pixels.to_vec::<u8>().unwrap(), [0, 64, 128, 255]);

    // 16비트 실수는 정보는 읽히지만 아직 텐서로 꺼낼 수 없다.
    assert_eq!(file.info("half").unwrap().dtype, "F16");
    assert!(matches!(file.tensor("half"), Err(Error::Unsupported(_))));
}
// ANCHOR_END: safetensors

// ANCHOR: gguf
#[test]
fn reads_a_file_written_by_the_gguf_library() {
    let file = Gguf::open(data("tiny.gguf")).unwrap();
    assert_eq!(file.version(), 3);

    let metadata = file.metadata();
    assert_eq!(
        metadata["general.architecture"].as_str(),
        Some("badak-test")
    );
    assert_eq!(metadata["general.name"].as_str(), Some("tiny"));
    assert_eq!(metadata["test.uint32"], MetadataValue::U32(7));
    assert_eq!(metadata["test.int64"], MetadataValue::I64(-9_000_000_000));
    assert_eq!(metadata["test.float32"].as_f64(), Some(0.5));
    assert_eq!(metadata["test.bool"], MetadataValue::Bool(true));
    assert_eq!(metadata["test.string"].as_str(), Some("밑바닥 badak"));
    let numbers: Vec<u64> = metadata["test.int_array"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_u64().unwrap())
        .collect();
    assert_eq!(numbers, [1, 2, 3]);
    let strings: Vec<&str> = metadata["test.string_array"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s.as_str().unwrap())
        .collect();
    assert_eq!(strings, ["<s>", "</s>", "안녕"]);

    assert_eq!(file.names().collect::<Vec<_>>(), ["half", "ids", "weight"]);
    // NumPy에서 (2, 3)이던 텐서가 같은 형상, 같은 값으로 읽혀야 한다.
    let weight = file.tensor("weight").unwrap();
    assert_eq!(weight.shape(), [2, 3]);
    assert_eq!(
        weight.to_vec::<f32>().unwrap(),
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    );
    assert_eq!(weight.get::<f32>(&[1, 0]).unwrap(), 3.0);
    assert!(weight.is_borrowed());
    assert_eq!(
        file.tensor("ids").unwrap().to_vec::<i32>().unwrap(),
        [3, 1, 4, 1]
    );

    assert_eq!(file.info("half").unwrap().ggml_type, GgmlType::F16);
    assert!(matches!(file.tensor("half"), Err(Error::Unsupported(_))));
}
// ANCHOR_END: gguf
