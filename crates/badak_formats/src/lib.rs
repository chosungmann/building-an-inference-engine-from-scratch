//! `badak`이 읽는 모델 파일 형식 (6장).
//!
//! 모델 파일은 믿을 수 없는 입력이다. 인터넷에서 내려받은 파일일 수 있고,
//! 깨졌거나 악의적으로 만들어졌을 수 있다. 그래서 이 크레이트의 파서는
//! 파일에 적힌 어떤 길이나 위치도 확인 없이 쓰지 않는다.

pub mod error;
pub mod flatbuffers;
pub mod gguf;
pub mod idx;
pub mod json;
pub mod mmap;
pub mod onnx;
pub mod protobuf;
mod reader;
pub mod safetensors;
pub mod sha256;
pub mod store;
pub mod tflite;

pub use error::{Error, Result};
pub use mmap::Mmap;
