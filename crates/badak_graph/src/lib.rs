//! 그래프 해석기 (26장).
//!
//! `badak_models`는 모델의 계산 순서를 Rust 코드로 적는다. 이 크레이트는
//! 계산 순서를 **데이터**로 받는다. 파일에서 읽은 그래프를 훑어 무엇을 어떤
//! 순서로 실행할지, 중간 결과를 어디에 둘지를 실행 전에 정하고(계획), 그
//! 계획대로 실행한다. 1장에서 본 엔진의 둘째 계열이다.

pub mod conv;
pub mod error;
pub mod graph;
pub mod interpreter;
pub mod kernels;
pub mod memory;
pub mod registry;
pub mod spatial;
pub mod winograd;

pub use error::{Error, Result};
pub use graph::{Activation, Graph, Node, Op, Padding, Value};
pub use interpreter::Interpreter;
pub use kernels::{Kernel, View, ViewMut};
pub use memory::{Layout, Lifetime, Strategy};
pub use registry::Registry;

use std::path::PathBuf;

/// 저장소 최상위의 `models` 디렉터리 아래 경로를 만든다. 환경 변수
/// `BADAK_MODELS`가 있으면 그 디렉터리를 대신 쓴다.
pub fn models_dir() -> PathBuf {
    std::env::var_os("BADAK_MODELS").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models"),
        PathBuf::from,
    )
}
