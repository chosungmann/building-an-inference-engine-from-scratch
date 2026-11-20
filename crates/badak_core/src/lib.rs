//! `badak`의 핵심 수치 연산.
//!
//! 이 크레이트는 Rust 표준 라이브러리만 사용한다. 책의 진행에 따라 텐서,
//! 연산자, 커널, 스레드 풀이 차례로 추가된다.

pub mod approx;
pub mod arena;
pub mod dims;
pub mod error;
pub mod kernel;
pub mod linalg;
pub mod nn;
pub mod ops;
pub mod packed;
pub mod prob;
pub mod profile;
pub mod quant;
pub mod random;
pub mod simd;
pub mod sparse;
pub mod tensor;
pub mod threads;

pub use dims::Dims;
pub use error::{Error, Result};
pub use tensor::{DType, Tensor};
