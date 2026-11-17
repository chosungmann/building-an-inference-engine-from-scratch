//! `badak`의 측정 도구 (14장).
//!
//! 무엇인가를 여러 번 실행해 시간을 재고, 그 숫자들을 요약하고, 어떤
//! 환경에서 쟀는지 적는다. Rust 표준 라이브러리만 쓴다.

pub mod counters;
pub mod environment;
pub mod stats;
pub mod timing;

pub use counters::Counters;
pub use stats::Summary;
pub use timing::{Plan, measure};
