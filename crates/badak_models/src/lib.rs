//! `badak`으로 실행하는 모델들.
//!
//! 이 크레이트는 모델의 계산 순서를 Rust 코드로 직접 적는다. 1장에서 본
//! 엔진의 첫째 계열이 하는 방식이다. 모델 파일에서는 가중치만 읽는다.

pub mod bert;
pub mod diffusion;
pub mod llama;
pub mod mnist;
pub mod sampling;
pub mod search;
pub mod smolvlm;
pub mod tokenizer;
pub mod tts;
pub mod vit;
pub mod whisper;

use std::path::PathBuf;

/// 저장소 최상위의 `models` 디렉터리 아래 경로를 만든다.
///
/// 예제와 테스트가 내려받은 파일을 찾을 때 쓴다. 환경 변수
/// `BADAK_MODELS`가 있으면 그 디렉터리를 대신 쓴다.
pub fn models_dir() -> PathBuf {
    std::env::var_os("BADAK_MODELS").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models"),
        PathBuf::from,
    )
}
