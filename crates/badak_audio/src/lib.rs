//! 소리를 모델의 입력으로 바꾼다 (33장).
//!
//! 음성 인식 모델은 소리의 파형을 직접 받지 않는다. 파형을 짧게 끊어
//! 구간마다 "어떤 높이의 소리가 얼마나 들어 있는가"를 구한 그림(멜
//! 스펙트로그램)을 받는다. 이 크레이트는 소리 파일을 읽는 일부터 그 그림을
//! 만드는 일까지를 한다. 27장의 `badak_image`가 사진에 한 일이다.

pub mod fft;
pub mod istft;
pub mod mel;
pub mod resample;
pub mod vad;
pub mod wav;

pub use fft::Complex;
pub use istft::Istft;
pub use mel::{LogMel, MelSpectrogram};
pub use resample::resample;
pub use wav::Wav;

/// 이 크레이트의 오류.
#[derive(Debug)]
pub enum Error {
    /// 파일을 읽지 못했다.
    Io(std::io::Error),
    /// 파일의 내용이 WAV 형식에 맞지 않는다.
    Format(String),
    /// 형식에는 맞지만 아직 지원하지 않는 내용이다.
    Unsupported(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "파일을 읽지 못했습니다: {error}"),
            Self::Format(message) => write!(f, "WAV 파일이 올바르지 않습니다: {message}"),
            Self::Unsupported(what) => write!(f, "아직 지원하지 않습니다: {what}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

/// 이 크레이트의 결과 타입.
pub type Result<T> = std::result::Result<T, Error>;
