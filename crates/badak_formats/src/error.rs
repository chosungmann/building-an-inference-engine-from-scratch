//! 모델 파일을 읽다가 생기는 오류.

use std::fmt;

// ANCHOR: error
/// 모델 파일을 열거나 해석할 때 생기는 오류.
#[derive(Debug)]
pub enum Error {
    /// 파일을 열거나 읽지 못했다.
    Io(std::io::Error),
    /// 파일의 내용이 형식에 맞지 않는다.
    Parse {
        /// 어떤 형식을 읽고 있었는지.
        format: &'static str,
        /// 문제가 생긴 위치(바이트).
        offset: usize,
        /// 무엇이 잘못되었는지.
        message: String,
    },
    /// 형식에는 맞지만 아직 지원하지 않는 내용이다.
    Unsupported(String),
    /// 그런 이름의 텐서가 없다.
    NotFound(String),
    /// 텐서를 만들다가 생긴 오류.
    Tensor(badak_core::Error),
}
// ANCHOR_END: error

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "파일을 읽지 못했습니다: {error}"),
            Self::Parse {
                format,
                offset,
                message,
            } => write!(
                f,
                "{format} 파일이 올바르지 않습니다 (위치 {offset}): {message}"
            ),
            Self::Unsupported(what) => write!(f, "아직 지원하지 않습니다: {what}"),
            Self::NotFound(name) => write!(f, "텐서 {name:?}가 파일에 없습니다"),
            Self::Tensor(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Tensor(error) => Some(error),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<badak_core::Error> for Error {
    fn from(error: badak_core::Error) -> Self {
        Self::Tensor(error)
    }
}

/// 이 크레이트의 결과 타입.
pub type Result<T> = std::result::Result<T, Error>;
