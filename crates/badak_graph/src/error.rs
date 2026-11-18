//! 그래프를 만들고 실행하다가 생기는 오류.

use std::fmt;

/// 그래프 해석기의 오류.
#[derive(Debug)]
pub enum Error {
    /// 그래프가 올바르지 않다. 없는 값을 가리키거나, 돌고 도는 길이 있거나,
    /// 형상이 맞지 않는다.
    Graph(String),
    /// 올바른 그래프이지만 이 해석기가 아직 실행하지 못한다.
    Unsupported(String),
    /// 모델 파일을 읽다가 생긴 오류.
    Format(badak_formats::Error),
    /// 텐서를 다루다가 생긴 오류.
    Tensor(badak_core::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Graph(message) => write!(f, "그래프가 올바르지 않습니다: {message}"),
            Self::Unsupported(what) => write!(f, "아직 지원하지 않습니다: {what}"),
            Self::Format(error) => write!(f, "{error}"),
            Self::Tensor(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Format(error) => Some(error),
            Self::Tensor(error) => Some(error),
            _ => None,
        }
    }
}

impl From<badak_formats::Error> for Error {
    fn from(error: badak_formats::Error) -> Self {
        Self::Format(error)
    }
}

impl From<badak_core::Error> for Error {
    fn from(error: badak_core::Error) -> Self {
        Self::Tensor(error)
    }
}

/// 이 크레이트의 결과 타입.
pub type Result<T> = std::result::Result<T, Error>;
