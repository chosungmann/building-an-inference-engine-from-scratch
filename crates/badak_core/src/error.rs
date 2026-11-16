//! `badak`의 오류 타입 (4장).
//!
//! 형상이나 자료형이 맞지 않는 것은 모델 파일이나 입력에 따라 실행 중에
//! 얼마든지 일어날 수 있다. 그래서 텐서부터는 패닉 대신 [`Result`]로
//! 오류를 돌려준다.

use std::fmt;

use crate::tensor::DType;

// ANCHOR: error
/// 텐서를 만들거나 다룰 때 생기는 오류.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// 원소의 개수가 형상과 맞지 않는다.
    ElementCount {
        /// 형상이 요구하는 원소 수.
        expected: usize,
        /// 실제로 주어진 원소 수.
        actual: usize,
    },
    /// 텐서의 자료형이 요청한 자료형과 다르다.
    DType {
        /// 요청한 자료형.
        expected: DType,
        /// 텐서의 실제 자료형.
        actual: DType,
    },
    /// 축 번호가 차원 수를 벗어났다.
    Axis {
        /// 요청한 축.
        axis: usize,
        /// 텐서의 차원 수.
        ndim: usize,
    },
    /// 위치나 범위가 그 축의 길이를 벗어났다.
    OutOfBounds {
        /// 문제가 된 축.
        axis: usize,
        /// 요청한 위치(범위라면 끝).
        index: usize,
        /// 그 축의 길이.
        len: usize,
    },
    /// 두 형상이 이 연산에 함께 쓰일 수 없다.
    Shape {
        /// 무엇을 하다가 생긴 오류인지.
        operation: &'static str,
        /// 왼쪽(또는 원래) 형상.
        left: Vec<usize>,
        /// 오른쪽(또는 목표) 형상.
        right: Vec<usize>,
    },
    /// 메모리에 순서대로 놓여 있어야 하는 연산을 그렇지 않은 뷰에 썼다.
    NotContiguous,
}
// ANCHOR_END: error

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ElementCount { expected, actual } => {
                write!(
                    f,
                    "형상에는 원소 {expected}개가 필요한데 {actual}개가 주어졌습니다"
                )
            }
            Self::DType { expected, actual } => {
                write!(f, "자료형 {expected}을 요청했지만 텐서는 {actual}입니다")
            }
            Self::Axis { axis, ndim } => {
                write!(f, "축 {axis}은 {ndim}차원 텐서의 범위를 벗어납니다")
            }
            Self::OutOfBounds { axis, index, len } => {
                write!(f, "축 {axis}의 길이는 {len}인데 {index}에 접근했습니다")
            }
            Self::Shape {
                operation,
                left,
                right,
            } => write!(f, "{operation}: 형상 {left:?}와 {right:?}는 맞지 않습니다"),
            Self::NotContiguous => {
                write!(
                    f,
                    "메모리에 순서대로 놓인 텐서가 필요합니다. contiguous()를 먼저 호출하십시오"
                )
            }
        }
    }
}

impl std::error::Error for Error {}

/// `badak`의 결과 타입.
pub type Result<T> = std::result::Result<T, Error>;
