//! 사진 분류기. 문 안쪽의 보통 Rust다.

use std::fmt;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use badak_formats::Mmap;
use badak_formats::tflite::Model;
use badak_graph::{Graph, Interpreter, Registry};
use badak_image::{Image, Preprocess};

/// 문 밖으로 나가는 오류의 종류. 값은 `badak.h`의 `BADAK_*`와 같다.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Status {
    /// 잘 끝났다.
    Ok = 0,
    /// 넘겨받은 값이 잘못되었다(빈 포인터, 맞지 않는 길이).
    InvalidArgument = 1,
    /// 파일을 읽지 못했다.
    Io = 2,
    /// 모델 파일이 올바르지 않거나 이 엔진이 돌리지 못한다.
    Model = 3,
    /// 사진 파일을 풀지 못했다.
    Image = 4,
    /// 받은 모델 파일이 명세와 다르다. 다시 받아야 한다.
    Mismatch = 5,
    /// 엔진 안에서 패닉이 났다. 엔진의 버그다.
    Panic = 6,
}

/// 종류와 사람이 읽을 설명.
#[derive(Debug)]
pub struct Error {
    /// 종류.
    pub status: Status,
    /// 설명.
    pub message: String,
}

impl Error {
    /// 넘겨받은 값이 잘못되었다는 오류.
    pub fn argument(message: impl Into<String>) -> Self {
        Self {
            status: Status::InvalidArgument,
            message: message.into(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Error {}

impl From<badak_formats::Error> for Error {
    fn from(error: badak_formats::Error) -> Self {
        let status = match &error {
            badak_formats::Error::Io(_) => Status::Io,
            other if badak_formats::store::is_mismatch(other) => Status::Mismatch,
            _ => Status::Model,
        };
        Self {
            status,
            message: error.to_string(),
        }
    }
}

impl From<badak_graph::Error> for Error {
    fn from(error: badak_graph::Error) -> Self {
        match error {
            badak_graph::Error::Format(inner) => inner.into(),
            other => Self {
                status: Status::Model,
                message: other.to_string(),
            },
        }
    }
}

impl From<badak_image::Error> for Error {
    fn from(error: badak_image::Error) -> Self {
        Self {
            status: Status::Image,
            message: error.to_string(),
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self {
            status: Status::Io,
            message: error.to_string(),
        }
    }
}

/// 분류 결과 하나. C의 `badak_score`와 메모리에서 같은 모양이다.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct Score {
    /// 부류의 번호.
    pub index: u32,
    /// 확률.
    pub probability: f32,
}

// ANCHOR: classifier
/// 모델 하나와 그것을 돌릴 준비. 한 번에 한 스레드만 써야 한다.
pub struct Classifier {
    interpreter: Interpreter,
}

impl fmt::Debug for Classifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Classifier")
            .field("input_len", &self.input_len())
            .field("output_len", &self.output_len())
            .finish()
    }
}

impl Classifier {
    /// 모델 파일을 메모리에 매핑해 연다.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        Self::from_model(&Model::open(path)?)
    }

    /// 열려 있는 파일의 한 구간에 든 모델을 메모리에 매핑해 연다.
    pub fn open_range(file: &File, offset: u64, len: usize) -> Result<Self, Error> {
        let map = Mmap::open_range(file, offset, len)?;
        Self::from_model(&Model::parse(Arc::new(map))?)
    }

    fn from_model(model: &Model) -> Result<Self, Error> {
        let graph = Graph::from_tflite(model)?;
        let interpreter = Interpreter::new(&graph, &Registry::builtin())?;
        Ok(Self { interpreter })
    }

    /// 입력의 값의 수.
    pub fn input_len(&self) -> usize {
        self.interpreter.input_shape(0).iter().product()
    }

    /// 출력의 값의 수.
    pub fn output_len(&self) -> usize {
        self.interpreter.output_shape(0).iter().product()
    }

    /// 이미 숫자로 바꾼 입력을 넣고 점수를 받는다.
    pub fn run(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), Error> {
        if input.len() != self.input_len() || output.len() != self.output_len() {
            return Err(Error::argument(format!(
                "입력 {}개와 출력 {}개를 받았습니다 (모델은 {}개와 {}개)",
                input.len(),
                output.len(),
                self.input_len(),
                self.output_len()
            )));
        }
        self.interpreter.input_mut(0).copy_from_slice(input);
        self.interpreter.run()?;
        output.copy_from_slice(self.interpreter.output(0));
        Ok(())
    }

    /// JPEG 파일의 바이트를 받아 확률이 높은 부류부터 `top`에 채운다. 채운
    /// 수를 돌려준다. 전처리는 27장의 것이다.
    pub fn classify_jpeg(&mut self, bytes: &[u8], top: &mut [Score]) -> Result<usize, Error> {
        let image = Image::decode_jpeg(bytes)?;
        let input = Preprocess::IMAGENET.apply(&image);
        if input.len() != self.input_len() {
            return Err(Error::argument(format!(
                "이 모델의 입력은 값 {}개입니다 (사진에서 만든 것은 {}개)",
                self.input_len(),
                input.len()
            )));
        }
        self.interpreter.input_mut(0).copy_from_slice(&input);
        self.interpreter.run()?;
        let probabilities = badak_core::prob::softmax(self.interpreter.output(0));
        let mut order: Vec<usize> = (0..probabilities.len()).collect();
        order.sort_by(|&a, &b| probabilities[b].total_cmp(&probabilities[a]));
        let count = top.len().min(order.len());
        for (slot, &index) in top.iter_mut().zip(&order) {
            *slot = Score {
                index: index as u32,
                probability: probabilities[index],
            };
        }
        Ok(count)
    }
}
// ANCHOR_END: classifier
