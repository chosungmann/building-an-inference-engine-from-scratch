//! `.tflite` 파일을 읽는다 (26장).
//!
//! `.tflite`는 LiteRT(옛 이름 TensorFlow Lite)의 모델 파일이다. 가중치만
//! 든 safetensors나 GGUF와 달리 **계산의 순서**가 함께 들어 있다. 어떤
//! 연산이 어떤 텐서를 받아 어떤 텐서를 내는지가 파일에 적혀 있다.
//!
//! 파일은 FlatBuffers로 적혀 있고, 무엇이 몇 번째 필드인지는 스키마
//! (`schema.fbs`)가 정한다. 이 모듈의 숫자들은 그 스키마에서 옮긴 것이다.

use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use badak_core::{DType, Tensor};

use crate::error::{Error, Result};
use crate::flatbuffers::Table;
use crate::mmap::Mmap;

const FORMAT: &str = "tflite";
/// 파일의 4~8번째 바이트에 적힌 식별자.
const IDENTIFIER: &[u8; 4] = b"TFL3";
/// 이 파서가 아는 스키마의 판.
const SCHEMA_VERSION: u32 = 3;

// ANCHOR: codes
/// 스키마의 `BuiltinOperator`에서 이 책이 쓰는 것들.
pub mod builtin {
    /// 두 텐서를 더한다.
    pub const ADD: u32 = 0;
    /// 창 안의 평균을 낸다 (27장).
    pub const AVERAGE_POOL_2D: u32 = 1;
    /// 합성곱 (27장).
    pub const CONV_2D: u32 = 3;
    /// 채널마다 따로 하는 합성곱 (27장).
    pub const DEPTHWISE_CONV_2D: u32 = 4;
    /// 선형 층.
    pub const FULLY_CONNECTED: u32 = 9;
    /// 창 안의 최댓값을 고른다 (27장).
    pub const MAX_POOL_2D: u32 = 17;
    /// 두 텐서를 곱한다.
    pub const MUL: u32 = 18;
    /// ReLU.
    pub const RELU: u32 = 19;
    /// 소프트맥스.
    pub const SOFTMAX: u32 = 25;
    /// 파일에 이름으로 적힌, 표준에 없는 연산.
    pub const CUSTOM: u32 = 32;
    /// 가장자리에 값을 패딩한다 (27장).
    pub const PAD: u32 = 34;
    /// 축의 순서를 바꾼다 (27장).
    pub const TRANSPOSE: u32 = 39;
    /// 축을 따라 평균을 낸다 (27장).
    pub const MEAN: u32 = 40;
    /// 가장자리에 정해 준 값을 패딩한다 (27장).
    pub const PADV2: u32 = 60;
    /// 옛 필드(1바이트)에 담을 수 있는 가장 큰 값. 이보다 큰 번호는 새
    /// 필드에만 적힌다.
    pub const PLACEHOLDER_FOR_GREATER_OP_CODES: u32 = 127;
}

/// 스키마의 `BuiltinOptions`에서 이 책이 쓰는 것들. 연산마다 설정을 담는
/// 표의 종류가 다르고, 이 번호가 그 종류를 말한다.
mod options {
    pub const CONV_2D: u8 = 1;
    pub const DEPTHWISE_CONV_2D: u8 = 2;
    pub const POOL_2D: u8 = 5;
    pub const REDUCER: u8 = 27;
    pub const FULLY_CONNECTED: u8 = 8;
    pub const SOFTMAX: u8 = 9;
    pub const ADD: u8 = 11;
    pub const MUL: u8 = 21;
}
// ANCHOR_END: codes

// ANCHOR: types
/// 텐서의 원소가 어떤 숫자인지. 스키마의 `TensorType`이다.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorType {
    /// 32비트 부동소수점.
    Float32,
    /// 16비트 부동소수점.
    Float16,
    /// 32비트 정수.
    Int32,
    /// 부호 없는 8비트 정수.
    UInt8,
    /// 64비트 정수.
    Int64,
    /// 부호 있는 8비트 정수.
    Int8,
    /// 이 책이 아직 다루지 않는 자료형. 스키마의 번호를 담는다.
    Other(i8),
}

impl TensorType {
    fn from_code(code: i8) -> Self {
        match code {
            0 => Self::Float32,
            1 => Self::Float16,
            2 => Self::Int32,
            3 => Self::UInt8,
            4 => Self::Int64,
            9 => Self::Int8,
            other => Self::Other(other),
        }
    }
}

/// 연산의 결과에 곧바로 이어 붙이는 활성화 함수. 스키마의
/// `ActivationFunctionType`이다.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// 없음.
    None,
    /// `max(x, 0)`
    Relu,
    /// `-1`과 `1` 사이로 자른다.
    ReluN1To1,
    /// `0`과 `6` 사이로 자른다.
    Relu6,
    /// 쌍곡 탄젠트.
    Tanh,
    /// 이 책이 다루지 않는 것.
    Other(i8),
}

impl Activation {
    fn from_code(code: i8) -> Self {
        match code {
            0 => Self::None,
            1 => Self::Relu,
            2 => Self::ReluN1To1,
            3 => Self::Relu6,
            4 => Self::Tanh,
            other => Self::Other(other),
        }
    }
}

// ANCHOR: padding
/// 합성곱과 풀링이 가장자리를 다루는 방법. 스키마의 `Padding`이다.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Padding {
    /// 출력의 크기가 `입력 / 스트라이드`(올림)이 되도록 가장자리에 0을 패딩한다.
    Same,
    /// 패딩하지 않는다. 창이 입력 안에 온전히 들어가는 자리만 계산한다.
    Valid,
}

impl Padding {
    fn from_code(code: i8) -> Result<Self> {
        match code {
            0 => Ok(Self::Same),
            1 => Ok(Self::Valid),
            other => Err(invalid(format!("패딩 방법 {other}번을 모릅니다"))),
        }
    }
}
// ANCHOR_END: padding

/// 연산의 종류. 모델이 쓰는 종류를 한 번씩만 적어 두고, 연산마다 이 목록의
/// 번호로 가리킨다.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperatorCode {
    /// 표준 연산의 번호([`builtin`]).
    pub builtin: u32,
    /// 표준에 없는 연산이면 그 이름.
    pub custom: Option<String>,
    /// 연산의 판. 설정이 추가될 때마다 오른다.
    pub version: i32,
}

/// 연산마다 다른 설정.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Options {
    /// 설정이 없다.
    None,
    /// 선형 층.
    FullyConnected {
        /// 결과에 이어 붙이는 활성화 함수.
        activation: Activation,
        /// 입력의 앞쪽 차원들을 결과에 그대로 둘지.
        keep_num_dims: bool,
    },
    /// 덧셈.
    Add {
        /// 결과에 이어 붙이는 활성화 함수.
        activation: Activation,
    },
    /// 곱셈.
    Mul {
        /// 결과에 이어 붙이는 활성화 함수.
        activation: Activation,
    },
    /// 합성곱 (27장).
    Conv2d {
        /// 가장자리를 다루는 방법.
        padding: Padding,
        /// 스트라이드. (세로, 가로).
        stride: [usize; 2],
        /// 창의 칸 사이의 간격. (세로, 가로).
        dilation: [usize; 2],
        /// 결과에 이어 붙이는 활성화 함수.
        activation: Activation,
    },
    /// 채널마다 따로 하는 합성곱 (27장).
    DepthwiseConv2d {
        /// 가장자리를 다루는 방법.
        padding: Padding,
        /// 스트라이드. (세로, 가로).
        stride: [usize; 2],
        /// 창의 칸 사이의 간격. (세로, 가로).
        dilation: [usize; 2],
        /// 입력 채널 하나가 내는 출력 채널의 수.
        multiplier: usize,
        /// 결과에 이어 붙이는 활성화 함수.
        activation: Activation,
    },
    /// 풀링 (27장). 최댓값과 평균이 같은 설정을 쓴다.
    Pool2d {
        /// 가장자리를 다루는 방법.
        padding: Padding,
        /// 스트라이드. (세로, 가로).
        stride: [usize; 2],
        /// 창의 크기. (세로, 가로).
        filter: [usize; 2],
        /// 결과에 이어 붙이는 활성화 함수.
        activation: Activation,
    },
    /// 축을 따라 줄이는 연산 (27장).
    Reducer {
        /// 줄인 축을 길이 1로 남길지.
        keep_dims: bool,
    },
    /// 소프트맥스. 점수에 `beta`를 곱한 뒤 계산한다.
    Softmax {
        /// 점수에 곱하는 값.
        beta: f32,
    },
    /// 이 파서가 풀지 않는 설정. 스키마의 번호를 담는다.
    Other(u8),
}

/// 연산 하나. 그래프의 마디다.
#[derive(Debug, Clone, PartialEq)]
pub struct Operator {
    /// [`Model::operator_codes`]에서의 번호.
    pub code: usize,
    /// 입력 텐서들의 번호. 생략한 입력은 `None`이다.
    pub inputs: Vec<Option<usize>>,
    /// 출력 텐서들의 번호.
    pub outputs: Vec<usize>,
    /// 설정.
    pub options: Options,
}

// ANCHOR: quantization
/// 정수로 적힌 값을 실수로 되돌리는 방법 (29장). 스키마의
/// `QuantizationParameters`다. 실수 = 눈금 × (정수 − 영점).
#[derive(Debug, Clone, PartialEq)]
pub struct Quantization {
    /// 눈금. 하나이면 텐서 전체에, 여럿이면 `axis` 축의 자리마다 하나씩이다.
    pub scales: Vec<f32>,
    /// 영점. 실수 0에 해당하는 정수다. 눈금과 수가 같다.
    pub zero_points: Vec<i32>,
    /// 눈금이 여럿일 때 그것들이 따르는 축.
    pub axis: usize,
}
// ANCHOR_END: quantization

/// 텐서 하나에 대한 설명. 값이 아니라 값이 놓일 자리의 이름표다.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorInfo {
    /// 이름.
    pub name: String,
    /// 형상.
    pub shape: Vec<usize>,
    /// 자료형.
    pub dtype: TensorType,
    /// 값이 든 버퍼의 번호. 0이면 파일에 값이 없다. 실행하면서 계산되는
    /// 텐서라는 뜻이다.
    pub buffer: usize,
    /// 정수 텐서이면 실수로 되돌리는 방법 (29장).
    pub quantization: Option<Quantization>,
}

/// 부분 그래프 하나. 텐서들과, 그것들을 잇는 연산들이다.
#[derive(Debug, Clone, PartialEq)]
pub struct Subgraph {
    /// 이름.
    pub name: String,
    /// 텐서들.
    pub tensors: Vec<TensorInfo>,
    /// 밖에서 값을 넣어 주는 텐서들의 번호.
    pub inputs: Vec<usize>,
    /// 결과로 내놓는 텐서들의 번호.
    pub outputs: Vec<usize>,
    /// 연산들. 실행할 순서대로 적혀 있다.
    pub operators: Vec<Operator>,
}

/// `.tflite` 파일 하나.
pub struct Model {
    /// 스키마의 판.
    pub version: u32,
    /// 파일을 만든 도구가 남긴 설명.
    pub description: String,
    /// 모델이 쓰는 연산의 종류들.
    pub operator_codes: Vec<OperatorCode>,
    /// 부분 그래프들. 첫째가 모델의 본체다.
    pub subgraphs: Vec<Subgraph>,
    /// 버퍼마다 값이 놓인 파일 안의 구간.
    buffers: Vec<Range<usize>>,
    owner: Arc<dyn AsRef<[u8]> + Send + Sync>,
}
// ANCHOR_END: types

fn invalid(message: impl Into<String>) -> Error {
    Error::Parse {
        format: FORMAT,
        offset: 0,
        message: message.into(),
    }
}

/// 파일에 적힌 번호를 `usize`로 바꾸고 `count`보다 작은지 본다.
fn index(value: i64, count: usize, what: &str) -> Result<usize> {
    usize::try_from(value)
        .ok()
        .filter(|&index| index < count)
        .ok_or_else(|| {
            invalid(format!(
                "{what} 번호 {value}가 범위(0..{count})를 벗어납니다"
            ))
        })
}

impl Model {
    /// 파일을 메모리에 비춰 읽는다. 가중치는 복사하지 않는다.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::parse(Arc::new(Mmap::open(path)?))
    }

    // ANCHOR: parse
    /// 바이트를 해석한다. `owner`는 그 바이트를 가진 것이고, 가중치 텐서가
    /// 이것을 빌려 쓴다.
    pub fn parse(owner: Arc<dyn AsRef<[u8]> + Send + Sync>) -> Result<Self> {
        let bytes = (*owner).as_ref();
        let root = Table::root(bytes, Some(IDENTIFIER), FORMAT)?;
        let version = root.scalar::<u32, 4>(0, 0)?;
        if version != SCHEMA_VERSION {
            return Err(Error::Unsupported(format!(
                "tflite 스키마 {version}판 ({SCHEMA_VERSION}판만 읽습니다)"
            )));
        }

        let mut operator_codes = Vec::new();
        if let Some(codes) = root.vector(1, 4)? {
            for code in codes.tables() {
                operator_codes.push(operator_code(&code?)?);
            }
        }

        // 버퍼의 값은 읽지 않고 어디 있는지만 적어 둔다.
        let mut buffers = Vec::new();
        if let Some(tables) = root.vector(4, 4)? {
            for buffer in tables.tables() {
                let buffer = buffer?;
                // 값이 FlatBuffers 밖에 있을 수도 있다(29장). 그때는 파일의
                // 처음부터 잰 위치와 길이가 적혀 있다. 2GB를 넘는 모델을 위한
                // 방법인데, 작은 파일에 쓰는 도구도 있다.
                let (offset, size) = (
                    buffer.scalar::<u64, 8>(1, 0)?,
                    buffer.scalar::<u64, 8>(2, 0)?,
                );
                if offset > 1 {
                    let range = usize::try_from(offset)
                        .ok()
                        .zip(usize::try_from(size).ok())
                        .and_then(|(start, len)| Some(start..start.checked_add(len)?))
                        .filter(|range| range.end <= bytes.len())
                        .ok_or_else(|| invalid("버퍼가 파일을 벗어납니다"))?;
                    buffers.push(range);
                    continue;
                }
                buffers.push(buffer.vector(0, 1)?.map_or(0..0, |data| data.bytes()));
            }
        }

        let mut subgraphs = Vec::new();
        if let Some(tables) = root.vector(2, 4)? {
            for table in tables.tables() {
                subgraphs.push(subgraph(&table?, operator_codes.len(), buffers.len())?);
            }
        }
        if subgraphs.is_empty() {
            return Err(invalid("부분 그래프가 없습니다"));
        }
        let description = root.string(3)?.unwrap_or_default().to_string();
        Ok(Self {
            version,
            description,
            operator_codes,
            subgraphs,
            buffers,
            owner,
        })
    }
    // ANCHOR_END: parse

    // ANCHOR: constant
    /// 파일에 값이 든 텐서(가중치)를 꺼낸다. 값이 없는 텐서면 `None`이다.
    ///
    /// 값을 복사하지 않고 파일을 비춘 메모리를 그대로 가리킨다(6.4절).
    pub fn constant(&self, info: &TensorInfo) -> Result<Option<Tensor>> {
        let range = self
            .buffers
            .get(info.buffer)
            .ok_or_else(|| invalid(format!("버퍼 {}가 없습니다", info.buffer)))?;
        if info.buffer == 0 || range.is_empty() {
            return Ok(None);
        }
        let dtype = match info.dtype {
            TensorType::Float32 => DType::F32,
            TensorType::Int32 => DType::I32,
            TensorType::UInt8 => DType::U8,
            // 부호 있는 8비트는 바이트 그대로 읽는다. 쓰는 쪽이 부호를 붙여
            // 해석한다(29장).
            TensorType::Int8 => DType::U8,
            other => {
                return Err(Error::Unsupported(format!(
                    "자료형이 {other:?}인 텐서 {:?}",
                    info.name
                )));
            }
        };
        let tensor = Tensor::from_bytes(Arc::clone(&self.owner), range.clone(), dtype, &info.shape)
            .map_err(|error| invalid(format!("텐서 {:?}: {error}", info.name)))?;
        Ok(Some(tensor))
    }
    // ANCHOR_END: constant

    /// 파일 전체의 크기(바이트).
    pub fn file_len(&self) -> usize {
        (*self.owner).as_ref().len()
    }
}

// ANCHOR: operator_code
fn operator_code(table: &Table<'_>) -> Result<OperatorCode> {
    // 연산의 번호는 두 필드에 적힌다. 처음에는 1바이트였는데 연산이 127가지를
    // 넘게 되자 4바이트 필드를 새로 만들었다. 옛 프로그램도 읽을 수 있게
    // 둘 다 적고, 읽는 쪽은 큰 쪽을 쓴다.
    let deprecated = table.scalar::<i8, 1>(0, 0)?;
    let extended = table.scalar::<i32, 4>(3, 0)?;
    let builtin = u32::try_from(extended.max(i32::from(deprecated)))
        .map_err(|_| invalid("연산의 번호가 음수입니다"))?;
    Ok(OperatorCode {
        builtin,
        custom: table.string(1)?.map(str::to_string),
        version: table.scalar::<i32, 4>(2, 1)?,
    })
}
// ANCHOR_END: operator_code

// ANCHOR: subgraph
fn subgraph(table: &Table<'_>, code_count: usize, buffer_count: usize) -> Result<Subgraph> {
    let mut tensors = Vec::new();
    if let Some(tables) = table.vector(0, 4)? {
        for tensor in tables.tables() {
            tensors.push(tensor_info(&tensor?, buffer_count)?);
        }
    }
    let count = tensors.len();
    let indices = |slot: usize, what: &str| -> Result<Vec<usize>> {
        let Some(vector) = table.vector(slot, 4)? else {
            return Ok(Vec::new());
        };
        vector
            .scalars::<i32, 4>()?
            .into_iter()
            .map(|value| index(i64::from(value), count, what))
            .collect()
    };
    let inputs = indices(1, "입력 텐서")?;
    let outputs = indices(2, "출력 텐서")?;

    let mut operators = Vec::new();
    if let Some(tables) = table.vector(3, 4)? {
        for operator in tables.tables() {
            operators.push(self::operator(&operator?, code_count, count)?);
        }
    }
    Ok(Subgraph {
        name: table.string(4)?.unwrap_or_default().to_string(),
        tensors,
        inputs,
        outputs,
        operators,
    })
}

fn tensor_info(table: &Table<'_>, buffer_count: usize) -> Result<TensorInfo> {
    let name = table.string(3)?.unwrap_or_default().to_string();
    let shape = match table.vector(0, 4)? {
        Some(vector) => vector
            .scalars::<i32, 4>()?
            .into_iter()
            .map(|extent| {
                usize::try_from(extent)
                    .map_err(|_| invalid(format!("텐서 {name:?}의 형상에 음수가 있습니다")))
            })
            .collect::<Result<Vec<_>>>()?,
        None => Vec::new(),
    };
    // 형상의 곱이 원소의 수다. 깨진 파일은 곱이 넘치는 형상을 적을 수 있다.
    shape
        .iter()
        .try_fold(1_usize, |count, &extent| count.checked_mul(extent))
        .ok_or_else(|| invalid(format!("텐서 {name:?}의 형상 {shape:?}가 너무 큽니다")))?;
    // 양자화 설정: 눈금(2번 필드)과 영점(3번 필드)의 벡터, 축(6번 필드).
    let quantization = match table.table(4)? {
        Some(parameters) => {
            let scales = match parameters.vector(2, 4)? {
                Some(vector) => vector.scalars::<f32, 4>()?,
                None => Vec::new(),
            };
            let zero_points = match parameters.vector(3, 8)? {
                Some(vector) => vector
                    .scalars::<i64, 8>()?
                    .into_iter()
                    .map(|zero| {
                        i32::try_from(zero).map_err(|_| invalid("영점이 32비트를 넘습니다"))
                    })
                    .collect::<Result<Vec<_>>>()?,
                None => Vec::new(),
            };
            let axis = usize::try_from(parameters.scalar::<i32, 4>(6, 0)?)
                .map_err(|_| invalid("양자화 축이 음수입니다"))?;
            if scales.len() != zero_points.len() {
                return Err(invalid(format!(
                    "텐서 {name:?}의 눈금과 영점의 수가 다릅니다"
                )));
            }
            (!scales.is_empty()).then_some(Quantization {
                scales,
                zero_points,
                axis,
            })
        }
        None => None,
    };
    Ok(TensorInfo {
        quantization,
        shape,
        dtype: TensorType::from_code(table.scalar::<i8, 1>(1, 0)?),
        buffer: index(
            i64::from(table.scalar::<u32, 4>(2, 0)?),
            buffer_count.max(1),
            "버퍼",
        )?,
        name,
    })
}

fn operator(table: &Table<'_>, code_count: usize, tensor_count: usize) -> Result<Operator> {
    let code = index(
        i64::from(table.scalar::<u32, 4>(0, 0)?),
        code_count,
        "연산 종류",
    )?;
    let numbers = |slot: usize| -> Result<Vec<i32>> {
        match table.vector(slot, 4)? {
            Some(vector) => vector.scalars::<i32, 4>(),
            None => Ok(Vec::new()),
        }
    };
    let inputs = numbers(1)?
        .into_iter()
        .map(|value| match value {
            // 생략한 입력은 -1로 적는다. 치우침이 없는 선형 층이 그렇다.
            -1 => Ok(None),
            value => index(i64::from(value), tensor_count, "텐서").map(Some),
        })
        .collect::<Result<Vec<_>>>()?;
    let outputs = numbers(2)?
        .into_iter()
        .map(|value| index(i64::from(value), tensor_count, "텐서"))
        .collect::<Result<Vec<_>>>()?;

    // 설정은 "어떤 종류의 표인가"(3번 필드)와 "그 표"(4번 필드)의 쌍이다.
    // FlatBuffers가 여러 종류 가운데 하나를 담는 방법(union)이다.
    let kind = table.scalar::<u8, 1>(3, 0)?;
    let activation = |table: &Table<'_>| -> Result<Activation> {
        Ok(Activation::from_code(table.scalar::<i8, 1>(0, 0)?))
    };
    // 스트라이드나 창의 크기처럼 양수여야 하는 값을 읽는다.
    let positive = |table: &Table<'_>, slot: usize, default: i32| -> Result<usize> {
        usize::try_from(table.scalar::<i32, 4>(slot, default)?)
            .ok()
            .filter(|&value| value > 0)
            .ok_or_else(|| invalid("스트라이드나 창의 크기가 양수가 아닙니다"))
    };
    let options = match (kind, table.table(4)?) {
        (0, _) | (_, None) => Options::None,
        (options::CONV_2D, Some(options)) => Options::Conv2d {
            padding: Padding::from_code(options.scalar::<i8, 1>(0, 0)?)?,
            stride: [positive(&options, 2, 0)?, positive(&options, 1, 0)?],
            activation: Activation::from_code(options.scalar::<i8, 1>(3, 0)?),
            dilation: [positive(&options, 5, 1)?, positive(&options, 4, 1)?],
        },
        (options::DEPTHWISE_CONV_2D, Some(options)) => Options::DepthwiseConv2d {
            padding: Padding::from_code(options.scalar::<i8, 1>(0, 0)?)?,
            stride: [positive(&options, 2, 0)?, positive(&options, 1, 0)?],
            multiplier: positive(&options, 3, 0)?,
            activation: Activation::from_code(options.scalar::<i8, 1>(4, 0)?),
            dilation: [positive(&options, 6, 1)?, positive(&options, 5, 1)?],
        },
        (options::POOL_2D, Some(options)) => Options::Pool2d {
            padding: Padding::from_code(options.scalar::<i8, 1>(0, 0)?)?,
            stride: [positive(&options, 2, 0)?, positive(&options, 1, 0)?],
            filter: [positive(&options, 4, 0)?, positive(&options, 3, 0)?],
            activation: Activation::from_code(options.scalar::<i8, 1>(5, 0)?),
        },
        (options::REDUCER, Some(options)) => Options::Reducer {
            keep_dims: options.scalar::<u8, 1>(0, 0)? != 0,
        },
        (options::FULLY_CONNECTED, Some(options)) => {
            if options.scalar::<i8, 1>(1, 0)? != 0 {
                return Err(Error::Unsupported(
                    "가중치를 다른 배치로 담은 FULLY_CONNECTED".into(),
                ));
            }
            Options::FullyConnected {
                activation: activation(&options)?,
                keep_num_dims: options.scalar::<u8, 1>(2, 0)? != 0,
            }
        }
        (options::ADD, Some(options)) => Options::Add {
            activation: activation(&options)?,
        },
        (options::MUL, Some(options)) => Options::Mul {
            activation: activation(&options)?,
        },
        (options::SOFTMAX, Some(options)) => Options::Softmax {
            beta: options.scalar::<f32, 4>(0, 0.0)?,
        },
        (other, Some(_)) => Options::Other(other),
    };
    Ok(Operator {
        code,
        inputs,
        outputs,
        options,
    })
}
// ANCHOR_END: subgraph

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flatbuffers::build::{Value, file};

    fn floats(values: &[f32]) -> Value {
        Value::Bytes(values.iter().flat_map(|v| v.to_le_bytes()).collect())
    }

    fn tensor(name: &str, shape: &[i32], buffer: u32) -> Vec<Value> {
        vec![
            Value::Ints(shape.to_vec()),
            Value::I8(0),
            Value::U32(buffer),
            Value::Str(name.into()),
        ]
    }

    /// 선형 층 하나와 소프트맥스로 된 모델. `y = softmax(relu(W x + b))`.
    fn tiny() -> Vec<u8> {
        let fully_connected = vec![
            Value::U32(0),
            Value::Ints(vec![0, 1, 2]),
            Value::Ints(vec![3]),
            Value::U8(options::FULLY_CONNECTED),
            Value::Table(vec![Value::I8(1)]),
        ];
        let softmax = vec![
            Value::U32(1),
            Value::Ints(vec![3]),
            Value::Ints(vec![4]),
            Value::U8(options::SOFTMAX),
            Value::Table(vec![Value::F32(1.0)]),
        ];
        let subgraph = vec![
            Value::Tables(vec![
                tensor("x", &[1, 3], 0),
                tensor("w", &[2, 3], 1),
                tensor("b", &[2], 2),
                tensor("h", &[1, 2], 0),
                tensor("y", &[1, 2], 0),
            ]),
            Value::Ints(vec![0]),
            Value::Ints(vec![4]),
            Value::Tables(vec![fully_connected, softmax]),
            Value::Str("main".into()),
        ];
        file(
            IDENTIFIER,
            &[
                Value::U32(3),
                Value::Tables(vec![
                    // 옛 필드와 새 필드에 같은 번호를 적는다.
                    vec![Value::I8(9), Value::Absent, Value::I32(1), Value::I32(9)],
                    vec![Value::I8(25), Value::Absent, Value::I32(1), Value::I32(25)],
                ]),
                Value::Tables(vec![subgraph]),
                Value::Str("시험 모델".into()),
                Value::Tables(vec![
                    vec![],
                    vec![floats(&[1.0, 0.0, -1.0, 0.5, 0.5, 0.5])],
                    vec![floats(&[0.25, -0.25])],
                ]),
            ],
        )
    }

    #[test]
    fn parses_a_hand_written_model() {
        let model = Model::parse(Arc::new(tiny())).unwrap();
        assert_eq!(model.version, 3);
        assert_eq!(model.description, "시험 모델");
        assert_eq!(model.operator_codes[0].builtin, builtin::FULLY_CONNECTED);
        assert_eq!(model.operator_codes[1].builtin, builtin::SOFTMAX);
        let graph = &model.subgraphs[0];
        assert_eq!(graph.name, "main");
        assert_eq!(
            (graph.inputs.as_slice(), graph.outputs.as_slice()),
            (&[0][..], &[4][..])
        );
        assert_eq!(graph.tensors[1].shape, [2, 3]);
        assert_eq!(
            graph.operators[0],
            Operator {
                code: 0,
                inputs: vec![Some(0), Some(1), Some(2)],
                outputs: vec![3],
                options: Options::FullyConnected {
                    activation: Activation::Relu,
                    keep_num_dims: false,
                },
            }
        );
        assert_eq!(graph.operators[1].options, Options::Softmax { beta: 1.0 });

        // 계산되는 텐서에는 값이 없고, 가중치에는 있다.
        assert!(model.constant(&graph.tensors[0]).unwrap().is_none());
        let weight = model.constant(&graph.tensors[1]).unwrap().unwrap();
        assert_eq!(
            weight.to_vec::<f32>().unwrap(),
            [1.0, 0.0, -1.0, 0.5, 0.5, 0.5]
        );
        let bias = model.constant(&graph.tensors[2]).unwrap().unwrap();
        assert_eq!(bias.to_vec::<f32>().unwrap(), [0.25, -0.25]);
    }

    #[test]
    fn new_operator_codes_use_the_wide_field() {
        // 127번보다 큰 연산은 옛 필드에 127을 적고 새 필드에 진짜 번호를
        // 적는다.
        let bytes = file(
            IDENTIFIER,
            &[
                Value::U32(3),
                Value::Tables(vec![vec![
                    Value::I8(127),
                    Value::Absent,
                    Value::I32(2),
                    Value::I32(150),
                ]]),
                Value::Tables(vec![vec![]]),
            ],
        );
        let model = Model::parse(Arc::new(bytes)).unwrap();
        assert_eq!(model.operator_codes[0].builtin, 150);
        assert_eq!(model.operator_codes[0].version, 2);
    }

    #[test]
    fn rejects_other_files() {
        assert!(Model::parse(Arc::new(b"GGUF\x03\x00\x00\x00".to_vec())).is_err());
        assert!(Model::parse(Arc::new(Vec::new())).is_err());
        let mut wrong_version = tiny();
        // 뿌리 표의 첫 필드(스키마의 판)를 찾아 4로 바꾼다.
        let root = u32::from_le_bytes(wrong_version[0..4].try_into().unwrap()) as usize;
        let vtable =
            root - i32::from_le_bytes(wrong_version[root..root + 4].try_into().unwrap()) as usize;
        let field = u16::from_le_bytes(wrong_version[vtable + 4..vtable + 6].try_into().unwrap());
        wrong_version[root + usize::from(field)] = 4;
        assert!(matches!(
            Model::parse(Arc::new(wrong_version)),
            Err(Error::Unsupported(_))
        ));
    }

    #[test]
    fn broken_files_are_errors_not_panics() {
        let bytes = tiny();
        let read = |bytes: Vec<u8>| {
            if let Ok(model) = Model::parse(Arc::new(bytes)) {
                for graph in &model.subgraphs {
                    for tensor in &graph.tensors {
                        let _ = model.constant(tensor);
                    }
                }
            }
        };
        for len in 0..bytes.len() {
            read(bytes[..len].to_vec());
        }
        for position in 0..bytes.len() {
            for value in [0x00, 0x01, 0x7f, 0x80, 0xff] {
                let mut broken = bytes.clone();
                broken[position] = value;
                read(broken);
            }
        }
    }
}
