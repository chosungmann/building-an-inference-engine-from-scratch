//! GGUF 형식. llama.cpp가 쓰는 모델 파일이다.
//!
//! 파일은 네 부분이다. 고정 길이 머리, 키와 값으로 된 메타데이터, 텐서
//! 정보의 목록, 그리고 정렬에 맞춰 놓인 텐서의 값. 모든 숫자는 리틀
//! 엔디언이다.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use badak_core::{DType, Tensor};

use crate::error::{Error, Result};
use crate::mmap::Mmap;
use crate::reader::Reader;

const FORMAT: &str = "GGUF";

/// `general.alignment`가 없을 때의 정렬 단위.
const DEFAULT_ALIGNMENT: usize = 32;

/// 배열이 이보다 깊이 중첩되면 거부한다.
const MAX_ARRAY_DEPTH: usize = 4;

/// 텐서의 차원 수가 이보다 크면 거부한다.
const MAX_DIMS: usize = 8;

type Owner = Arc<dyn AsRef<[u8]> + Send + Sync>;

// ANCHOR: metadata_value
/// 메타데이터의 값. 형식이 정한 열세 가지 종류 가운데 하나다.
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    /// 8비트 부호 없는 정수
    U8(u8),
    /// 8비트 부호 있는 정수
    I8(i8),
    /// 16비트 부호 없는 정수
    U16(u16),
    /// 16비트 부호 있는 정수
    I16(i16),
    /// 32비트 부호 없는 정수
    U32(u32),
    /// 32비트 부호 있는 정수
    I32(i32),
    /// 32비트 실수
    F32(f32),
    /// 참 또는 거짓
    Bool(bool),
    /// UTF-8 문자열
    String(String),
    /// 같은 종류의 값이 이어진 배열
    Array(Vec<MetadataValue>),
    /// 64비트 부호 없는 정수
    U64(u64),
    /// 64비트 부호 있는 정수
    I64(i64),
    /// 64비트 실수
    F64(f64),
}
// ANCHOR_END: metadata_value

impl MetadataValue {
    /// 문자열이면 그 내용을 돌려준다.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(text) => Some(text),
            _ => None,
        }
    }

    /// 음이 아닌 정수이면 `u64`로 돌려준다. 정수의 크기는 가리지 않는다.
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Self::U8(n) => Some(n.into()),
            Self::U16(n) => Some(n.into()),
            Self::U32(n) => Some(n.into()),
            Self::U64(n) => Some(n),
            Self::I8(n) => n.try_into().ok(),
            Self::I16(n) => n.try_into().ok(),
            Self::I32(n) => n.try_into().ok(),
            Self::I64(n) => n.try_into().ok(),
            _ => None,
        }
    }

    /// 실수이면 `f64`로 돌려준다.
    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Self::F32(n) => Some(n.into()),
            Self::F64(n) => Some(n),
            _ => None,
        }
    }

    /// 배열이면 그 원소들을 돌려준다.
    pub fn as_array(&self) -> Option<&[MetadataValue]> {
        match self {
            Self::Array(items) => Some(items),
            _ => None,
        }
    }
}

// ANCHOR: ggml_type
/// 텐서의 값이 저장된 방식. 파일에는 번호로 적혀 있다.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    /// 32비트 실수
    F32,
    /// 16비트 실수
    F16,
    /// 8, 16, 32, 64비트 정수와 64비트 실수
    I8,
    /// 16비트 정수
    I16,
    /// 32비트 정수
    I32,
    /// 64비트 정수
    I64,
    /// 64비트 실수
    F64,
    /// 양자화된 형식을 비롯해 이 장에서 다루지 않는 종류. 번호를 그대로
    /// 담는다.
    Other(u32),
}

impl GgmlType {
    fn from_code(code: u32) -> Self {
        match code {
            0 => Self::F32,
            1 => Self::F16,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            other => Self::Other(other),
        }
    }

    /// 원소 하나의 바이트 수. 양자화된 형식은 원소 단위로 저장되지
    /// 않으므로 `None`이다.
    pub fn size_in_bytes(self) -> Option<usize> {
        Some(match self {
            Self::I8 => 1,
            Self::F16 | Self::I16 => 2,
            Self::F32 | Self::I32 => 4,
            Self::I64 | Self::F64 => 8,
            Self::Other(_) => return None,
        })
    }
}
// ANCHOR_END: ggml_type

// ANCHOR: info
/// 텐서 하나의 정보.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    /// 형상. 파일에 적힌 순서를 뒤집어, 이 책의 다른 곳과 같은 행 우선
    /// 순서로 담는다.
    pub shape: Vec<usize>,
    /// 값이 저장된 방식.
    pub ggml_type: GgmlType,
    /// 값이 시작하는 위치. 파일의 처음부터 센 바이트 위치다.
    pub offset: usize,
}

/// 열어 둔 GGUF 파일.
pub struct Gguf {
    owner: Owner,
    version: u32,
    metadata: BTreeMap<String, MetadataValue>,
    tensors: BTreeMap<String, TensorInfo>,
}
// ANCHOR_END: info

impl Gguf {
    /// 파일을 메모리에 비추고 머리, 메타데이터, 텐서 정보를 해석한다.
    /// 텐서의 값은 읽지 않는다.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::parse(Arc::new(Mmap::open(path)?))
    }

    // ANCHOR: parse
    /// 이미 메모리에 있는 바이트를 GGUF 파일로 해석한다.
    pub fn parse(owner: Owner) -> Result<Self> {
        let bytes = (*owner).as_ref();
        let mut reader = Reader::new(bytes, FORMAT);

        // 1. 머리: 식별 문자, 버전, 텐서 수, 메타데이터 항목 수.
        if reader.take(4)? != b"GGUF" {
            return Err(Error::Parse {
                format: FORMAT,
                offset: 0,
                message: "파일이 GGUF로 시작하지 않습니다".into(),
            });
        }
        let version = reader.u32()?;
        if !(2..=3).contains(&version) {
            return Err(Error::Unsupported(format!("GGUF 버전 {version}")));
        }
        let tensor_count = reader.len()?;
        let metadata_count = reader.len()?;

        // 2. 메타데이터: 키, 값의 종류, 값이 차례로 이어진다.
        let mut metadata = BTreeMap::new();
        for _ in 0..metadata_count {
            let key = reader.string()?.to_string();
            let kind = reader.u32()?;
            let value = read_value(&mut reader, kind, 0)?;
            metadata.insert(key, value);
        }

        // 3. 텐서 정보: 이름, 차원 수, 각 차원의 길이, 종류, 위치.
        let mut infos = Vec::new();
        for _ in 0..tensor_count {
            let name = reader.string()?.to_string();
            let ndim = reader.u32()? as usize;
            if ndim > MAX_DIMS {
                return Err(reader.error(format!("텐서 {name:?}의 차원 수 {ndim}가 너무 큽니다")));
            }
            let mut shape = Vec::with_capacity(ndim);
            for _ in 0..ndim {
                shape.push(reader.len()?);
            }
            // 파일에는 가장 빨리 변하는 축이 먼저 적혀 있다. 뒤집어서
            // 행 우선 순서로 만든다.
            shape.reverse();
            let ggml_type = GgmlType::from_code(reader.u32()?);
            let offset = reader.len()?;
            infos.push((name, shape, ggml_type, offset));
        }

        // 4. 값이 시작하는 곳: 지금 위치를 정렬 단위의 배수로 올림한 곳.
        let alignment = match metadata.get("general.alignment") {
            None => DEFAULT_ALIGNMENT,
            Some(value) => value
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .filter(|&n| n > 0)
                .ok_or_else(|| reader.error("general.alignment가 올바르지 않습니다"))?,
        };
        let data_start = reader.position().next_multiple_of(alignment);

        let mut tensors = BTreeMap::new();
        for (name, shape, ggml_type, relative) in infos {
            let error = |message: &str| Error::Parse {
                format: FORMAT,
                offset: data_start,
                message: format!("텐서 {name:?}: {message}"),
            };
            if relative % alignment != 0 {
                return Err(error("값의 위치가 정렬 단위의 배수가 아닙니다"));
            }
            let offset = data_start
                .checked_add(relative)
                .filter(|&offset| offset <= bytes.len())
                .ok_or_else(|| error("값의 위치가 파일을 벗어납니다"))?;
            // 크기를 아는 종류라면 값 전체가 파일 안에 있는지도 확인한다.
            if let Some(item_size) = ggml_type.size_in_bytes() {
                shape
                    .iter()
                    .try_fold(item_size, |size, &dim| size.checked_mul(dim))
                    .and_then(|size| offset.checked_add(size))
                    .filter(|&end| end <= bytes.len())
                    .ok_or_else(|| error("값이 파일을 벗어납니다"))?;
            }
            tensors.insert(
                name,
                TensorInfo {
                    shape,
                    ggml_type,
                    offset,
                },
            );
        }

        Ok(Self {
            owner,
            version,
            metadata,
            tensors,
        })
    }
    // ANCHOR_END: parse

    /// 파일 형식의 버전.
    pub fn version(&self) -> u32 {
        self.version
    }

    /// 메타데이터 전체. 키의 사전 순이다.
    pub fn metadata(&self) -> &BTreeMap<String, MetadataValue> {
        &self.metadata
    }

    /// 텐서의 이름들. 사전 순이다.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// 이름으로 텐서의 정보를 찾는다.
    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    // ANCHOR: tensor
    /// 이름으로 텐서를 꺼낸다. 파일의 메모리를 그대로 빌려 쓴다.
    pub fn tensor(&self, name: &str) -> Result<Tensor> {
        let info = self
            .info(name)
            .ok_or_else(|| Error::NotFound(name.to_string()))?;
        let dtype = match info.ggml_type {
            GgmlType::F32 => DType::F32,
            GgmlType::I32 => DType::I32,
            other => {
                return Err(Error::Unsupported(format!(
                    "텐서 {name:?}의 저장 방식 {other:?}"
                )));
            }
        };
        let len: usize = info.shape.iter().product();
        let end = info.offset + len * dtype.size_in_bytes();
        Ok(Tensor::from_bytes(
            Arc::clone(&self.owner),
            info.offset..end,
            dtype,
            &info.shape,
        )?)
    }
    // ANCHOR_END: tensor
}

// ANCHOR: read_value
/// 메타데이터 값 하나를 읽는다. `kind`는 파일에 적힌 종류 번호다.
fn read_value(reader: &mut Reader<'_>, kind: u32, depth: usize) -> Result<MetadataValue> {
    Ok(match kind {
        0 => MetadataValue::U8(reader.number()?),
        1 => MetadataValue::I8(reader.number()?),
        2 => MetadataValue::U16(reader.number()?),
        3 => MetadataValue::I16(reader.number()?),
        4 => MetadataValue::U32(reader.number()?),
        5 => MetadataValue::I32(reader.number()?),
        6 => MetadataValue::F32(reader.number()?),
        7 => match reader.u8()? {
            0 => MetadataValue::Bool(false),
            1 => MetadataValue::Bool(true),
            other => return Err(reader.error(format!("참거짓 값이 {other}입니다"))),
        },
        8 => MetadataValue::String(reader.string()?.to_string()),
        9 => {
            if depth == MAX_ARRAY_DEPTH {
                return Err(reader.error("배열이 너무 깊이 중첩되어 있습니다"));
            }
            let element_kind = reader.u32()?;
            let count = reader.len()?;
            // 파일이 주장하는 개수만큼 미리 할당하지 않는다. 원소를 읽다가
            // 파일이 끝나면 그때 오류가 난다.
            let mut items = Vec::new();
            for _ in 0..count {
                items.push(read_value(reader, element_kind, depth + 1)?);
            }
            MetadataValue::Array(items)
        }
        10 => MetadataValue::U64(reader.number()?),
        11 => MetadataValue::I64(reader.number()?),
        12 => MetadataValue::F64(reader.number()?),
        other => return Err(reader.error(format!("알 수 없는 값의 종류 {other}"))),
    })
}
// ANCHOR_END: read_value

#[cfg(test)]
mod tests {
    use super::*;

    /// 테스트용 GGUF 바이트를 조립한다.
    #[derive(Default)]
    struct Builder {
        metadata: Vec<u8>,
        metadata_count: u64,
        infos: Vec<u8>,
        tensor_count: u64,
        data: Vec<u8>,
    }

    fn string(text: &str) -> Vec<u8> {
        let mut bytes = (text.len() as u64).to_le_bytes().to_vec();
        bytes.extend(text.as_bytes());
        bytes
    }

    impl Builder {
        fn kv(mut self, key: &str, kind: u32, value: &[u8]) -> Self {
            self.metadata.extend(string(key));
            self.metadata.extend(kind.to_le_bytes());
            self.metadata.extend(value);
            self.metadata_count += 1;
            self
        }

        /// `dims`는 파일에 적히는 순서(가장 빨리 변하는 축이 먼저)다.
        fn tensor(mut self, name: &str, dims: &[u64], kind: u32, values: &[u8]) -> Self {
            self.infos.extend(string(name));
            self.infos.extend((dims.len() as u32).to_le_bytes());
            for dim in dims {
                self.infos.extend(dim.to_le_bytes());
            }
            self.infos.extend(kind.to_le_bytes());
            self.infos.extend((self.data.len() as u64).to_le_bytes());
            self.data.extend(values);
            self.data.resize(self.data.len().next_multiple_of(32), 0);
            self.tensor_count += 1;
            self
        }

        fn build(self) -> Owner {
            let mut bytes = b"GGUF".to_vec();
            bytes.extend(3_u32.to_le_bytes());
            bytes.extend(self.tensor_count.to_le_bytes());
            bytes.extend(self.metadata_count.to_le_bytes());
            bytes.extend(self.metadata);
            bytes.extend(self.infos);
            bytes.resize(bytes.len().next_multiple_of(32), 0);
            bytes.extend(self.data);
            Arc::new(bytes)
        }
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn parses_header_metadata_and_tensors() {
        let owner = Builder::default()
            .kv("general.architecture", 8, &string("test"))
            .kv("n", 4, &7_u32.to_le_bytes())
            .tensor("w", &[3, 2], 0, &f32_bytes(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
            .build();
        let gguf = Gguf::parse(owner).unwrap();
        assert_eq!(gguf.version(), 3);
        assert_eq!(
            gguf.metadata()["general.architecture"].as_str(),
            Some("test")
        );
        assert_eq!(gguf.metadata()["n"].as_u64(), Some(7));

        // 파일의 [3, 2]는 행 우선으로 (2, 3)이다.
        let info = gguf.info("w").unwrap();
        assert_eq!(info.shape, [2, 3]);
        assert_eq!(info.ggml_type, GgmlType::F32);
        let w = gguf.tensor("w").unwrap();
        assert_eq!(w.get::<f32>(&[1, 0]).unwrap(), 3.0);
    }

    #[test]
    fn parses_arrays_including_arrays_of_strings() {
        let mut array = 8_u32.to_le_bytes().to_vec();
        array.extend(2_u64.to_le_bytes());
        array.extend(string("a"));
        array.extend(string("나"));
        let gguf = Gguf::parse(Builder::default().kv("tokens", 9, &array).build()).unwrap();
        let tokens: Vec<&str> = gguf.metadata()["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|token| token.as_str().unwrap())
            .collect();
        assert_eq!(tokens, ["a", "나"]);
    }

    #[test]
    fn rejects_bad_magic_and_unknown_versions() {
        assert!(Gguf::parse(Arc::new(b"GGML\x03\0\0\0".to_vec())).is_err());
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(99_u32.to_le_bytes());
        bytes.extend([0; 16]);
        assert!(matches!(
            Gguf::parse(Arc::new(bytes)),
            Err(Error::Unsupported(_))
        ));
    }

    #[test]
    fn rejects_counts_and_lengths_that_exceed_the_file() {
        // 텐서가 2^60개 있다고 주장하는 머리.
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(3_u32.to_le_bytes());
        bytes.extend((1_u64 << 60).to_le_bytes());
        bytes.extend(0_u64.to_le_bytes());
        assert!(Gguf::parse(Arc::new(bytes)).is_err());

        // 길이가 2^62인 문자열.
        let huge = (1_u64 << 62).to_le_bytes();
        assert!(Gguf::parse(Builder::default().kv("k", 8, &huge).build()).is_err());

        // 원소가 2^40개 있다고 주장하는 배열.
        let mut array = 0_u32.to_le_bytes().to_vec();
        array.extend((1_u64 << 40).to_le_bytes());
        assert!(Gguf::parse(Builder::default().kv("k", 9, &array).build()).is_err());
    }

    #[test]
    fn rejects_tensors_that_do_not_fit_in_the_file() {
        // 형상은 원소 100개를 말하지만 값은 2개뿐이다.
        let owner = Builder::default()
            .tensor("w", &[100], 0, &f32_bytes(&[1.0, 2.0]))
            .build();
        match Gguf::parse(owner) {
            Err(error) => assert!(error.to_string().contains("벗어납니다")),
            Ok(_) => panic!("거부되어야 한다"),
        }
    }

    #[test]
    fn unsupported_tensor_types_are_listed_but_not_loaded() {
        // 12번은 양자화 형식 Q4_K다. 정보는 읽히지만 값은 아직 꺼낼 수 없다.
        let owner = Builder::default().tensor("q", &[32], 12, &[0; 32]).build();
        let gguf = Gguf::parse(owner).unwrap();
        assert_eq!(gguf.info("q").unwrap().ggml_type, GgmlType::Other(12));
        assert!(matches!(gguf.tensor("q"), Err(Error::Unsupported(_))));
    }

    #[test]
    fn honours_a_custom_alignment() {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(3_u32.to_le_bytes());
        bytes.extend(1_u64.to_le_bytes());
        bytes.extend(1_u64.to_le_bytes());
        bytes.extend(string("general.alignment"));
        bytes.extend(4_u32.to_le_bytes());
        bytes.extend(8_u32.to_le_bytes());
        bytes.extend(string("x"));
        bytes.extend(1_u32.to_le_bytes());
        bytes.extend(1_u64.to_le_bytes());
        bytes.extend(26_u32.to_le_bytes());
        bytes.extend(0_u64.to_le_bytes());
        bytes.resize(bytes.len().next_multiple_of(8), 0);
        bytes.extend(42_i32.to_le_bytes());
        let gguf = Gguf::parse(Arc::new(bytes)).unwrap();
        assert_eq!(gguf.tensor("x").unwrap().to_vec::<i32>().unwrap(), [42]);
    }
}
