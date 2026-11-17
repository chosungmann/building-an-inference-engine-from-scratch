//! safetensors 형식.
//!
//! 파일은 세 부분이다. 헤더의 길이를 적은 8바이트, JSON으로 된 헤더, 그리고
//! 모든 텐서의 값이 이어 붙은 바이트 덩어리. 헤더에는 텐서마다 이름,
//! 자료형, 형상, 덩어리 안에서의 구간이 적혀 있다.

use std::collections::BTreeMap;
use std::io::Write;
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use badak_core::{DType, Tensor};

use crate::error::{Error, Result};
use crate::json::{self, Value};
use crate::mmap::Mmap;

const FORMAT: &str = "safetensors";

/// 이보다 긴 헤더는 거부한다. 헤더 길이는 파일이 주장하는 값일 뿐이다.
const MAX_HEADER_LEN: usize = 100 * 1024 * 1024;

type Owner = Arc<dyn AsRef<[u8]> + Send + Sync>;

// ANCHOR: info
/// 헤더에 적힌 텐서 하나의 정보.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    /// 파일에 적힌 자료형 이름. 예: `"F32"`.
    pub dtype: String,
    /// 형상.
    pub shape: Vec<usize>,
    /// 값이 놓인 구간. 파일의 처음부터 센 바이트 위치다.
    pub range: Range<usize>,
}

/// 열어 둔 safetensors 파일.
pub struct SafeTensors {
    owner: Owner,
    tensors: BTreeMap<String, TensorInfo>,
    metadata: BTreeMap<String, String>,
}
// ANCHOR_END: info

/// 파일에 적힌 자료형 이름에서 원소 하나의 바이트 수를 구한다.
fn dtype_size(name: &str) -> Option<usize> {
    Some(match name {
        "BOOL" | "U8" | "I8" => 1,
        "I16" | "U16" | "F16" | "BF16" => 2,
        "I32" | "U32" | "F32" => 4,
        "I64" | "U64" | "F64" => 8,
        _ => return None,
    })
}

impl SafeTensors {
    /// 파일을 메모리에 비추고 헤더를 해석한다. 텐서의 값은 읽지 않는다.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::parse(Arc::new(Mmap::open(path)?))
    }

    // ANCHOR: parse
    /// 이미 메모리에 있는 바이트를 safetensors 파일로 해석한다.
    pub fn parse(owner: Owner) -> Result<Self> {
        let bytes = (*owner).as_ref();
        let error = |offset: usize, message: String| Error::Parse {
            format: FORMAT,
            offset,
            message,
        };

        // 1. 헤더의 길이: 리틀 엔디언 64비트 정수.
        let Some(len_bytes) = bytes.first_chunk::<8>() else {
            return Err(error(0, "헤더 길이를 담은 8바이트가 없습니다".into()));
        };
        let header_len = u64::from_le_bytes(*len_bytes);
        let header_len = usize::try_from(header_len)
            .ok()
            .filter(|&len| len <= MAX_HEADER_LEN && len <= bytes.len() - 8)
            .ok_or_else(|| error(0, format!("헤더 길이 {header_len}가 올바르지 않습니다")))?;

        // 2. 헤더: UTF-8로 적힌 JSON 객체.
        let data_start = 8 + header_len;
        let header = std::str::from_utf8(&bytes[8..data_start])
            .map_err(|_| error(8, "헤더가 UTF-8이 아닙니다".into()))?;
        let header = json::parse(header)?;
        let Some(entries) = header.as_object() else {
            return Err(error(8, "헤더가 JSON 객체가 아닙니다".into()));
        };

        // 3. 텐서마다 정보를 읽고, 구간이 파일 안에 있는지 확인한다.
        let buffer_len = bytes.len() - data_start;
        let mut tensors = BTreeMap::new();
        let mut metadata = BTreeMap::new();
        for (name, entry) in entries {
            if name == "__metadata__" {
                metadata = parse_metadata(entry).ok_or_else(|| {
                    error(
                        8,
                        "__metadata__는 문자열에서 문자열로 가는 객체여야 합니다".into(),
                    )
                })?;
                continue;
            }
            let info = parse_info(entry, data_start, buffer_len)
                .map_err(|message| error(8, format!("텐서 {name:?}: {message}")))?;
            tensors.insert(name.clone(), info);
        }

        // 4. 구간들이 겹치거나 빈틈을 남기지 않고 덩어리 전체를 덮는지 확인한다.
        let mut ranges: Vec<&Range<usize>> = tensors.values().map(|info| &info.range).collect();
        ranges.sort_by_key(|range| (range.start, range.end));
        let mut covered = data_start;
        for range in ranges {
            if range.start != covered {
                return Err(error(
                    covered,
                    "텐서의 구간이 겹치거나 사이에 빈틈이 있습니다".into(),
                ));
            }
            covered = range.end;
        }
        if covered != bytes.len() {
            return Err(error(
                covered,
                "어느 텐서에도 속하지 않는 바이트가 있습니다".into(),
            ));
        }

        Ok(Self {
            owner,
            tensors,
            metadata,
        })
    }
    // ANCHOR_END: parse

    /// 텐서의 이름들. 사전 순이다.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// 이름으로 텐서의 정보를 찾는다.
    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// 파일에 붙어 있는 자유 형식의 메타데이터.
    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    // ANCHOR: tensor
    /// 이름으로 텐서를 꺼낸다.
    ///
    /// 값이 정렬에 맞게 놓여 있으면 파일의 메모리를 그대로 빌려 쓰는
    /// 텐서가 된다. 복사가 일어나지 않는다.
    pub fn tensor(&self, name: &str) -> Result<Tensor> {
        let info = self
            .info(name)
            .ok_or_else(|| Error::NotFound(name.to_string()))?;
        let dtype = match info.dtype.as_str() {
            "F32" => DType::F32,
            "I32" => DType::I32,
            "U8" => DType::U8,
            "BF16" => return self.widen_bf16(info),
            other => {
                return Err(Error::Unsupported(format!(
                    "텐서 {name:?}의 자료형 {other}"
                )));
            }
        };
        Ok(Tensor::from_bytes(
            Arc::clone(&self.owner),
            info.range.clone(),
            dtype,
            &info.shape,
        )?)
    }
    // ANCHOR_END: tensor

    // ANCHOR: tensor_as_stored
    /// 이름으로 텐서를 꺼내되, 16비트 실수를 `f32`로 펼치지 않고 저장된
    /// 그대로 돌려준다 (19장).
    ///
    /// BF16과 F16 텐서가 파일의 메모리를 빌려 쓰는 16비트 텐서로 나온다.
    /// 복사가 없고 메모리도 절반이지만, 5장의 연산자는 이 자료형으로
    /// 계산하지 못한다. 그 밖의 자료형은 [`tensor`](Self::tensor)와 같다.
    pub fn tensor_as_stored(&self, name: &str) -> Result<Tensor> {
        let info = self
            .info(name)
            .ok_or_else(|| Error::NotFound(name.to_string()))?;
        let dtype = match info.dtype.as_str() {
            "BF16" => DType::Bf16,
            "F16" => DType::F16,
            _ => return self.tensor(name),
        };
        Ok(Tensor::from_bytes(
            Arc::clone(&self.owner),
            info.range.clone(),
            dtype,
            &info.shape,
        )?)
    }
    // ANCHOR_END: tensor_as_stored

    // ANCHOR: bf16
    /// BF16으로 저장된 텐서를 `f32`로 펼쳐 읽는다.
    ///
    /// BF16은 `f32`의 32비트 가운데 위쪽 16비트만 남긴 형식이다. 부호와
    /// 지수는 그대로이고 가수만 7비트로 줄어 있다. 그래서 아래쪽 16비트를
    /// 0으로 채우면 같은 값을 나타내는 `f32`가 된다. 값은 정확히 보존되지만
    /// 복사가 일어나고 메모리는 두 배가 든다.
    fn widen_bf16(&self, info: &TensorInfo) -> Result<Tensor> {
        let bytes = &(*self.owner).as_ref()[info.range.clone()];
        let (pairs, _) = bytes.as_chunks::<2>();
        let values = pairs
            .iter()
            .map(|&pair| f32::from_bits(u32::from(u16::from_le_bytes(pair)) << 16))
            .collect();
        Ok(Tensor::from_vec(values, &info.shape)?)
    }
    // ANCHOR_END: bf16
}

// ANCHOR: parse_info
/// 헤더의 항목 하나를 해석하고 검증한다.
fn parse_info(
    entry: &Value,
    data_start: usize,
    buffer_len: usize,
) -> std::result::Result<TensorInfo, String> {
    let dtype = entry
        .get("dtype")
        .and_then(Value::as_str)
        .ok_or("dtype이 없습니다")?;
    let item_size = dtype_size(dtype).ok_or_else(|| format!("알 수 없는 자료형 {dtype}"))?;

    let shape = entry
        .get("shape")
        .and_then(Value::as_array)
        .and_then(|dims| dims.iter().map(Value::as_usize).collect::<Option<Vec<_>>>())
        .ok_or("shape가 음이 아닌 정수의 배열이 아닙니다")?;

    let offsets = entry
        .get("data_offsets")
        .and_then(Value::as_array)
        .and_then(|pair| pair.iter().map(Value::as_usize).collect::<Option<Vec<_>>>())
        .ok_or("data_offsets가 올바르지 않습니다")?;
    let &[begin, end] = offsets.as_slice() else {
        return Err("data_offsets는 숫자 두 개여야 합니다".into());
    };
    if begin > end || end > buffer_len {
        return Err(format!("구간 {begin}..{end}가 파일을 벗어납니다"));
    }

    // 형상이 말하는 크기와 구간의 길이가 같아야 한다. 곱셈의 넘침도 확인한다.
    let expected = shape
        .iter()
        .try_fold(item_size, |size, &dim| size.checked_mul(dim))
        .ok_or("형상이 너무 큽니다")?;
    if end - begin != expected {
        return Err(format!(
            "형상 {shape:?}에는 {expected}바이트가 필요한데 구간은 {}바이트입니다",
            end - begin
        ));
    }

    Ok(TensorInfo {
        dtype: dtype.to_string(),
        shape,
        range: data_start + begin..data_start + end,
    })
}
// ANCHOR_END: parse_info

fn parse_metadata(entry: &Value) -> Option<BTreeMap<String, String>> {
    entry
        .as_object()?
        .iter()
        .map(|(key, value)| Some((key.clone(), value.as_str()?.to_string())))
        .collect()
}

// ANCHOR: write
/// 텐서들을 safetensors 파일로 쓴다.
///
/// 원소가 큰 자료형의 텐서를 앞에 두고 헤더를 8바이트 단위로 채워, 모든
/// 텐서의 값이 자기 자료형의 정렬에 맞는 위치에 놓이게 한다.
pub fn write(
    path: impl AsRef<Path>,
    tensors: &[(&str, &Tensor)],
    metadata: &[(&str, &str)],
) -> Result<()> {
    let mut ordered: Vec<&(&str, &Tensor)> = tensors.iter().collect();
    ordered
        .sort_by_key(|(name, tensor)| (std::cmp::Reverse(tensor.dtype().size_in_bytes()), *name));

    let mut entries = Vec::new();
    if !metadata.is_empty() {
        let members: Vec<String> = metadata
            .iter()
            .map(|(key, value)| format!("{}:{}", json::quote(key), json::quote(value)))
            .collect();
        entries.push(format!("\"__metadata__\":{{{}}}", members.join(",")));
    }

    let mut buffer = Vec::new();
    for (name, tensor) in ordered {
        let begin = buffer.len();
        let tensor = tensor.contiguous();
        let dtype = match tensor.dtype() {
            DType::F32 => {
                buffer.extend(
                    tensor
                        .as_slice::<f32>()?
                        .iter()
                        .flat_map(|v| v.to_le_bytes()),
                );
                "F32"
            }
            DType::I32 => {
                buffer.extend(
                    tensor
                        .as_slice::<i32>()?
                        .iter()
                        .flat_map(|v| v.to_le_bytes()),
                );
                "I32"
            }
            DType::U8 => {
                buffer.extend(tensor.as_slice::<u8>()?);
                "U8"
            }
            DType::F16 => {
                let values = tensor.as_slice::<badak_core::quant::F16>()?;
                buffer.extend(values.iter().flat_map(|v| v.to_le_bytes()));
                "F16"
            }
            DType::Bf16 => {
                let values = tensor.as_slice::<badak_core::quant::Bf16>()?;
                buffer.extend(values.iter().flat_map(|v| v.to_le_bytes()));
                "BF16"
            }
        };
        entries.push(format!(
            "{}:{{\"dtype\":\"{dtype}\",\"shape\":{:?},\"data_offsets\":[{begin},{}]}}",
            json::quote(name),
            tensor.shape(),
            buffer.len()
        ));
    }

    let mut header = format!("{{{}}}", entries.join(","));
    // 헤더 뒤에 공백을 붙여 값이 8의 배수 위치에서 시작하게 한다.
    header.push_str(&" ".repeat(header.len().next_multiple_of(8) - header.len()));

    let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
    file.write_all(&(header.len() as u64).to_le_bytes())?;
    file.write_all(header.as_bytes())?;
    file.write_all(&buffer)?;
    file.flush()?;
    Ok(())
}
// ANCHOR_END: write

#[cfg(test)]
mod tests {
    use super::*;

    /// 헤더 문자열과 값 바이트로 파일 내용을 만든다.
    fn file(header: &str, data: &[u8]) -> Owner {
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header.as_bytes());
        bytes.extend(data);
        Arc::new(bytes)
    }

    fn parse_error(header: &str, data: &[u8]) -> String {
        match SafeTensors::parse(file(header, data)) {
            Err(error) => error.to_string(),
            Ok(_) => panic!("거부되어야 하는 파일이 통과했다"),
        }
    }

    #[test]
    fn parses_a_minimal_file() {
        let header = r#"{"x":{"dtype":"U8","shape":[2,2],"data_offsets":[0,4]}}"#;
        let st = SafeTensors::parse(file(header, &[1, 2, 3, 4])).unwrap();
        assert_eq!(st.names().collect::<Vec<_>>(), ["x"]);
        let x = st.tensor("x").unwrap();
        assert_eq!(x.shape(), [2, 2]);
        assert_eq!(x.to_vec::<u8>().unwrap(), [1, 2, 3, 4]);
        assert!(matches!(st.tensor("y"), Err(Error::NotFound(_))));
    }

    #[test]
    fn rejects_truncated_and_oversized_headers() {
        assert!(SafeTensors::parse(Arc::new(vec![1_u8, 2, 3])).is_err());
        // 헤더 길이가 파일보다 길다고 주장한다.
        let mut bytes = u64::MAX.to_le_bytes().to_vec();
        bytes.extend(b"{}");
        assert!(SafeTensors::parse(Arc::new(bytes)).is_err());
    }

    #[test]
    fn rejects_ranges_outside_the_file() {
        let header = r#"{"x":{"dtype":"U8","shape":[8],"data_offsets":[0,8]}}"#;
        assert!(parse_error(header, &[0; 4]).contains("벗어납니다"));
    }

    #[test]
    fn rejects_a_shape_that_disagrees_with_the_range() {
        let header = r#"{"x":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}}"#;
        assert!(parse_error(header, &[0; 8]).contains("12바이트가 필요"));
    }

    #[test]
    fn rejects_a_shape_whose_size_overflows() {
        let header = r#"{"x":{"dtype":"F64","shape":[4294967296,4294967296,4294967296],"data_offsets":[0,0]}}"#;
        assert!(parse_error(header, &[]).contains("너무 큽니다"));
    }

    #[test]
    fn rejects_overlaps_and_holes() {
        let overlap = r#"{"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},"b":{"dtype":"U8","shape":[4],"data_offsets":[2,6]}}"#;
        assert!(parse_error(overlap, &[0; 6]).contains("겹치거나"));
        let hole = r#"{"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"b":{"dtype":"U8","shape":[2],"data_offsets":[4,6]}}"#;
        assert!(parse_error(hole, &[0; 6]).contains("겹치거나"));
        let trailing = r#"{"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}"#;
        assert!(parse_error(trailing, &[0; 6]).contains("속하지 않는"));
    }

    #[test]
    fn rejects_unknown_dtypes_and_bad_metadata() {
        let header = r#"{"x":{"dtype":"F128","shape":[1],"data_offsets":[0,16]}}"#;
        assert!(parse_error(header, &[0; 16]).contains("알 수 없는 자료형"));
        assert!(parse_error(r#"{"__metadata__":{"a":1}}"#, &[]).contains("__metadata__"));
        assert!(parse_error("[]", &[]).contains("객체가 아닙니다"));
    }

    #[test]
    fn written_files_read_back_identically() {
        let weight = Tensor::from_vec(vec![1.5_f32, -2.0, 0.25, 8.0], &[2, 2]).unwrap();
        let ids = Tensor::from_vec(vec![7_i32, -1, 0], &[3]).unwrap();
        let pixels = Tensor::from_vec(vec![0_u8, 128, 255], &[3]).unwrap();
        // 연속이 아닌 뷰도 쓸 수 있어야 한다.
        let transposed = weight.transpose(0, 1).unwrap();

        let path =
            std::env::temp_dir().join(format!("badak_st_{}.safetensors", std::process::id()));
        write(
            &path,
            &[
                ("pixels", &pixels),
                ("w", &weight),
                ("ids", &ids),
                ("wt", &transposed),
            ],
            &[("note", "따옴표 \" 포함")],
        )
        .unwrap();

        let st = SafeTensors::open(&path).unwrap();
        assert_eq!(st.names().collect::<Vec<_>>(), ["ids", "pixels", "w", "wt"]);
        assert_eq!(st.metadata()["note"], "따옴표 \" 포함");
        assert_eq!(
            st.tensor("w").unwrap().to_vec::<f32>().unwrap(),
            [1.5, -2.0, 0.25, 8.0]
        );
        assert_eq!(
            st.tensor("wt").unwrap().to_vec::<f32>().unwrap(),
            [1.5, 0.25, -2.0, 8.0]
        );
        assert_eq!(
            st.tensor("ids").unwrap().to_vec::<i32>().unwrap(),
            [7, -1, 0]
        );
        assert_eq!(
            st.tensor("pixels").unwrap().to_vec::<u8>().unwrap(),
            [0, 128, 255]
        );
        // 우리가 쓴 파일은 모든 텐서가 정렬에 맞아 복사 없이 읽힌다.
        for name in ["w", "wt", "ids", "pixels"] {
            assert!(st.tensor(name).unwrap().is_borrowed(), "{name}");
        }
        drop(st);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn sixteen_bit_tensors_can_be_read_as_stored_and_written_back() {
        use badak_core::quant::{Bf16, F16};
        let header = r#"{"b":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"h":{"dtype":"F16","shape":[2],"data_offsets":[4,8]}}"#;
        // BF16: 1.0, -2.5. F16: 1.0, -2.5.
        let data = [0x80, 0x3f, 0x20, 0xc0, 0x00, 0x3c, 0x00, 0xc1];
        let parsed = SafeTensors::parse(file(header, &data)).unwrap();
        let b = parsed.tensor_as_stored("b").unwrap();
        let h = parsed.tensor_as_stored("h").unwrap();
        assert_eq!((b.dtype(), h.dtype()), (DType::Bf16, DType::F16));
        assert_eq!(b.size_in_bytes(), 4);
        let b_values: Vec<f32> = b.iter::<Bf16>().unwrap().map(Bf16::to_f32).collect();
        let h_values: Vec<f32> = h.iter::<F16>().unwrap().map(F16::to_f32).collect();
        assert_eq!(b_values, [1.0, -2.5]);
        assert_eq!(h_values, [1.0, -2.5]);
        // 펼쳐 읽은 것과 값이 같다.
        assert_eq!(
            parsed.tensor("b").unwrap().to_vec::<f32>().unwrap(),
            b_values
        );

        // 16비트인 채로 다시 적을 수 있다.
        let path =
            std::env::temp_dir().join(format!("badak_half_{}.safetensors", std::process::id()));
        write(&path, &[("b", &b), ("h", &h)], &[]).unwrap();
        let again = SafeTensors::open(&path).unwrap();
        assert_eq!(again.info("b").unwrap().dtype, "BF16");
        assert_eq!(
            again
                .tensor_as_stored("h")
                .unwrap()
                .to_vec::<F16>()
                .unwrap(),
            h.to_vec::<F16>().unwrap()
        );
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn bf16_tensors_are_widened_to_f32() {
        // 1.0 = 0x3f80, -2.5 = 0xc020, 0.1에 가장 가까운 BF16 = 0x3dcd
        let header = r#"{"x":{"dtype":"BF16","shape":[3],"data_offsets":[0,6]}}"#;
        let data = [0x80, 0x3f, 0x20, 0xc0, 0xcd, 0x3d];
        let x = SafeTensors::parse(file(header, &data))
            .unwrap()
            .tensor("x")
            .unwrap();
        assert_eq!(x.dtype(), DType::F32);
        let values = x.to_vec::<f32>().unwrap();
        assert_eq!(values[..2], [1.0, -2.5]);
        // 가수가 7비트뿐이므로 0.1은 0.1003...이 된다.
        assert_eq!(values[2], f32::from_bits(0x3dcd_0000));
        assert!((values[2] - 0.1).abs() < 1e-3);
    }
}
