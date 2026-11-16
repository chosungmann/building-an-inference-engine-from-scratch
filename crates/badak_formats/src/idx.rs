//! IDX 형식. MNIST 데이터가 담겨 있는 단순한 배열 파일이다.
//!
//! 머리는 4바이트 식별값과 차원별 길이로 이루어지고, 그 뒤로 값이 행 우선
//! 순서로 이어진다. 이 장에서 다룬 다른 형식과 달리 머리의 숫자가 빅
//! 엔디언이다.

use std::path::Path;
use std::sync::Arc;

use badak_core::{DType, Tensor};

use crate::error::{Error, Result};
use crate::mmap::Mmap;

// ANCHOR: idx
/// IDX 파일을 메모리에 비추고 `u8` 텐서로 읽는다.
pub fn open(path: impl AsRef<Path>) -> Result<Tensor> {
    parse(Arc::new(Mmap::open(path)?))
}

/// 이미 메모리에 있는 바이트를 IDX 파일로 해석한다.
pub fn parse(owner: Arc<dyn AsRef<[u8]> + Send + Sync>) -> Result<Tensor> {
    let bytes = (*owner).as_ref();
    let error = |offset: usize, message: &str| Error::Parse {
        format: "IDX",
        offset,
        message: message.to_string(),
    };

    // 식별값 4바이트: 0, 0, 자료형 번호, 차원 수.
    let Some(&[0, 0, type_code, ndim]) = bytes.first_chunk::<4>() else {
        return Err(error(0, "IDX 파일의 식별값이 아닙니다"));
    };
    if type_code != 0x08 {
        return Err(Error::Unsupported(format!(
            "IDX 자료형 번호 {type_code:#04x} (부호 없는 8비트만 지원)"
        )));
    }

    // 차원별 길이: 빅 엔디언 32비트 정수.
    let data_start = 4 + 4 * usize::from(ndim);
    let Some(header) = bytes.get(4..data_start) else {
        return Err(error(4, "차원 정보가 잘렸습니다"));
    };
    let (dims, _) = header.as_chunks::<4>();
    let shape: Vec<usize> = dims
        .iter()
        .map(|&dim| u32::from_be_bytes(dim) as usize)
        .collect();

    // 값의 길이가 형상과 맞는지는 from_bytes가 확인한다.
    Ok(Tensor::from_bytes(
        Arc::clone(&owner),
        data_start..bytes.len(),
        DType::U8,
        &shape,
    )?)
}
// ANCHOR_END: idx

#[cfg(test)]
mod tests {
    use super::*;

    fn file(header: &[u8], data: &[u8]) -> Arc<dyn AsRef<[u8]> + Send + Sync> {
        Arc::new([header, data].concat())
    }

    #[test]
    fn parses_shape_and_values() {
        // 2 × 3 배열
        let header = [0, 0, 8, 2, 0, 0, 0, 2, 0, 0, 0, 3];
        let t = parse(file(&header, &[1, 2, 3, 4, 5, 6])).unwrap();
        assert_eq!(t.shape(), [2, 3]);
        assert_eq!(t.get::<u8>(&[1, 0]).unwrap(), 4);
        assert!(t.is_borrowed());
    }

    #[test]
    fn rejects_bad_files() {
        assert!(parse(file(&[1, 2, 3, 4], &[])).is_err());
        assert!(parse(file(&[0, 0, 8, 2, 0, 0], &[])).is_err());
        // 형상은 6개를 말하는데 값은 5개다.
        let header = [0, 0, 8, 2, 0, 0, 0, 2, 0, 0, 0, 3];
        assert!(parse(file(&header, &[1, 2, 3, 4, 5])).is_err());
        // 실수형(0x0d)은 지원하지 않는다.
        assert!(matches!(
            parse(file(&[0, 0, 0x0d, 1, 0, 0, 0, 0], &[])),
            Err(Error::Unsupported(_))
        ));
    }
}
