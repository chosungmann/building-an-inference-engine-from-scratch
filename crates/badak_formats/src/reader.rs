//! 바이트 슬라이스를 앞에서부터 차례로 읽는 도우미.

use crate::error::{Error, Result};

// ANCHOR: reader
/// 바이트 슬라이스 위의 읽기 위치. 범위를 벗어난 읽기는 모두 오류가 된다.
pub(crate) struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
    format: &'static str,
}

impl<'a> Reader<'a> {
    pub fn new(bytes: &'a [u8], format: &'static str) -> Self {
        Self {
            bytes,
            position: 0,
            format,
        }
    }

    /// 지금까지 읽은 바이트 수.
    pub fn position(&self) -> usize {
        self.position
    }

    /// 지금 위치를 담은 오류를 만든다.
    pub fn error(&self, message: impl Into<String>) -> Error {
        Error::Parse {
            format: self.format,
            offset: self.position,
            message: message.into(),
        }
    }

    /// 다음 `len`바이트를 읽고 위치를 옮긴다.
    pub fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        // 파일에 적힌 길이를 그대로 믿지 않는다. 넘침까지 포함해 검사한다.
        let end = self
            .position
            .checked_add(len)
            .filter(|&end| end <= self.bytes.len())
            .ok_or_else(|| self.error(format!("{len}바이트를 읽으려 했지만 파일이 끝났습니다")))?;
        let bytes = &self.bytes[self.position..end];
        self.position = end;
        Ok(bytes)
    }

    /// 다음 `N`바이트를 고정 길이 배열로 읽는다.
    pub fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        Ok(self.take(N)?.try_into().expect("take가 길이를 보장한다"))
    }

    /// 리틀 엔디언으로 적힌 숫자 하나를 읽는다.
    pub fn number<T: LittleEndian<N>, const N: usize>(&mut self) -> Result<T> {
        Ok(T::from_le(self.array()?))
    }

    pub fn u8(&mut self) -> Result<u8> {
        self.number()
    }

    pub fn u32(&mut self) -> Result<u32> {
        self.number()
    }

    pub fn u64(&mut self) -> Result<u64> {
        self.number()
    }

    /// 길이가 앞에 붙은 UTF-8 문자열을 읽는다.
    pub fn string(&mut self) -> Result<&'a str> {
        let len = self.len()?;
        let start = self.position;
        let bytes = self.take(len)?;
        std::str::from_utf8(bytes).map_err(|_| Error::Parse {
            format: self.format,
            offset: start,
            message: "문자열이 UTF-8이 아닙니다".to_string(),
        })
    }

    /// 64비트 길이를 읽어 `usize`로 바꾼다.
    pub fn len(&mut self) -> Result<usize> {
        let value = self.u64()?;
        usize::try_from(value).map_err(|_| self.error(format!("길이 {value}가 너무 큽니다")))
    }
}
// ANCHOR_END: reader

/// 리틀 엔디언 바이트 `N`개에서 만들 수 있는 숫자 타입.
pub trait LittleEndian<const N: usize> {
    /// 바이트 `N`개를 숫자로 바꾼다.
    fn from_le(bytes: [u8; N]) -> Self;
}

macro_rules! impl_little_endian {
    ($($type:ty),*) => {$(
        impl LittleEndian<{ size_of::<$type>() }> for $type {
            fn from_le(bytes: [u8; size_of::<$type>()]) -> Self {
                Self::from_le_bytes(bytes)
            }
        }
    )*};
}

impl_little_endian!(u8, i8, u16, i16, u32, i32, u64, i64, f32, f64);
