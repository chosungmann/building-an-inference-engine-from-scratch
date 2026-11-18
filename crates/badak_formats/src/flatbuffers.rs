//! FlatBuffers로 적힌 바이트를 읽는다 (26장).
//!
//! FlatBuffers는 파일을 풀어서 새 자료구조로 옮기지 않고, 파일의 바이트를
//! 그 자리에서 읽도록 설계한 형식이다. 값이 있는 곳을 가리키는 것은 모두
//! 위치의 차이(오프셋)이고, 이 모듈이 하는 일은 그 오프셋을 따라가는 것이
//! 전부다. 파일을 믿지 않으므로 따라가기 전에 매번 범위를 확인한다.

use std::ops::Range;

use crate::error::{Error, Result};
use crate::reader::LittleEndian;

// ANCHOR: read
/// `position`에서 리틀 엔디언 숫자 하나를 읽는다.
fn read<T: LittleEndian<N>, const N: usize>(
    bytes: &[u8],
    position: usize,
    format: &'static str,
) -> Result<T> {
    position
        .checked_add(N)
        .and_then(|end| bytes.get(position..end))
        .map(|slice| T::from_le(slice.try_into().expect("길이가 N이다")))
        .ok_or_else(|| Error::Parse {
            format,
            offset: position,
            message: format!("{N}바이트를 읽으려 했지만 파일을 벗어납니다"),
        })
}

/// `position`에 적힌 오프셋을 따라간다. 오프셋은 자기가 적힌 자리에서 잰
/// 거리이고, 부호가 없으므로 언제나 뒤쪽을 가리킨다.
fn follow(bytes: &[u8], position: usize, format: &'static str) -> Result<usize> {
    let distance: u32 = read(bytes, position, format)?;
    position
        .checked_add(distance as usize)
        .filter(|&target| target < bytes.len())
        .ok_or_else(|| Error::Parse {
            format,
            offset: position,
            message: format!("오프셋 {distance}가 파일을 벗어납니다"),
        })
}
// ANCHOR_END: read

// ANCHOR: table
/// 표(table) 하나. 필드가 어디 있는지를 vtable에서 찾는다.
#[derive(Debug, Clone, Copy)]
pub struct Table<'a> {
    bytes: &'a [u8],
    /// 표가 시작하는 위치.
    position: usize,
    /// vtable이 시작하는 위치.
    vtable: usize,
    /// vtable의 길이(바이트).
    vtable_len: usize,
    format: &'static str,
}

impl<'a> Table<'a> {
    /// 파일의 뿌리 표를 연다. `identifier`를 주면 4~8번째 바이트와 견준다.
    pub fn root(
        bytes: &'a [u8],
        identifier: Option<&[u8; 4]>,
        format: &'static str,
    ) -> Result<Self> {
        if let Some(expected) = identifier
            && bytes.get(4..8) != Some(expected)
        {
            return Err(Error::Parse {
                format,
                offset: 4,
                message: format!(
                    "파일 식별자가 {:?}가 아닙니다",
                    String::from_utf8_lossy(expected)
                ),
            });
        }
        Self::at(bytes, follow(bytes, 0, format)?, format)
    }

    /// `position`에서 시작하는 표를 연다.
    fn at(bytes: &'a [u8], position: usize, format: &'static str) -> Result<Self> {
        let error = |message: &str| Error::Parse {
            format,
            offset: position,
            message: message.to_string(),
        };
        // 표의 첫 값은 vtable까지의 거리다. 부호가 있고, 표의 위치에서 뺀다.
        let distance: i32 = read(bytes, position, format)?;
        let vtable = (position as i64)
            .checked_sub(i64::from(distance))
            .and_then(|vtable| usize::try_from(vtable).ok())
            .ok_or_else(|| error("vtable의 위치가 파일을 벗어납니다"))?;
        // vtable의 첫 값은 vtable 자신의 길이다.
        let vtable_len = usize::from(read::<u16, 2>(bytes, vtable, format)?);
        let fits = vtable
            .checked_add(vtable_len)
            .is_some_and(|end| end <= bytes.len());
        if vtable_len < 4 || !vtable_len.is_multiple_of(2) || !fits {
            return Err(error("vtable의 길이가 올바르지 않습니다"));
        }
        Ok(Self {
            bytes,
            position,
            vtable,
            vtable_len,
            format,
        })
    }

    /// `slot`번째 필드가 적힌 위치. 필드가 없으면 `None`이다.
    ///
    /// 스키마에 필드가 나중에 추가되었다면 옛 파일의 vtable은 그 칸까지
    /// 오지 않는다. 그때도 "없음"이다. 이것이 FlatBuffers가 앞뒤 버전을
    /// 함께 읽는 방법이다.
    fn field(&self, slot: usize) -> Result<Option<usize>> {
        // vtable의 앞 두 칸은 vtable의 길이와 표의 길이다.
        let entry = 4 + 2 * slot;
        if entry + 2 > self.vtable_len {
            return Ok(None);
        }
        let distance: u16 = read(self.bytes, self.vtable + entry, self.format)?;
        if distance == 0 {
            return Ok(None);
        }
        Ok(Some(self.position + usize::from(distance)))
    }

    /// 숫자 필드. 없으면 `default`다.
    pub fn scalar<T: LittleEndian<N>, const N: usize>(&self, slot: usize, default: T) -> Result<T> {
        match self.field(slot)? {
            Some(position) => read(self.bytes, position, self.format),
            None => Ok(default),
        }
    }

    /// 다른 표를 가리키는 필드.
    pub fn table(&self, slot: usize) -> Result<Option<Table<'a>>> {
        let Some(position) = self.field(slot)? else {
            return Ok(None);
        };
        let target = follow(self.bytes, position, self.format)?;
        Table::at(self.bytes, target, self.format).map(Some)
    }

    /// 벡터를 가리키는 필드. 원소 하나의 크기가 `width`바이트다.
    pub fn vector(&self, slot: usize, width: usize) -> Result<Option<Vector<'a>>> {
        let Some(position) = self.field(slot)? else {
            return Ok(None);
        };
        let target = follow(self.bytes, position, self.format)?;
        Vector::at(self.bytes, target, width, self.format).map(Some)
    }

    /// 문자열 필드. 문자열은 바이트의 벡터다.
    pub fn string(&self, slot: usize) -> Result<Option<&'a str>> {
        let Some(vector) = self.vector(slot, 1)? else {
            return Ok(None);
        };
        let range = vector.bytes();
        std::str::from_utf8(&self.bytes[range.clone()])
            .map(Some)
            .map_err(|_| Error::Parse {
                format: self.format,
                offset: range.start,
                message: "문자열이 UTF-8이 아닙니다".to_string(),
            })
    }
}
// ANCHOR_END: table

// ANCHOR: vector
/// 벡터 하나. 원소의 수가 앞에 적혀 있고 원소들이 이어진다.
#[derive(Debug, Clone, Copy)]
pub struct Vector<'a> {
    bytes: &'a [u8],
    /// 첫 원소의 위치.
    start: usize,
    len: usize,
    width: usize,
    format: &'static str,
}

impl<'a> Vector<'a> {
    fn at(bytes: &'a [u8], position: usize, width: usize, format: &'static str) -> Result<Self> {
        let len = read::<u32, 4>(bytes, position, format)? as usize;
        let start = position + 4;
        // 원소 수는 파일에 적힌 값이다. 원소가 모두 파일 안에 있는지 본다.
        let fits = len
            .checked_mul(width)
            .and_then(|size| start.checked_add(size))
            .is_some_and(|end| end <= bytes.len());
        if !fits {
            return Err(Error::Parse {
                format,
                offset: position,
                message: format!("원소 {len}개인 벡터가 파일을 벗어납니다"),
            });
        }
        Ok(Self {
            bytes,
            start,
            len,
            width,
            format,
        })
    }

    /// 원소의 수.
    pub fn len(&self) -> usize {
        self.len
    }

    /// 비어 있는지.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// 원소들이 차지하는 파일 안의 구간.
    pub fn bytes(&self) -> Range<usize> {
        self.start..self.start + self.len * self.width
    }

    /// 숫자의 벡터를 모두 읽는다.
    pub fn scalars<T: LittleEndian<N>, const N: usize>(&self) -> Result<Vec<T>> {
        debug_assert_eq!(self.width, N);
        (0..self.len)
            .map(|index| read(self.bytes, self.start + index * N, self.format))
            .collect()
    }

    /// 표의 벡터에서 `index`번째 표. 원소는 표를 가리키는 오프셋이다.
    pub fn table(&self, index: usize) -> Result<Table<'a>> {
        debug_assert_eq!(self.width, 4);
        if index >= self.len {
            return Err(Error::Parse {
                format: self.format,
                offset: self.start,
                message: format!("원소가 {}개인 벡터의 {index}번째를 찾았습니다", self.len),
            });
        }
        let position = self.start + index * 4;
        Table::at(
            self.bytes,
            follow(self.bytes, position, self.format)?,
            self.format,
        )
    }

    /// 표의 벡터를 차례로 돈다.
    pub fn tables(&self) -> impl Iterator<Item = Result<Table<'a>>> + '_ {
        (0..self.len).map(|index| self.table(index))
    }
}
// ANCHOR_END: vector

/// 시험에 쓸 FlatBuffers 바이트를 만든다. 읽는 쪽을 시험하려면 쓰는 쪽이
/// 있어야 한다.
#[cfg(test)]
pub(crate) mod build {
    /// 표의 필드나 벡터의 원소가 될 수 있는 값.
    #[derive(Debug, Clone)]
    pub enum Value {
        /// 필드가 없다.
        Absent,
        I8(i8),
        U8(u8),
        Bool(bool),
        I32(i32),
        U32(u32),
        F32(f32),
        Str(String),
        Bytes(Vec<u8>),
        Ints(Vec<i32>),
        Table(Vec<Value>),
        Tables(Vec<Vec<Value>>),
    }

    fn align(out: &mut Vec<u8>, to: usize) {
        while !out.len().is_multiple_of(to) {
            out.push(0);
        }
    }

    /// 표를 적고 그 위치를 돌려준다. vtable, 표, 표가 가리키는 것들의
    /// 순서로 적는다. 가리키는 것이 언제나 뒤에 있으므로 오프셋이 양수다.
    fn table(out: &mut Vec<u8>, fields: &[Value]) -> usize {
        align(out, 4);
        let inline = |value: &Value| match value {
            Value::Absent => 0,
            Value::I8(_) | Value::U8(_) | Value::Bool(_) => 1,
            _ => 4,
        };
        // 표 안에서 필드가 놓일 자리를 정한다. 4바이트짜리를 먼저 둔다.
        let mut offsets = vec![0_usize; fields.len()];
        let mut size = 4;
        for wide in [true, false] {
            for (index, field) in fields.iter().enumerate() {
                let width = inline(field);
                if width != 0 && (width == 4) == wide {
                    offsets[index] = size;
                    size += width;
                }
            }
        }
        let vtable = out.len();
        let vtable_len = 4 + 2 * fields.len();
        out.extend((vtable_len as u16).to_le_bytes());
        out.extend((size as u16).to_le_bytes());
        for &offset in &offsets {
            out.extend((offset as u16).to_le_bytes());
        }
        align(out, 4);
        let position = out.len();
        out.extend(((position - vtable) as i32).to_le_bytes());
        out.resize(position + size, 0);
        let mut pending = Vec::new();
        for (field, &offset) in fields.iter().zip(&offsets) {
            let at = position + offset;
            match field {
                Value::Absent => {}
                Value::I8(value) => out[at] = *value as u8,
                Value::U8(value) => out[at] = *value,
                Value::Bool(value) => out[at] = u8::from(*value),
                Value::I32(value) => out[at..at + 4].copy_from_slice(&value.to_le_bytes()),
                Value::U32(value) => out[at..at + 4].copy_from_slice(&value.to_le_bytes()),
                Value::F32(value) => out[at..at + 4].copy_from_slice(&value.to_le_bytes()),
                _ => pending.push((at, field)),
            }
        }
        for (at, field) in pending {
            let target = reference(out, field);
            out[at..at + 4].copy_from_slice(&((target - at) as u32).to_le_bytes());
        }
        position
    }

    /// 표 밖에 따로 놓이는 값을 적고 그 위치를 돌려준다.
    fn reference(out: &mut Vec<u8>, value: &Value) -> usize {
        match value {
            Value::Table(fields) => table(out, fields),
            Value::Str(text) => bytes(out, text.as_bytes(), 4, true),
            Value::Bytes(data) => bytes(out, data, 16, false),
            Value::Ints(values) => {
                let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                align(out, 4);
                let position = out.len();
                out.extend((values.len() as u32).to_le_bytes());
                out.extend(data);
                position
            }
            Value::Tables(tables) => {
                align(out, 4);
                let position = out.len();
                out.extend((tables.len() as u32).to_le_bytes());
                out.resize(position + 4 + 4 * tables.len(), 0);
                for (index, fields) in tables.iter().enumerate() {
                    let at = position + 4 + 4 * index;
                    let target = table(out, fields);
                    out[at..at + 4].copy_from_slice(&((target - at) as u32).to_le_bytes());
                }
                position
            }
            other => unreachable!("표 안에 적는 값이다: {other:?}"),
        }
    }

    /// 바이트의 벡터를 적는다. 원소들이 `to`의 배수인 위치에서 시작한다.
    fn bytes(out: &mut Vec<u8>, data: &[u8], to: usize, terminated: bool) -> usize {
        while !(out.len() + 4).is_multiple_of(to) {
            out.push(0);
        }
        let position = out.len();
        out.extend((data.len() as u32).to_le_bytes());
        out.extend(data);
        if terminated {
            out.push(0);
        }
        position
    }

    /// 뿌리 표와 식별자를 가진 파일 하나를 만든다.
    pub fn file(identifier: &[u8; 4], root: &[Value]) -> Vec<u8> {
        let mut out = vec![0_u8; 4];
        out.extend(identifier);
        let position = table(&mut out, root);
        out[0..4].copy_from_slice(&(position as u32).to_le_bytes());
        out
    }
}

#[cfg(test)]
mod tests {
    use super::build::{Value, file};
    use super::*;

    fn sample() -> Vec<u8> {
        file(
            b"TEST",
            &[
                Value::U32(7),
                Value::Absent,
                Value::Str("바닥".into()),
                Value::Ints(vec![3, -1, 4]),
                Value::Tables(vec![
                    vec![Value::I8(-5), Value::F32(1.5)],
                    vec![Value::Absent, Value::F32(2.5)],
                ]),
                Value::Bytes(vec![1, 2, 3, 4, 5]),
                Value::Table(vec![Value::Bool(true)]),
            ],
        )
    }

    #[test]
    fn reads_every_kind_of_field() {
        let bytes = sample();
        let root = Table::root(&bytes, Some(b"TEST"), "시험").unwrap();
        assert_eq!(root.scalar::<u32, 4>(0, 0).unwrap(), 7);
        // 없는 필드는 기본값이다.
        assert_eq!(root.scalar::<i32, 4>(1, 42).unwrap(), 42);
        assert_eq!(root.string(2).unwrap(), Some("바닥"));
        let ints = root.vector(3, 4).unwrap().unwrap();
        assert_eq!(ints.scalars::<i32, 4>().unwrap(), [3, -1, 4]);
        let tables = root.vector(4, 4).unwrap().unwrap();
        assert_eq!(tables.len(), 2);
        let first = tables.table(0).unwrap();
        assert_eq!(first.scalar::<i8, 1>(0, 0).unwrap(), -5);
        assert_eq!(first.scalar::<f32, 4>(1, 0.0).unwrap(), 1.5);
        let second = tables.table(1).unwrap();
        assert_eq!(second.scalar::<i8, 1>(0, 9).unwrap(), 9);
        assert!(tables.table(2).is_err());
        let data = root.vector(5, 1).unwrap().unwrap();
        assert_eq!(&bytes[data.bytes()], [1, 2, 3, 4, 5]);
        assert_eq!(data.bytes().start % 16, 0);
        let nested = root.table(6).unwrap().unwrap();
        assert!(nested.scalar::<u8, 1>(0, 0).unwrap() == 1);
        // vtable보다 뒤의 칸은 옛 파일에 없는 필드다.
        assert_eq!(root.scalar::<u32, 4>(40, 11).unwrap(), 11);
        assert!(root.table(40).unwrap().is_none());
    }

    #[test]
    fn rejects_wrong_identifier() {
        let bytes = sample();
        assert!(Table::root(&bytes, Some(b"TFL3"), "시험").is_err());
        assert!(Table::root(&bytes, None, "시험").is_ok());
    }

    /// 파일의 모든 필드를 읽어 본다. 오류는 괜찮고 패닉은 안 된다.
    fn walk(bytes: &[u8]) -> Result<()> {
        let root = Table::root(bytes, Some(b"TEST"), "시험")?;
        root.scalar::<u32, 4>(0, 0)?;
        root.string(2)?;
        if let Some(ints) = root.vector(3, 4)? {
            ints.scalars::<i32, 4>()?;
        }
        if let Some(tables) = root.vector(4, 4)? {
            for table in tables.tables() {
                let table = table?;
                table.scalar::<i8, 1>(0, 0)?;
                table.scalar::<f32, 4>(1, 0.0)?;
            }
        }
        root.vector(5, 1)?;
        if let Some(nested) = root.table(6)? {
            nested.scalar::<u8, 1>(0, 0)?;
        }
        Ok(())
    }

    #[test]
    fn truncated_files_are_errors_not_panics() {
        let bytes = sample();
        walk(&bytes).unwrap();
        for len in 0..bytes.len() {
            // 잘린 파일은 읽히기도 하고(끝의 채움만 잘렸을 때) 오류가 되기도
            // 한다. 어느 쪽이든 패닉은 없어야 한다.
            let _ = walk(&bytes[..len]);
        }
    }

    #[test]
    fn corrupted_bytes_are_errors_not_panics() {
        let bytes = sample();
        for position in 0..bytes.len() {
            for value in [0x00, 0x01, 0x7f, 0x80, 0xff] {
                let mut broken = bytes.clone();
                broken[position] = value;
                let _ = walk(&broken);
            }
        }
    }
}
