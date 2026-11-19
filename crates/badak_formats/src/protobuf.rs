//! Protocol Buffers의 바이트 형식을 읽는다 (30장).
//!
//! ONNX 파일은 Protocol Buffers 메시지 하나다. 메시지는 필드들을 이어 붙인
//! 것이고, 필드마다 앞에 **꼬리표**(tag)가 붙는다. 꼬리표는 필드의 번호와,
//! 값을 어떻게 적었는지(wire type)를 한 숫자에 담은 것이다. FlatBuffers
//! (26장)와 달리 필드의 위치를 적어 둔 표가 없다. 원하는 필드를 찾으려면
//! 앞에서부터 차례로 읽어야 한다.
//!
//! 필드의 번호가 무엇을 뜻하는지는 바이트에 적혀 있지 않다. 읽는 쪽이
//! `.proto` 파일을 보고 알아야 한다. 여기서는 번호와 값만 내어 주고, 뜻은
//! 부르는 쪽([`onnx`](crate::onnx))이 붙인다.

use crate::error::{Error, Result};

// ANCHOR: field
/// 필드 하나의 값. 값을 적는 방법은 네 가지다.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Value<'a> {
    /// 길이가 변하는 정수(varint). 정수, 불, 열거형이 이렇게 적힌다.
    Varint(u64),
    /// 8바이트. `double`과 `fixed64`.
    Fixed64([u8; 8]),
    /// 길이가 앞에 붙은 바이트들. 문자열, 바이트열, 안에 든 메시지, 그리고
    /// 숫자 여러 개를 이어 붙인 것(packed)이 모두 이 모양이다.
    Bytes(&'a [u8]),
    /// 4바이트. `float`와 `fixed32`.
    Fixed32([u8; 4]),
}

/// 필드 하나.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Field<'a> {
    /// 필드의 번호.
    pub number: u32,
    /// 값.
    pub value: Value<'a>,
    /// 값이 `Bytes`일 때, 그 바이트들이 가장 바깥 메시지에서 시작하는 위치.
    pub offset: usize,
}
// ANCHOR_END: field

/// 메시지 하나를 앞에서부터 읽는 반복자.
#[derive(Debug, Clone)]
pub struct Message<'a> {
    bytes: &'a [u8],
    position: usize,
    /// 이 메시지가 가장 바깥 메시지에서 시작하는 위치. 오류와 `offset`에 쓴다.
    base: usize,
    format: &'static str,
}

impl<'a> Message<'a> {
    /// 가장 바깥 메시지.
    pub fn new(bytes: &'a [u8], format: &'static str) -> Self {
        Self::nested(bytes, 0, format)
    }

    /// 다른 메시지 안에 든 메시지. `base`는 [`Field::offset`]이다.
    pub fn nested(bytes: &'a [u8], base: usize, format: &'static str) -> Self {
        Self {
            bytes,
            position: 0,
            base,
            format,
        }
    }

    fn error(&self, message: impl Into<String>) -> Error {
        Error::Parse {
            format: self.format,
            offset: self.base + self.position,
            message: message.into(),
        }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self
            .position
            .checked_add(len)
            .filter(|&end| end <= self.bytes.len())
            .ok_or_else(|| {
                self.error(format!("{len}바이트를 읽으려 했지만 메시지가 끝났습니다"))
            })?;
        let bytes = &self.bytes[self.position..end];
        self.position = end;
        Ok(bytes)
    }

    // ANCHOR: varint
    /// 길이가 변하는 정수를 읽는다.
    ///
    /// 한 바이트에 7비트씩, 낮은 자리부터 적는다. 바이트의 가장 높은 비트는
    /// "뒤에 더 있다"는 표시다. 작은 수는 한 바이트로 끝난다. 64비트 수는
    /// 많아야 10바이트다.
    fn varint(&mut self) -> Result<u64> {
        let mut value = 0_u64;
        for shift in (0..64).step_by(7) {
            let [byte] = *self.take(1)? else {
                unreachable!("take가 길이를 보장한다")
            };
            let bits = u64::from(byte & 0x7f);
            // 열째 바이트에는 한 비트만 들어갈 자리가 남는다.
            if shift == 63 && bits > 1 {
                break;
            }
            value |= bits << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(self.error("길이가 변하는 정수가 64비트를 넘습니다"))
    }
    // ANCHOR_END: varint

    // ANCHOR: next
    fn field(&mut self) -> Result<Field<'a>> {
        // 꼬리표: (필드의 번호 << 3) | 값을 적는 방법.
        let tag = self.varint()?;
        let number = u32::try_from(tag >> 3)
            .ok()
            .filter(|&number| number != 0)
            .ok_or_else(|| self.error("필드의 번호가 올바르지 않습니다"))?;
        let mut offset = 0;
        let value = match tag & 7 {
            0 => Value::Varint(self.varint()?),
            1 => Value::Fixed64(self.take(8)?.try_into().expect("take가 길이를 보장한다")),
            2 => {
                let len = self.varint()?;
                let len = usize::try_from(len)
                    .map_err(|_| self.error(format!("길이 {len}이 너무 큽니다")))?;
                offset = self.base + self.position;
                Value::Bytes(self.take(len)?)
            }
            5 => Value::Fixed32(self.take(4)?.try_into().expect("take가 길이를 보장한다")),
            // 3과 4는 예전의 묶음(group)이다. 지금은 쓰지 않는다.
            other => return Err(self.error(format!("값을 적는 방법 {other}번은 읽지 않습니다"))),
        };
        Ok(Field {
            number,
            value,
            offset,
        })
    }
    // ANCHOR_END: next
}

impl<'a> Iterator for Message<'a> {
    type Item = Result<Field<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position == self.bytes.len() {
            return None;
        }
        let field = self.field();
        if field.is_err() {
            // 오류 뒤에는 더 읽지 않는다.
            self.position = self.bytes.len();
        }
        Some(field)
    }
}

impl<'a> Field<'a> {
    fn mismatch(&self, format: &'static str, expected: &str) -> Error {
        Error::Parse {
            format,
            offset: self.offset,
            message: format!("필드 {}번이 {expected}이 아닙니다", self.number),
        }
    }

    /// 정수로 읽는다. 음수는 2의 보수 64비트로 적혀 있다.
    pub fn int(&self, format: &'static str) -> Result<i64> {
        match self.value {
            Value::Varint(value) => Ok(value.cast_signed()),
            _ => Err(self.mismatch(format, "정수")),
        }
    }

    /// `float`로 읽는다.
    pub fn float(&self, format: &'static str) -> Result<f32> {
        match self.value {
            Value::Fixed32(bytes) => Ok(f32::from_le_bytes(bytes)),
            _ => Err(self.mismatch(format, "float")),
        }
    }

    /// 바이트들로 읽는다.
    pub fn bytes(&self, format: &'static str) -> Result<&'a [u8]> {
        match self.value {
            Value::Bytes(bytes) => Ok(bytes),
            _ => Err(self.mismatch(format, "바이트열")),
        }
    }

    /// UTF-8 문자열로 읽는다.
    pub fn string(&self, format: &'static str) -> Result<&'a str> {
        std::str::from_utf8(self.bytes(format)?).map_err(|_| self.mismatch(format, "UTF-8 문자열"))
    }

    /// 안에 든 메시지로 읽는다.
    pub fn message(&self, format: &'static str) -> Result<Message<'a>> {
        Ok(Message::nested(self.bytes(format)?, self.offset, format))
    }

    // ANCHOR: packed
    /// 되풀이되는 정수 필드의 값들을 `into`에 더한다.
    ///
    /// 되풀이되는 숫자는 두 가지로 적힐 수 있다. 값마다 꼬리표를 붙여 따로
    /// 적거나, 꼬리표 하나 뒤에 값들을 이어 붙이거나(packed). 읽는 쪽은 둘
    /// 다 받아야 한다.
    pub fn ints(&self, into: &mut Vec<i64>, format: &'static str) -> Result<()> {
        match self.value {
            Value::Varint(value) => into.push(value.cast_signed()),
            Value::Bytes(bytes) => {
                let mut packed = Message::nested(bytes, self.offset, format);
                while packed.position < packed.bytes.len() {
                    into.push(packed.varint()?.cast_signed());
                }
            }
            _ => return Err(self.mismatch(format, "정수")),
        }
        Ok(())
    }

    /// 되풀이되는 `float` 필드의 값들을 `into`에 더한다.
    pub fn floats(&self, into: &mut Vec<f32>, format: &'static str) -> Result<()> {
        match self.value {
            Value::Fixed32(bytes) => into.push(f32::from_le_bytes(bytes)),
            Value::Bytes(bytes) => {
                let (chunks, rest) = bytes.as_chunks::<4>();
                if !rest.is_empty() {
                    return Err(self.mismatch(format, "float의 묶음"));
                }
                into.extend(chunks.iter().map(|&chunk| f32::from_le_bytes(chunk)));
            }
            _ => return Err(self.mismatch(format, "float")),
        }
        Ok(())
    }
    // ANCHOR_END: packed
}

#[cfg(test)]
mod tests {
    use super::*;

    const FORMAT: &str = "시험";

    #[test]
    fn reads_the_documented_examples() {
        // Protocol Buffers 문서의 예: 필드 1번에 150을 적으면 08 96 01.
        let fields: Vec<_> = Message::new(&[0x08, 0x96, 0x01], FORMAT)
            .collect::<Result<_>>()
            .unwrap();
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].number, 1);
        assert_eq!(fields[0].int(FORMAT).unwrap(), 150);
        // 필드 2번에 문자열 "testing": 12 07 74 65 73 74 69 6e 67.
        let bytes = [0x12, 0x07, 0x74, 0x65, 0x73, 0x74, 0x69, 0x6e, 0x67];
        let field = Message::new(&bytes, FORMAT).next().unwrap().unwrap();
        assert_eq!(field.number, 2);
        assert_eq!(field.string(FORMAT).unwrap(), "testing");
        assert_eq!(field.offset, 2);
    }

    #[test]
    fn reads_negative_and_packed_numbers() {
        // -1은 10바이트다.
        let mut bytes = vec![0x08];
        bytes.extend([0xff; 9]);
        bytes.push(0x01);
        let field = Message::new(&bytes, FORMAT).next().unwrap().unwrap();
        assert_eq!(field.int(FORMAT).unwrap(), -1);

        // 묶어 적은 것과 따로 적은 것이 같은 값이 된다.
        let packed = [0x22, 0x03, 0x03, 0x8e, 0x02];
        let separate = [0x20, 0x03, 0x20, 0x8e, 0x02];
        for bytes in [&packed[..], &separate[..]] {
            let mut values = Vec::new();
            for field in Message::new(bytes, FORMAT) {
                field.unwrap().ints(&mut values, FORMAT).unwrap();
            }
            assert_eq!(values, [3, 270]);
        }

        let mut floats = Vec::new();
        let mut bytes = vec![0x0a, 0x08];
        bytes.extend(1.5_f32.to_le_bytes());
        bytes.extend((-2.0_f32).to_le_bytes());
        for field in Message::new(&bytes, FORMAT) {
            field.unwrap().floats(&mut floats, FORMAT).unwrap();
        }
        assert_eq!(floats, [1.5, -2.0]);
    }

    #[test]
    fn nested_messages_report_absolute_offsets() {
        // 바깥 필드 3번 안에 필드 1번(문자열 "ab").
        let bytes = [0x1a, 0x04, 0x0a, 0x02, b'a', b'b'];
        let outer = Message::new(&bytes, FORMAT).next().unwrap().unwrap();
        let inner = outer.message(FORMAT).unwrap().next().unwrap().unwrap();
        assert_eq!(inner.string(FORMAT).unwrap(), "ab");
        assert_eq!(inner.offset, 4);
    }

    #[test]
    fn rejects_broken_input() {
        fn all(bytes: &[u8]) -> Result<Vec<Field<'_>>> {
            Message::new(bytes, FORMAT).collect()
        }
        // 끝나지 않는 정수.
        assert!(all(&[0x08, 0x80]).is_err());
        // 64비트를 넘는 정수.
        let mut long = vec![0x08];
        long.extend([0xff; 10]);
        assert!(all(&long).is_err());
        // 메시지보다 긴 길이.
        assert!(all(&[0x12, 0x05, 0x00]).is_err());
        // 번호가 0인 필드, 쓰지 않는 방법.
        assert!(all(&[0x00, 0x00]).is_err());
        assert!(all(&[0x0b]).is_err());
        // 잘린 고정 길이 값.
        assert!(all(&[0x0d, 0x00, 0x00]).is_err());
        // 자료형이 다르다.
        let field = Message::new(&[0x08, 0x01], FORMAT).next().unwrap().unwrap();
        assert!(field.bytes(FORMAT).is_err() && field.float(FORMAT).is_err());
        // 오류 뒤에는 멈춘다.
        let mut message = Message::new(&[0x0b, 0x08, 0x01], FORMAT);
        assert!(message.next().unwrap().is_err());
        assert!(message.next().is_none());
    }
}
