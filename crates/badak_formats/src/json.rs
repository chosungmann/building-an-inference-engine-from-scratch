//! 작은 JSON 파서.
//!
//! safetensors의 헤더를 읽는 데 필요한 만큼만 구현한다. JSON의 문법
//! 전체를 받아들이지만, 숫자는 모두 `f64`로 읽는다.

use std::collections::BTreeMap;

use crate::error::{Error, Result};

// ANCHOR: value
/// JSON 값.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// `null`
    Null,
    /// `true` 또는 `false`
    Bool(bool),
    /// 숫자. 정수도 `f64`로 담는다.
    Number(f64),
    /// 문자열
    String(String),
    /// 배열
    Array(Vec<Value>),
    /// 객체. 키의 순서로 정렬해 담는다.
    Object(BTreeMap<String, Value>),
}
// ANCHOR_END: value

impl Value {
    /// 문자열이면 그 내용을 돌려준다.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(text) => Some(text),
            _ => None,
        }
    }

    /// 숫자이면 `f64`로 돌려준다.
    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Self::Number(number) => Some(number),
            _ => None,
        }
    }

    /// 음이 아닌 정수이면 `usize`로 돌려준다.
    pub fn as_usize(&self) -> Option<usize> {
        /// `f64`가 정확히 나타낼 수 있는 가장 큰 정수, 2의 53제곱.
        const MAX_EXACT: f64 = 9_007_199_254_740_992.0;
        match *self {
            Self::Number(n) if n >= 0.0 && n.fract() == 0.0 && n <= MAX_EXACT => Some(n as usize),
            _ => None,
        }
    }

    /// 배열이면 그 원소들을 돌려준다.
    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Self::Array(items) => Some(items),
            _ => None,
        }
    }

    /// 객체이면 그 내용을 돌려준다.
    pub fn as_object(&self) -> Option<&BTreeMap<String, Value>> {
        match self {
            Self::Object(members) => Some(members),
            _ => None,
        }
    }

    /// 객체에서 키에 해당하는 값을 찾는다.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.as_object()?.get(key)
    }
}

/// JSON 문서 하나를 해석한다. 앞뒤의 공백은 허용하고, 값 뒤에 다른 내용이
/// 있으면 오류다.
pub fn parse(text: &str) -> Result<Value> {
    let mut parser = Parser {
        bytes: text.as_bytes(),
        position: 0,
        depth: 0,
    };
    let value = parser.value()?;
    parser.skip_whitespace();
    if parser.position != parser.bytes.len() {
        return Err(parser.error("값 뒤에 다른 내용이 있습니다"));
    }
    Ok(value)
}

/// 이보다 깊이 중첩된 문서는 거부한다. 악의적인 입력이 재귀로 스택을
/// 넘치게 하는 것을 막는다.
const MAX_DEPTH: usize = 64;

struct Parser<'a> {
    bytes: &'a [u8],
    position: usize,
    depth: usize,
}

// ANCHOR: parser
impl Parser<'_> {
    fn error(&self, message: &str) -> Error {
        Error::Parse {
            format: "JSON",
            offset: self.position,
            message: message.to_string(),
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.position).copied()
    }

    fn skip_whitespace(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.position += 1;
        }
    }

    /// 다음 글자가 `expected`이면 지나가고, 아니면 오류를 낸다.
    fn expect(&mut self, expected: u8) -> Result<()> {
        if self.peek() == Some(expected) {
            self.position += 1;
            Ok(())
        } else {
            Err(self.error(&format!("{:?}가 와야 합니다", expected as char)))
        }
    }

    /// 값 하나를 읽는다. 첫 글자를 보고 종류를 정한다.
    fn value(&mut self) -> Result<Value> {
        self.skip_whitespace();
        match self.peek() {
            Some(b'{') => self.nested(Self::object),
            Some(b'[') => self.nested(Self::array),
            Some(b'"') => Ok(Value::String(self.string()?)),
            Some(b't') => self.literal("true", Value::Bool(true)),
            Some(b'f') => self.literal("false", Value::Bool(false)),
            Some(b'n') => self.literal("null", Value::Null),
            Some(b'-' | b'0'..=b'9') => self.number(),
            _ => Err(self.error("값이 와야 합니다")),
        }
    }

    fn nested(&mut self, parse: fn(&mut Self) -> Result<Value>) -> Result<Value> {
        if self.depth == MAX_DEPTH {
            return Err(self.error("너무 깊이 중첩되어 있습니다"));
        }
        self.depth += 1;
        let value = parse(self);
        self.depth -= 1;
        value
    }

    fn literal(&mut self, word: &str, value: Value) -> Result<Value> {
        if self.bytes[self.position..].starts_with(word.as_bytes()) {
            self.position += word.len();
            Ok(value)
        } else {
            Err(self.error("알 수 없는 값입니다"))
        }
    }

    fn object(&mut self) -> Result<Value> {
        self.expect(b'{')?;
        let mut members = BTreeMap::new();
        self.skip_whitespace();
        if self.peek() == Some(b'}') {
            self.position += 1;
            return Ok(Value::Object(members));
        }
        loop {
            self.skip_whitespace();
            let key = self.string()?;
            self.skip_whitespace();
            self.expect(b':')?;
            let value = self.value()?;
            if members.insert(key, value).is_some() {
                return Err(self.error("같은 키가 두 번 나왔습니다"));
            }
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => self.position += 1,
                Some(b'}') => {
                    self.position += 1;
                    return Ok(Value::Object(members));
                }
                _ => return Err(self.error("',' 또는 '}'가 와야 합니다")),
            }
        }
    }

    fn array(&mut self) -> Result<Value> {
        self.expect(b'[')?;
        let mut items = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some(b']') {
            self.position += 1;
            return Ok(Value::Array(items));
        }
        loop {
            items.push(self.value()?);
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => self.position += 1,
                Some(b']') => {
                    self.position += 1;
                    return Ok(Value::Array(items));
                }
                _ => return Err(self.error("',' 또는 ']'가 와야 합니다")),
            }
        }
    }
    // ANCHOR_END: parser

    fn number(&mut self) -> Result<Value> {
        let start = self.position;
        while matches!(
            self.peek(),
            Some(b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9')
        ) {
            self.position += 1;
        }
        // 숫자에 쓰이는 글자만 모았으므로 ASCII이고, 따라서 유효한 UTF-8이다.
        let text = std::str::from_utf8(&self.bytes[start..self.position]).expect("ASCII");
        match text.parse::<f64>() {
            Ok(number) if number.is_finite() => Ok(Value::Number(number)),
            _ => {
                self.position = start;
                Err(self.error("숫자가 올바르지 않습니다"))
            }
        }
    }

    fn string(&mut self) -> Result<String> {
        self.expect(b'"')?;
        let mut text = String::new();
        loop {
            // 따옴표나 역슬래시가 나올 때까지를 한 번에 옮긴다.
            let start = self.position;
            while !matches!(self.peek(), Some(b'"' | b'\\') | None) {
                if self.bytes[self.position] < 0x20 {
                    return Err(self.error("문자열 안에 제어 문자가 있습니다"));
                }
                self.position += 1;
            }
            // 입력이 &str이었고 ASCII 글자에서만 끊었으므로 유효한 UTF-8이다.
            text.push_str(std::str::from_utf8(&self.bytes[start..self.position]).expect("UTF-8"));
            match self.peek() {
                Some(b'"') => {
                    self.position += 1;
                    return Ok(text);
                }
                Some(b'\\') => {
                    self.position += 1;
                    text.push(self.escape()?);
                }
                _ => return Err(self.error("문자열이 닫히지 않았습니다")),
            }
        }
    }

    /// 역슬래시 뒤의 글자를 해석한다.
    fn escape(&mut self) -> Result<char> {
        let Some(code) = self.peek() else {
            return Err(self.error("문자열이 닫히지 않았습니다"));
        };
        self.position += 1;
        Ok(match code {
            b'"' => '"',
            b'\\' => '\\',
            b'/' => '/',
            b'b' => '\u{8}',
            b'f' => '\u{c}',
            b'n' => '\n',
            b'r' => '\r',
            b't' => '\t',
            b'u' => {
                let first = self.hex4()?;
                let code_point = match first {
                    // UTF-16 대행 쌍: 두 개의 \u가 글자 하나를 이룬다.
                    0xD800..=0xDBFF => {
                        if !self.bytes[self.position..].starts_with(b"\\u") {
                            return Err(self.error("대행 쌍의 뒤쪽이 없습니다"));
                        }
                        self.position += 2;
                        let second = self.hex4()?;
                        if !(0xDC00..=0xDFFF).contains(&second) {
                            return Err(self.error("대행 쌍이 올바르지 않습니다"));
                        }
                        0x10000 + ((first - 0xD800) << 10) + (second - 0xDC00)
                    }
                    other => other,
                };
                char::from_u32(code_point).ok_or_else(|| self.error("유효한 글자가 아닙니다"))?
            }
            _ => return Err(self.error("알 수 없는 이스케이프입니다")),
        })
    }

    fn hex4(&mut self) -> Result<u32> {
        let digits = self
            .bytes
            .get(self.position..self.position + 4)
            .and_then(|digits| std::str::from_utf8(digits).ok())
            .filter(|digits| digits.bytes().all(|digit| digit.is_ascii_hexdigit()))
            .ok_or_else(|| self.error("\\u 뒤에는 16진수 네 자리가 와야 합니다"))?;
        self.position += 4;
        Ok(u32::from_str_radix(digits, 16).expect("16진수임을 확인했다"))
    }
}

/// 문자열을 JSON 문자열 리터럴로 바꾼다. 따옴표를 포함한다.
pub fn quote(text: &str) -> String {
    let mut quoted = String::with_capacity(text.len() + 2);
    quoted.push('"');
    for character in text.chars() {
        match character {
            '"' => quoted.push_str("\\\""),
            '\\' => quoted.push_str("\\\\"),
            '\n' => quoted.push_str("\\n"),
            '\r' => quoted.push_str("\\r"),
            '\t' => quoted.push_str("\\t"),
            c if (c as u32) < 0x20 => quoted.push_str(&format!("\\u{:04x}", c as u32)),
            c => quoted.push(c),
        }
    }
    quoted.push('"');
    quoted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_every_kind_of_value() {
        let value = parse(r#" {"a": [1, -2.5, 1e3], "b": true, "c": null, "d": "x"} "#).unwrap();
        assert_eq!(
            value.get("a").unwrap().as_array().unwrap(),
            [
                Value::Number(1.0),
                Value::Number(-2.5),
                Value::Number(1000.0)
            ]
        );
        assert_eq!(value.get("b"), Some(&Value::Bool(true)));
        assert_eq!(value.get("c"), Some(&Value::Null));
        assert_eq!(value.get("d").unwrap().as_str(), Some("x"));
        assert_eq!(parse("[]").unwrap(), Value::Array(vec![]));
        assert_eq!(parse("{}").unwrap(), Value::Object(BTreeMap::new()));
    }

    #[test]
    fn parses_a_safetensors_style_header() {
        let header = r#"{"w":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
        let value = parse(header).unwrap();
        let info = value.get("w").unwrap();
        assert_eq!(info.get("dtype").unwrap().as_str(), Some("F32"));
        let shape: Vec<usize> = info
            .get("shape")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_usize().unwrap())
            .collect();
        assert_eq!(shape, [2, 3]);
    }

    #[test]
    fn handles_strings_with_escapes_and_unicode() {
        assert_eq!(parse(r#""a\"b\\c\n""#).unwrap().as_str(), Some("a\"b\\c\n"));
        assert_eq!(parse(r#""밑바닥""#).unwrap().as_str(), Some("밑바닥"));
        assert_eq!(parse(r#""바닥""#).unwrap().as_str(), Some("바닥"));
        // 대행 쌍으로 적힌 이모지
        assert_eq!(parse(r#""😀""#).unwrap().as_str(), Some("😀"));
        assert!(parse(r#""\ud83d""#).is_err());
    }

    #[test]
    fn as_usize_accepts_only_exact_non_negative_integers() {
        assert_eq!(Value::Number(42.0).as_usize(), Some(42));
        assert_eq!(Value::Number(-1.0).as_usize(), None);
        assert_eq!(Value::Number(1.5).as_usize(), None);
        assert_eq!(Value::Number(1e300).as_usize(), None);
        assert_eq!(Value::String("1".into()).as_usize(), None);
    }

    #[test]
    fn rejects_malformed_documents() {
        for text in [
            "",
            "{",
            "[1,]",
            "{\"a\":}",
            "{\"a\" 1}",
            "tru",
            "\"abc",
            "1 2",
            "{\"a\":1,\"a\":2}",
            "01x",
            "\"\x01\"",
            "nul",
            "--1",
        ] {
            assert!(parse(text).is_err(), "{text:?}는 거부되어야 한다");
        }
    }

    #[test]
    fn rejects_deep_nesting_instead_of_overflowing_the_stack() {
        let deep = "[".repeat(100_000);
        assert!(parse(&deep).is_err());
        let ok = format!("{}{}", "[".repeat(MAX_DEPTH), "]".repeat(MAX_DEPTH));
        assert!(parse(&ok).is_ok());
    }

    #[test]
    fn quote_round_trips_through_parse() {
        for text in ["plain", "a\"b\\c", "줄\n바꿈\t탭", "\u{1}", "😀"] {
            assert_eq!(parse(&quote(text)).unwrap().as_str(), Some(text));
        }
    }
}
