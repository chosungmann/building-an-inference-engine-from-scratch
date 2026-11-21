//! 문법으로 묶은 생성 (41장).
//!
//! 언어 모델은 토큰마다 "다음에 올 수 있는 모든 토큰의 점수"를 낸다.
//! 그 가운데 **지금 와도 되는 토큰만 남기고** 고르게 하면, 모델이 무엇을
//! 쓰든 결과는 정해 둔 꼴을 벗어나지 않는다. 이 파일은 그 "와도 되는가"를
//! 판단한다. JSON 스키마를 받아, 바이트를 하나씩 먹으며 지금까지의 글이
//! 그 스키마에 맞는 JSON의 앞부분인지를 따라가는 기계를 만든다.

use std::sync::Arc;
use std::time::Instant;

use badak_formats::Result;
use badak_formats::json::{self, Value};

use crate::llama::cache::KvCache;
use crate::llama::model::LlamaModel;
use crate::sampling::Sampler;
use crate::tokenizer::{TokenId, Tokenizer};

/// 길이를 정하지 않은 문자열이 가질 수 있는 글자 수의 한도.
const DEFAULT_MAX_LENGTH: usize = 256;
/// 길이를 정하지 않은 배열이 가질 수 있는 원소 수의 한도.
const DEFAULT_MAX_ITEMS: usize = 64;
/// 숫자의 한 부분(정수부, 소수부)이 가질 수 있는 자릿수의 한도.
const MAX_DIGITS: u8 = 15;

/// 스키마의 노드 하나.
#[derive(Debug, Clone, PartialEq)]
enum Node {
    /// 문자열. 글자 수의 한도가 있다.
    String {
        max: usize,
    },
    /// 정해진 글들 가운데 하나. 따옴표까지 넣은 바이트들이다.
    Choice(Vec<Vec<u8>>),
    Integer,
    Number,
    Boolean,
    Null,
    /// 배열. 원소의 노드와 개수의 범위.
    Array {
        item: usize,
        min: usize,
        max: usize,
    },
    /// 객체. 필드마다 `"이름":`의 바이트들과 값의 노드. 이 순서대로 쓴다.
    Object {
        fields: Vec<(Vec<u8>, usize)>,
    },
}

// ANCHOR: grammar
/// JSON 스키마를 기계가 따라가기 좋게 풀어 놓은 것.
#[derive(Debug, Clone, PartialEq)]
pub struct Grammar {
    nodes: Vec<Node>,
    root: usize,
}

impl Grammar {
    /// JSON 스키마에서 만든다. 스키마의 일부만 받는다.
    ///
    /// 받는 것: `type`이 `object`(`properties`, `required`), `array`(`items`,
    /// `minItems`, `maxItems`), `string`(`maxLength`), `integer`, `number`,
    /// `boolean`, `null`인 것과, 문자열들의 `enum`. 객체의 필드는 **모두
    /// 써야 하는 것**으로 보고, `required`에 적힌 순서대로(나머지는 이름
    /// 순서로) 쓰게 한다. 그 밖의 낱말(`anyOf`, `$ref`,
    /// `pattern`, `minimum` 등)이 하나라도 있으면 받지 않고 무엇을 못
    /// 받는지 말한다.
    pub fn from_schema(schema: &Value) -> std::result::Result<Self, String> {
        let mut nodes = Vec::new();
        let root = compile(schema, &mut nodes)?;
        Ok(Self { nodes, root })
    }

    /// 해석한 JSON 값이 스키마에 맞는가. 글자가 어떻게 놓였는지(빈칸,
    /// 줄바꿈, 필드의 순서)는 보지 않는다. 문법에 묶지 않고 쓴 답을 검사할
    /// 때 쓴다.
    pub fn validates(&self, value: &Value) -> bool {
        self.check(self.root, value)
    }

    fn check(&self, node: usize, value: &Value) -> bool {
        match (&self.nodes[node], value) {
            (Node::String { max }, Value::String(text)) => text.chars().count() <= *max,
            (Node::Choice(choices), Value::String(text)) => {
                choices.contains(&json::quote(text).into_bytes())
            }
            (Node::Integer, Value::Number(number)) => number.fract() == 0.0,
            (Node::Number, Value::Number(_))
            | (Node::Boolean, Value::Bool(_))
            | (Node::Null, Value::Null) => true,
            (Node::Array { item, min, max }, Value::Array(items)) => {
                (*min..=*max).contains(&items.len())
                    && items.iter().all(|value| self.check(*item, value))
            }
            (Node::Object { fields }, Value::Object(map)) => {
                map.len() == fields.len()
                    && fields.iter().all(|(key, node)| {
                        // `"이름":`에서 이름을 되찾는다.
                        let name = std::str::from_utf8(&key[..key.len() - 1])
                            .ok()
                            .and_then(|quoted| json::parse(quoted).ok());
                        name.as_ref()
                            .and_then(Value::as_str)
                            .and_then(|name| map.get(name))
                            .is_some_and(|value| self.check(*node, value))
                    })
            }
            _ => false,
        }
    }
}
// ANCHOR_END: grammar

fn compile(schema: &Value, nodes: &mut Vec<Node>) -> std::result::Result<usize, String> {
    let limit = |name: &str, default: usize| match schema.get(name) {
        None => Ok(default),
        Some(value) => value
            .as_usize()
            .ok_or_else(|| format!("{name}은 0 이상의 정수여야 합니다")),
    };
    // 모르는 낱말을 조용히 넘기면 "스키마에 맞는다"는 말이 거짓이 된다.
    // `minimum`이 적힌 스키마를 받아 놓고 음수를 쓰게 둘 수는 없다.
    const KNOWN: [&str; 11] = [
        "type",
        "enum",
        "properties",
        "required",
        "items",
        "minItems",
        "maxItems",
        "maxLength",
        "additionalProperties",
        "title",
        "description",
    ];
    let keys = schema.as_object().ok_or("스키마는 객체여야 합니다")?;
    if let Some(unknown) = keys.keys().find(|key| !KNOWN.contains(&key.as_str())) {
        return Err(format!("{unknown}은 받지 않습니다"));
    }
    if !matches!(
        schema.get("additionalProperties"),
        None | Some(Value::Bool(false))
    ) {
        return Err("additionalProperties는 false만 받습니다".into());
    }
    let node = if let Some(choices) = schema.get("enum") {
        let choices = choices.as_array().ok_or("enum은 배열이어야 합니다")?;
        let quoted: Option<Vec<Vec<u8>>> = choices
            .iter()
            .map(|choice| choice.as_str().map(|text| json::quote(text).into_bytes()))
            .collect();
        match quoted {
            Some(quoted) if (1..=64).contains(&quoted.len()) => Node::Choice(quoted),
            _ => return Err("enum은 문자열 1~64개의 배열이어야 합니다".into()),
        }
    } else {
        match schema.get("type").and_then(Value::as_str) {
            Some("string") => Node::String {
                max: limit("maxLength", DEFAULT_MAX_LENGTH)?,
            },
            Some("integer") => Node::Integer,
            Some("number") => Node::Number,
            Some("boolean") => Node::Boolean,
            Some("null") => Node::Null,
            Some("array") => {
                let items = schema
                    .get("items")
                    .ok_or("array에는 items가 있어야 합니다")?;
                let (min, max) = (limit("minItems", 0)?, limit("maxItems", DEFAULT_MAX_ITEMS)?);
                if min > max {
                    return Err("minItems가 maxItems보다 큽니다".into());
                }
                Node::Array {
                    item: compile(items, nodes)?,
                    min,
                    max,
                }
            }
            Some("object") => {
                let empty = std::collections::BTreeMap::new();
                let properties = match schema.get("properties") {
                    None => &empty,
                    Some(properties) => properties
                        .as_object()
                        .ok_or("properties는 객체여야 합니다")?,
                };
                // 쓰는 순서: required에 적힌 순서, 그 뒤에 나머지를 이름 순서로.
                let mut order: Vec<&str> = Vec::new();
                if let Some(required) = schema.get("required").and_then(Value::as_array) {
                    for name in required {
                        let name = name
                            .as_str()
                            .ok_or("required는 이름들의 배열이어야 합니다")?;
                        if !properties.contains_key(name) {
                            return Err(format!("required의 {name}이 properties에 없습니다"));
                        }
                        if !order.contains(&name) {
                            order.push(name);
                        }
                    }
                }
                for name in properties.keys() {
                    if !order.contains(&name.as_str()) {
                        order.push(name);
                    }
                }
                let mut fields = Vec::with_capacity(order.len());
                for name in order {
                    let mut key = json::quote(name).into_bytes();
                    key.push(b':');
                    fields.push((key, compile(&properties[name], nodes)?));
                }
                Node::Object { fields }
            }
            Some(other) => return Err(format!("type {other}은 받지 않습니다")),
            None => return Err("스키마에 type이나 enum이 있어야 합니다".into()),
        }
    };
    nodes.push(node);
    Ok(nodes.len() - 1)
}

/// 문자열 안에서 역슬래시 뒤에 무엇을 기다리는가.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Escape {
    /// 역슬래시 뒤가 아니다.
    None,
    /// 역슬래시 바로 뒤.
    Slash,
    /// `\u` 뒤. 16진수 몇 자리가 남았는가.
    Hex(u8),
}

/// 숫자의 어디까지 왔는가.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Digits {
    Start,
    Minus,
    Zero,
    Whole(u8),
    Dot,
    Fraction(u8),
    Exponent,
    ExponentSign,
    Power(u8),
}

impl Digits {
    /// 여기서 숫자가 끝나도 되는가.
    fn complete(self) -> bool {
        matches!(
            self,
            Self::Zero | Self::Whole(_) | Self::Fraction(_) | Self::Power(_)
        )
    }
}

/// 기계가 기다리고 있는 것 하나. 쌓아 둔다.
#[derive(Debug, Clone, PartialEq)]
enum Frame {
    /// 값이 시작될 자리. `space`가 참이면 그 앞에 빈칸 하나가 와도 된다.
    Value { node: usize, space: bool },
    /// `true`, `false`, `null`의 나머지.
    Fixed { text: &'static [u8], at: usize },
    /// 객체의 필드 이름(`"이름":`)의 나머지.
    Key {
        node: usize,
        field: usize,
        at: usize,
        space: bool,
    },
    /// 필드의 값이 끝난 뒤. 쉼표나 닫는 중괄호.
    AfterField { node: usize, field: usize },
    /// 여는 대괄호 바로 뒤.
    ArrayOpen { node: usize },
    /// 원소 `count`개를 쓴 뒤. 쉼표나 닫는 대괄호.
    AfterItem { node: usize, count: usize },
    /// 문자열의 안. 더 쓸 수 있는 글자 수, 역슬래시의 상태, UTF-8 한 글자의
    /// 남은 바이트 수.
    Text {
        left: usize,
        escape: Escape,
        tail: u8,
    },
    /// 정해진 글들 가운데 아직 맞는 것들(비트로)과 지금까지 맞춘 길이.
    Choice { node: usize, alive: u64, at: usize },
    /// 숫자.
    Number { state: Digits, integer: bool },
}

/// 바이트 하나를 먹은 결과.
enum Step {
    /// 받았다.
    Taken,
    /// 이 틀은 끝났고, 이 바이트는 그 아래의 틀이 봐야 한다.
    Pass,
    /// 올 수 없는 바이트다.
    Refused,
}

// ANCHOR: matcher
/// 지금까지의 글이 문법에 맞는 JSON의 앞부분인지를 따라가는 기계.
///
/// 상태는 "기다리고 있는 것들"을 쌓아 둔 것이다. 객체 안의 배열 안의
/// 문자열을 쓰는 중이라면 셋이 쌓여 있다. 토큰 하나가 와도 되는지 보려면
/// 기계를 베껴서 그 토큰의 바이트들을 먹여 본다.
#[derive(Debug, Clone)]
pub struct Matcher {
    grammar: Arc<Grammar>,
    stack: Vec<Frame>,
}

impl Matcher {
    /// 문법의 처음에 선 기계.
    pub fn new(grammar: &Arc<Grammar>) -> Self {
        Self {
            stack: vec![Frame::Value {
                node: grammar.root,
                space: false,
            }],
            grammar: Arc::clone(grammar),
        }
    }

    /// 문법에 맞는 글 전체인가. 다 쓴 글을 검사할 때 쓴다.
    pub fn matches(grammar: &Arc<Grammar>, text: &str) -> bool {
        let mut matcher = Self::new(grammar);
        matcher.consume(text.as_bytes()) && matcher.complete()
    }

    /// 여기서 글이 끝나도 되는가.
    pub fn complete(&self) -> bool {
        match self.stack.as_slice() {
            [] => true,
            // 맨 바깥이 숫자이면 끝을 알리는 글자가 없다.
            [Frame::Number { state, .. }] => state.complete(),
            _ => false,
        }
    }

    /// 더 올 수 있는 바이트가 없는가. 값이 다 끝났다.
    pub fn finished(&self) -> bool {
        self.stack.is_empty()
    }

    /// 바이트 하나를 먹는다. 올 수 없는 바이트이면 거짓이고, 그 뒤로 이
    /// 기계는 쓸 수 없다.
    pub fn advance(&mut self, byte: u8) -> bool {
        loop {
            let Some(frame) = self.stack.pop() else {
                return false;
            };
            match self.step(frame, byte) {
                Step::Taken => return true,
                Step::Pass => {}
                Step::Refused => return false,
            }
        }
    }

    /// 이 바이트들이 이어서 올 수 있는가. 기계는 그대로 둔다.
    pub fn accepts(&self, bytes: &[u8]) -> bool {
        let mut trial = self.clone();
        bytes.iter().all(|&byte| trial.advance(byte))
    }

    /// 바이트들을 먹는다. 고른 토큰을 기계에 알릴 때 쓴다.
    pub fn consume(&mut self, bytes: &[u8]) -> bool {
        bytes.iter().all(|&byte| self.advance(byte))
    }
    // ANCHOR_END: matcher

    // ANCHOR: step
    /// 맨 위의 틀 하나가 바이트 하나를 본다.
    fn step(&mut self, frame: Frame, byte: u8) -> Step {
        let grammar = Arc::clone(&self.grammar);
        let taken = |stack: &mut Vec<Frame>, frames: &[Frame]| {
            stack.extend_from_slice(frames);
            Step::Taken
        };
        match frame {
            Frame::Value { node, space } => {
                if space && byte == b' ' {
                    return taken(&mut self.stack, &[Frame::Value { node, space: false }]);
                }
                match (&grammar.nodes[node], byte) {
                    (Node::String { max }, b'"') => taken(
                        &mut self.stack,
                        &[Frame::Text {
                            left: *max,
                            escape: Escape::None,
                            tail: 0,
                        }],
                    ),
                    (Node::Choice(choices), _) => {
                        // 처음에는 모든 후보가 살아 있다. 이 바이트를 다시 본다.
                        let alive = u64::MAX >> (64 - choices.len());
                        self.stack.push(Frame::Choice { node, alive, at: 0 });
                        Step::Pass
                    }
                    (Node::Integer | Node::Number, _) => {
                        self.stack.push(Frame::Number {
                            state: Digits::Start,
                            integer: grammar.nodes[node] == Node::Integer,
                        });
                        Step::Pass
                    }
                    (Node::Boolean, b't') => taken(
                        &mut self.stack,
                        &[Frame::Fixed {
                            text: b"true",
                            at: 1,
                        }],
                    ),
                    (Node::Boolean, b'f') => taken(
                        &mut self.stack,
                        &[Frame::Fixed {
                            text: b"false",
                            at: 1,
                        }],
                    ),
                    (Node::Null, b'n') => taken(
                        &mut self.stack,
                        &[Frame::Fixed {
                            text: b"null",
                            at: 1,
                        }],
                    ),
                    (Node::Array { .. }, b'[') => {
                        taken(&mut self.stack, &[Frame::ArrayOpen { node }])
                    }
                    (Node::Object { fields }, b'{') if fields.is_empty() => {
                        taken(&mut self.stack, &[Frame::Fixed { text: b"{}", at: 1 }])
                    }
                    (Node::Object { .. }, b'{') => taken(
                        &mut self.stack,
                        &[Frame::Key {
                            node,
                            field: 0,
                            at: 0,
                            space: false,
                        }],
                    ),
                    _ => Step::Refused,
                }
            }
            Frame::Fixed { text, at } => match text.get(at) {
                Some(&expected) if expected == byte && at + 1 == text.len() => Step::Taken,
                Some(&expected) if expected == byte => {
                    taken(&mut self.stack, &[Frame::Fixed { text, at: at + 1 }])
                }
                _ => Step::Refused,
            },
            Frame::Key {
                node,
                field,
                at,
                space,
            } => {
                let Node::Object { fields } = &grammar.nodes[node] else {
                    return Step::Refused;
                };
                if space && byte == b' ' {
                    return taken(
                        &mut self.stack,
                        &[Frame::Key {
                            node,
                            field,
                            at,
                            space: false,
                        }],
                    );
                }
                let (key, value) = &fields[field];
                if key[at] != byte {
                    return Step::Refused;
                }
                if at + 1 < key.len() {
                    return taken(
                        &mut self.stack,
                        &[Frame::Key {
                            node,
                            field,
                            at: at + 1,
                            space: false,
                        }],
                    );
                }
                // 이름과 쌍점까지 썼다. 값을 쓰고, 그 뒤에 쉼표나 닫는 괄호가 온다.
                taken(
                    &mut self.stack,
                    &[
                        Frame::AfterField { node, field },
                        Frame::Value {
                            node: *value,
                            space: true,
                        },
                    ],
                )
            }
            Frame::AfterField { node, field } => {
                let Node::Object { fields } = &grammar.nodes[node] else {
                    return Step::Refused;
                };
                match (byte, field + 1 < fields.len()) {
                    (b',', true) => taken(
                        &mut self.stack,
                        &[Frame::Key {
                            node,
                            field: field + 1,
                            at: 0,
                            space: true,
                        }],
                    ),
                    (b'}', false) => Step::Taken,
                    _ => Step::Refused,
                }
            }
            Frame::ArrayOpen { node } => {
                let Node::Array { item, min, max } = grammar.nodes[node] else {
                    return Step::Refused;
                };
                if byte == b']' {
                    return if min == 0 { Step::Taken } else { Step::Refused };
                }
                if max == 0 {
                    return Step::Refused;
                }
                // 첫 원소가 시작된다. 이 바이트를 원소의 값이 다시 본다.
                self.stack.push(Frame::AfterItem { node, count: 1 });
                self.stack.push(Frame::Value {
                    node: item,
                    space: false,
                });
                Step::Pass
            }
            Frame::AfterItem { node, count } => {
                let Node::Array { item, min, max } = grammar.nodes[node] else {
                    return Step::Refused;
                };
                match byte {
                    b',' if count < max => taken(
                        &mut self.stack,
                        &[
                            Frame::AfterItem {
                                node,
                                count: count + 1,
                            },
                            Frame::Value {
                                node: item,
                                space: true,
                            },
                        ],
                    ),
                    b']' if count >= min => Step::Taken,
                    _ => Step::Refused,
                }
            }
            // ANCHOR: text
            Frame::Text { left, escape, tail } => {
                let text = |left, escape, tail| Frame::Text { left, escape, tail };
                // UTF-8 한 글자의 뒤쪽 바이트를 기다리는 중.
                if tail > 0 {
                    return if (0x80..=0xbf).contains(&byte) {
                        taken(&mut self.stack, &[text(left, escape, tail - 1)])
                    } else {
                        Step::Refused
                    };
                }
                match escape {
                    Escape::Slash => match byte {
                        b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {
                            taken(&mut self.stack, &[text(left, Escape::None, 0)])
                        }
                        b'u' => taken(&mut self.stack, &[text(left, Escape::Hex(4), 0)]),
                        _ => Step::Refused,
                    },
                    Escape::Hex(digits) if byte.is_ascii_hexdigit() => {
                        let escape = if digits == 1 {
                            Escape::None
                        } else {
                            Escape::Hex(digits - 1)
                        };
                        taken(&mut self.stack, &[text(left, escape, 0)])
                    }
                    Escape::Hex(_) => Step::Refused,
                    Escape::None => match byte {
                        b'"' => Step::Taken,
                        // 글자 수의 한도에 닿았으면 닫는 따옴표만 받는다.
                        _ if left == 0 => Step::Refused,
                        b'\\' => taken(&mut self.stack, &[text(left - 1, Escape::Slash, 0)]),
                        0x20..=0x7f => taken(&mut self.stack, &[text(left - 1, Escape::None, 0)]),
                        0xc2..=0xdf => taken(&mut self.stack, &[text(left - 1, Escape::None, 1)]),
                        0xe0..=0xef => taken(&mut self.stack, &[text(left - 1, Escape::None, 2)]),
                        0xf0..=0xf4 => taken(&mut self.stack, &[text(left - 1, Escape::None, 3)]),
                        // 제어 문자와, UTF-8의 첫 바이트가 될 수 없는 것.
                        _ => Step::Refused,
                    },
                }
            }
            // ANCHOR_END: text
            Frame::Choice { node, alive, at } => {
                let Node::Choice(choices) = &grammar.nodes[node] else {
                    return Step::Refused;
                };
                let mut still = 0u64;
                let mut ended = false;
                for (index, choice) in choices.iter().enumerate() {
                    if alive & (1 << index) != 0 && choice.get(at) == Some(&byte) {
                        still |= 1 << index;
                        ended |= at + 1 == choice.len();
                    }
                }
                match (still, ended) {
                    (0, _) => Step::Refused,
                    // 따옴표로 끝나는 글들이라 하나가 다른 하나의 앞부분일 수 없다.
                    (_, true) => Step::Taken,
                    _ => taken(
                        &mut self.stack,
                        &[Frame::Choice {
                            node,
                            alive: still,
                            at: at + 1,
                        }],
                    ),
                }
            }
            // ANCHOR: number
            Frame::Number { state, integer } => {
                let digit = byte.is_ascii_digit();
                let next = match (state, byte) {
                    (Digits::Start, b'-') => Some(Digits::Minus),
                    (Digits::Start | Digits::Minus, b'0') => Some(Digits::Zero),
                    (Digits::Start | Digits::Minus, b'1'..=b'9') => Some(Digits::Whole(1)),
                    (Digits::Whole(count), _) if digit && count < MAX_DIGITS => {
                        Some(Digits::Whole(count + 1))
                    }
                    (Digits::Zero | Digits::Whole(_), b'.') if !integer => Some(Digits::Dot),
                    (Digits::Dot, _) if digit => Some(Digits::Fraction(1)),
                    (Digits::Fraction(count), _) if digit && count < MAX_DIGITS => {
                        Some(Digits::Fraction(count + 1))
                    }
                    (Digits::Zero | Digits::Whole(_) | Digits::Fraction(_), b'e' | b'E')
                        if !integer =>
                    {
                        Some(Digits::Exponent)
                    }
                    (Digits::Exponent, b'+' | b'-') => Some(Digits::ExponentSign),
                    (Digits::Exponent | Digits::ExponentSign, _) if digit => Some(Digits::Power(1)),
                    (Digits::Power(count), _) if digit && count < 3 => {
                        Some(Digits::Power(count + 1))
                    }
                    _ => None,
                };
                match next {
                    Some(state) => taken(&mut self.stack, &[Frame::Number { state, integer }]),
                    // 숫자에는 끝을 알리는 글자가 없다. 숫자가 될 수 없는
                    // 바이트가 왔고 여기서 끝나도 되면, 그 바이트는 아래의
                    // 틀(쉼표나 닫는 괄호를 기다리는)이 본다.
                    None if state.complete() => Step::Pass,
                    None => Step::Refused,
                }
            } // ANCHOR_END: number
        }
    }
    // ANCHOR_END: step
}

/// 단어 사전을 바이트의 나무로 엮은 것의 노드.
#[derive(Debug, Default)]
struct Branch {
    /// (바이트, 그 바이트로 가는 노드). 바이트 순서로 놓인다.
    children: Vec<(u8, u32)>,
    /// 여기서 끝나는 토큰들.
    tokens: Vec<TokenId>,
}

// ANCHOR: vocabulary
/// 토큰마다의 바이트들. 한 번 만들어 두고 걸음마다 쓴다.
pub struct Vocabulary {
    /// 토큰 번호로 찾는 바이트들. 쓸 수 없는 토큰(특수 토큰)은 비어 있다.
    bytes: Vec<Vec<u8>>,
    /// 같은 앞부분을 가진 토큰들이 같은 가지를 함께 타는 나무.
    trie: Vec<Branch>,
}

impl Vocabulary {
    /// 토크나이저에서 만든다. 특수 토큰은 뺀다. 글의 내용이 될 수 없다.
    pub fn new(tokenizer: &Tokenizer) -> Self {
        let bytes: Vec<Vec<u8>> = (0..tokenizer.vocab_size() as TokenId)
            .map(|id| {
                if tokenizer.is_special(id) {
                    Vec::new()
                } else {
                    tokenizer.token_bytes(id)
                }
            })
            .collect();
        let mut trie = vec![Branch::default()];
        for (id, token) in bytes.iter().enumerate() {
            if token.is_empty() {
                continue;
            }
            let mut at = 0usize;
            for &byte in token {
                let found = trie[at]
                    .children
                    .binary_search_by_key(&byte, |(key, _)| *key);
                at = match found {
                    Ok(index) => trie[at].children[index].1 as usize,
                    Err(index) => {
                        let fresh = trie.len() as u32;
                        trie[at].children.insert(index, (byte, fresh));
                        trie.push(Branch::default());
                        fresh as usize
                    }
                };
            }
            trie[at].tokens.push(id as TokenId);
        }
        Self { bytes, trie }
    }

    /// 토큰의 바이트들.
    pub fn bytes(&self, id: TokenId) -> &[u8] {
        &self.bytes[id as usize]
    }

    /// 나무의 노드의 수.
    pub fn branches(&self) -> usize {
        self.trie.len()
    }

    /// 지금 와도 되는 토큰들. **모든 토큰을 하나씩** 기계에 먹여 본다.
    pub fn allowed_by_scan(&self, matcher: &Matcher) -> Vec<TokenId> {
        (0..self.bytes.len())
            .filter(|&id| !self.bytes[id].is_empty() && matcher.accepts(&self.bytes[id]))
            .map(|id| id as TokenId)
            .collect()
    }

    /// 지금 와도 되는 토큰들. 나무를 따라 내려가며, 기계가 받지 않는
    /// 바이트의 가지는 **통째로** 건너뛴다.
    pub fn allowed_by_trie(&self, matcher: &Matcher) -> Vec<TokenId> {
        let mut allowed = Vec::new();
        self.descend(0, matcher, &mut allowed);
        allowed
    }

    fn descend(&self, at: usize, matcher: &Matcher, allowed: &mut Vec<TokenId>) {
        for &(byte, child) in &self.trie[at].children {
            let mut next = matcher.clone();
            if next.advance(byte) {
                let child = child as usize;
                allowed.extend_from_slice(&self.trie[child].tokens);
                self.descend(child, &next, allowed);
            }
        }
    }
}
// ANCHOR_END: vocabulary

/// 와도 되는 토큰을 어떻게 찾을지.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Search {
    /// 모든 토큰을 하나씩 본다.
    Scan,
    /// 단어 사전의 나무를 따라 내려간다.
    Trie,
    /// 점수가 높은 토큰부터 보고, 처음으로 와도 되는 것에서 멈춘다. 점수가
    /// 가장 높은 것을 고를 때(온도 0)만 쓸 수 있다.
    Best,
}

// ANCHOR: constrain
/// 문법에 묶인 글 하나를 쓰는 동안의 상태.
pub struct Constraint {
    vocabulary: Arc<Vocabulary>,
    matcher: Matcher,
    search: Search,
    /// 걸음마다 와도 되는 토큰이 몇 개였는지. 측정에 쓴다.
    pub allowed: Vec<usize>,
}

impl Constraint {
    /// 문법의 처음에서 시작한다.
    pub fn new(vocabulary: &Arc<Vocabulary>, grammar: &Arc<Grammar>, search: Search) -> Self {
        Self {
            vocabulary: Arc::clone(vocabulary),
            matcher: Matcher::new(grammar),
            search,
            allowed: Vec::new(),
        }
    }

    /// 글이 다 끝났는가. 더 올 수 있는 것이 없다.
    pub fn finished(&self) -> bool {
        self.matcher.finished()
    }

    /// 점수에서 **지금 올 수 없는 토큰을 지운다**(음의 무한대로 만든다).
    /// 글이 여기서 끝나도 되면 끝 토큰들은 남긴다.
    pub fn mask(&mut self, logits: &mut [f32], end: &[TokenId]) {
        let allowed = match self.search {
            Search::Scan => self.vocabulary.allowed_by_scan(&self.matcher),
            Search::Trie => self.vocabulary.allowed_by_trie(&self.matcher),
            Search::Best => self.best(logits).into_iter().collect(),
        };
        self.allowed.push(allowed.len());
        let mut kept = vec![f32::NEG_INFINITY; logits.len()];
        for &id in &allowed {
            kept[id as usize] = logits[id as usize];
        }
        if self.matcher.complete() {
            for &id in end {
                kept[id as usize] = logits[id as usize];
            }
        }
        logits.copy_from_slice(&kept);
    }

    /// 와도 되는 토큰 가운데 점수가 가장 높은 것 하나.
    ///
    /// 점수가 가장 높은 토큰을 찾아 기계에 먹여 보고, 안 되면 그것을 빼고
    /// 다시 찾는다. 모델이 문법을 대체로 따르고 있으면 한두 번에 끝난다.
    fn best(&self, logits: &[f32]) -> Option<TokenId> {
        let mut refused = Vec::new();
        loop {
            let mut best = None;
            for (id, &logit) in logits.iter().enumerate() {
                let better = best.is_none_or(|(_, top)| logit > top);
                if better && logit > f32::NEG_INFINITY && !refused.contains(&id) {
                    best = Some((id, logit));
                }
            }
            let (id, _) = best?;
            let bytes = self.vocabulary.bytes(id as TokenId);
            if !bytes.is_empty() && self.matcher.accepts(bytes) {
                return Some(id as TokenId);
            }
            refused.push(id);
            // 여러 번 틀리면 한 번씩 훑는 것이 더 느리다. 나무로 찾는다.
            if refused.len() > 32 {
                let allowed = self.vocabulary.allowed_by_trie(&self.matcher);
                return allowed
                    .into_iter()
                    .max_by(|&a, &b| logits[a as usize].total_cmp(&logits[b as usize]));
            }
        }
    }

    /// 고른 토큰을 기계에 알린다.
    pub fn accept(&mut self, id: TokenId) {
        let taken = self.matcher.consume(self.vocabulary.bytes(id));
        debug_assert!(taken, "지워 둔 토큰이 골라졌다");
    }
}
// ANCHOR_END: constrain

// ANCHOR: generate
/// 문법에 묶어 글을 이어 쓴다. 쓴 토큰들과, 올 수 있는 토큰을 찾는 데 쓴
/// 시간(초)을 돌려준다.
///
/// 12장의 생성 루프에 두 줄이 더해졌다. 고르기 전에 점수를 지우고, 고른
/// 뒤에 기계를 한 걸음 보낸다.
pub fn generate_constrained(
    model: &LlamaModel,
    prompt: &[TokenId],
    max_new_tokens: usize,
    cache: &mut KvCache,
    sampler: &mut Sampler,
    constraint: &mut Constraint,
    mut on_token: impl FnMut(TokenId),
) -> Result<(Vec<TokenId>, f64)> {
    let end = &model.config.eos_token_ids;
    let mut history = prompt.to_vec();
    let mut input = prompt[cache.len()..].to_vec();
    let mut searching = 0.0;
    while history.len() < prompt.len() + max_new_tokens && !constraint.finished() {
        let logits = model.forward_cached(&input, cache)?;
        let mut logits = logits.as_slice::<f32>()?.to_vec();
        let start = Instant::now();
        constraint.mask(&mut logits, end);
        searching += start.elapsed().as_secs_f64();
        let next = sampler.sample(&logits, &history);
        if end.contains(&next) {
            break;
        }
        constraint.accept(next);
        history.push(next);
        on_token(next);
        input = vec![next];
    }
    Ok((history.split_off(prompt.len()), searching))
}
// ANCHOR_END: generate

#[cfg(test)]
mod tests {
    use super::*;

    fn grammar(schema: &str) -> Arc<Grammar> {
        Arc::new(Grammar::from_schema(&json::parse(schema).unwrap()).unwrap())
    }

    /// 글 전체가 문법에 맞고, 거기서 끝나도 되는가.
    fn whole(grammar: &Arc<Grammar>, text: &str) -> bool {
        let mut matcher = Matcher::new(grammar);
        matcher.consume(text.as_bytes()) && matcher.complete()
    }

    /// 글이 문법에 맞는 글의 앞부분인가.
    fn prefix(grammar: &Arc<Grammar>, text: &str) -> bool {
        Matcher::new(grammar).accepts(text.as_bytes())
    }

    const PERSON: &str = r#"{"type":"object","properties":{
        "name":{"type":"string","maxLength":8},
        "age":{"type":"integer"},
        "tags":{"type":"array","items":{"enum":["a","bc"]},"minItems":1,"maxItems":2},
        "score":{"type":"number"},
        "ok":{"type":"boolean"},
        "none":{"type":"null"}},
        "required":["name","age"]}"#;

    #[test]
    fn accepts_what_the_schema_describes() {
        let person = grammar(PERSON);
        for text in [
            r#"{"name":"Ada","age":36,"none":null,"ok":true,"score":-1.5e3,"tags":["a"]}"#,
            r#"{"name": "한글 \"q\" é", "age": 0, "none": null, "ok": false, "score": 0.25, "tags": ["bc", "a"]}"#,
            r#"{"name":"","age":-7,"none":null,"ok":true,"score":12,"tags":["a","a"]}"#,
        ] {
            assert!(whole(&person, text), "{text}");
            // 우리의 JSON 해석기도 읽을 수 있어야 한다.
            assert!(json::parse(text).is_ok(), "{text}");
        }
    }

    #[test]
    fn validates_parsed_values_whatever_their_spacing() {
        let person = grammar(PERSON);
        let pretty = "{\n  \"tags\": [\"a\"],\n  \"age\": 36,\n  \"name\": \"Ada\",\n  \"none\": null,\n  \"ok\": true,\n  \"score\": 1.5\n}";
        // 줄을 바꾸고 순서를 섞은 글은 기계가 받지 않지만, 값으로는 맞다.
        assert!(!Matcher::matches(&person, pretty));
        assert!(person.validates(&json::parse(pretty).unwrap()));
        for bad in [
            r#"{"name":"Ada","age":36,"none":null,"ok":true,"score":1}"#,
            r#"{"name":"Ada","age":36.5,"none":null,"ok":true,"score":1,"tags":["a"]}"#,
            r#"{"name":"Ada","age":36,"none":null,"ok":"yes","score":1,"tags":["a"]}"#,
            r#"{"name":"Ada","age":36,"none":null,"ok":true,"score":1,"tags":["a","a","a"]}"#,
            r#"{"name":"Ada","age":36,"none":null,"ok":true,"score":1,"tags":["z"]}"#,
            r#"{"name":"123456789","age":36,"none":null,"ok":true,"score":1,"tags":["a"]}"#,
            r#"{"name":"Ada","age":36,"none":null,"ok":true,"score":1,"tags":["a"],"extra":1}"#,
            r#"["Ada"]"#,
        ] {
            assert!(!person.validates(&json::parse(bad).unwrap()), "{bad}");
        }
    }

    #[test]
    fn refuses_everything_else() {
        let person = grammar(PERSON);
        let good = r#"{"name":"Ada","age":36,"none":null,"ok":true,"score":1,"tags":["a"]}"#;
        assert!(whole(&person, good));
        for bad in [
            // 필드의 순서, 빠진 필드, 모르는 필드.
            r#"{"age":36"#,
            r#"{"name":"Ada","none""#,
            r#"{"name":"Ada","age":36,"extra""#,
            // 값의 종류.
            r#"{"name":7"#,
            r#"{"name":"Ada","age":"x"#,
            r#"{"name":"Ada","age":3.5"#,
            r#"{"name":"Ada","age":01"#,
            r#"{"name":"Ada","age":36,"none":0"#,
            r#"{"name":"Ada","age":36,"none":null,"ok":1"#,
            // 문자열의 길이, 제어 문자, 틀린 역슬래시, 틀린 UTF-8.
            r#"{"name":"123456789"#,
            "{\"name\":\"a\nb",
            r#"{"name":"\x"#,
            // enum과 배열의 개수.
            r#"{"name":"Ada","age":36,"none":null,"ok":true,"score":1,"tags":[]"#,
            r#"{"name":"Ada","age":36,"none":null,"ok":true,"score":1,"tags":["b""#,
            r#"{"name":"Ada","age":36,"none":null,"ok":true,"score":1,"tags":["a","a","#,
            // 빈칸은 쌍점과 쉼표 뒤에 하나만.
            r#" {"#,
            r#"{ "name""#,
            r#"{"name":  "#,
            r#"{"name":"Ada" ,"#,
        ] {
            assert!(!prefix(&person, bad), "{bad}");
        }
        let bytes = [b'{', b'"', b'n', b'a', b'm', b'e', b'"', b':', b'"', 0xff];
        assert!(!Matcher::new(&person).accepts(&bytes));
        // 앞부분은 앞부분일 뿐, 끝난 것이 아니다. 끝난 뒤에는 아무것도 못 온다.
        assert!(prefix(&person, &good[..20]) && !whole(&person, &good[..20]));
        assert!(!prefix(&person, &format!("{good} ")));
    }

    #[test]
    fn a_bare_number_may_end_without_a_closing_character() {
        let number = grammar(r#"{"type":"number"}"#);
        for (text, complete) in [
            ("-", false),
            ("-0", true),
            ("12.", false),
            ("12.5", true),
            ("1e", false),
            ("1e+3", true),
        ] {
            let mut matcher = Matcher::new(&number);
            assert!(matcher.consume(text.as_bytes()), "{text}");
            assert_eq!(matcher.complete(), complete, "{text}");
        }
        assert!(!prefix(&number, "00") && !prefix(&number, "1.e") && !prefix(&number, "+1"));
        // 자릿수에 한도가 있다. 끝없이 숫자만 쓰는 모델을 멈춘다.
        assert!(!prefix(&number, &"9".repeat(MAX_DIGITS as usize + 1)));
    }

    #[test]
    fn says_what_it_does_not_support() {
        for (schema, word) in [
            (
                r#"{"type":"object","properties":{"a":{"anyOf":[]}}}"#,
                "anyOf",
            ),
            (r#"{"type":"integer","minimum":0}"#, "minimum"),
            (r#"{"type":"string","pattern":"a+"}"#, "pattern"),
            (
                r#"{"type":"object","additionalProperties":true}"#,
                "additionalProperties",
            ),
            (r#"{}"#, "type"),
            (r#"{"type":"tuple"}"#, "tuple"),
            (r#"{"type":"array"}"#, "items"),
            (r#"{"enum":[1,2]}"#, "enum"),
            (
                r#"{"type":"object","properties":{},"required":["x"]}"#,
                "required",
            ),
            (
                r#"{"type":"array","items":{"type":"null"},"minItems":3,"maxItems":2}"#,
                "minItems",
            ),
        ] {
            let error = Grammar::from_schema(&json::parse(schema).unwrap()).unwrap_err();
            assert!(error.contains(word), "{error}");
        }
    }
}
