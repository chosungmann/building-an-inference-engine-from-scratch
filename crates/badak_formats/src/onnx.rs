//! ONNX 파일 (30장).
//!
//! ONNX 파일은 `ModelProto`라는 Protocol Buffers 메시지 하나다. 그 안에
//! 그래프가 있고, 그래프는 마디들과 가중치들(initializer)로 되어 있다.
//! `.tflite`(26장)와 견주면 이렇다.
//!
//! - 값을 번호가 아니라 **이름**으로 가리킨다. 마디의 입력과 출력은
//!   문자열이다.
//! - 연산을 번호가 아니라 이름(`"Conv"`)으로 적고, 연산의 설정은 이름이
//!   붙은 속성(attribute)들의 목록이다.
//! - 중간 값의 형상은 적혀 있지 않아도 된다. 읽는 쪽이 계산한다.
//!
//! 필드의 번호는 ONNX 저장소의 `onnx/onnx.proto`(1.23.2판)에서 옮겼다. 이
//! 책에 필요한 필드만 읽고 나머지는 건너뛴다.

use std::path::Path;
use std::sync::Arc;

use badak_core::{DType, Tensor};

use crate::error::{Error, Result};
use crate::mmap::Mmap;
use crate::protobuf::{Field, Message};

const FORMAT: &str = "ONNX";

/// 이 파서가 아는 가장 높은 IR 판. `onnx.proto`의 `IR_VERSION`이다.
pub const IR_VERSION: i64 = 13;

/// `TensorProto.DataType`의 번호들.
pub mod data_type {
    /// 32비트 실수.
    pub const FLOAT: i64 = 1;
    /// 32비트 정수.
    pub const INT32: i64 = 6;
    /// 64비트 정수.
    pub const INT64: i64 = 7;
}

// ANCHOR: types
/// 속성 하나의 값.
#[derive(Debug, Clone)]
pub enum Attribute {
    /// 정수 하나.
    Int(i64),
    /// 실수 하나.
    Float(f32),
    /// 정수 여러 개.
    Ints(Vec<i64>),
    /// 실수 여러 개.
    Floats(Vec<f32>),
    /// 문자열.
    String(String),
    /// 텐서.
    Tensor(Tensor),
    /// 이 파서가 읽지 않는 종류. `AttributeProto.AttributeType`의 번호다.
    Other(i64),
}

/// 마디 하나.
#[derive(Debug, Clone, Default)]
pub struct Node {
    /// 연산의 이름. `"Conv"`, `"Relu"` 같은 것이다.
    pub op_type: String,
    /// 연산이 속한 묶음(domain). 표준 연산은 빈 문자열이다.
    pub domain: String,
    /// 마디의 이름. 없어도 된다.
    pub name: String,
    /// 입력 값들의 이름. 빈 문자열은 생략한 입력이다.
    pub inputs: Vec<String>,
    /// 출력 값들의 이름.
    pub outputs: Vec<String>,
    /// 속성들.
    pub attributes: Vec<(String, Attribute)>,
}

/// 그래프의 입력이나 출력 하나.
#[derive(Debug, Clone, Default)]
pub struct ValueInfo {
    /// 값의 이름.
    pub name: String,
    /// 원소의 자료형. [`data_type`]의 번호다.
    pub elem_type: i64,
    /// 형상. 실행할 때 정해지는 축은 `None`이다.
    pub shape: Vec<Option<usize>>,
}

/// 그래프.
#[derive(Debug, Clone, Default)]
pub struct Graph {
    /// 이름.
    pub name: String,
    /// 마디들. 실행할 수 있는 순서로 적혀 있어야 한다.
    pub nodes: Vec<Node>,
    /// 값이 정해진 텐서들과 그 이름.
    pub initializers: Vec<(String, Tensor)>,
    /// 입력들.
    pub inputs: Vec<ValueInfo>,
    /// 출력들.
    pub outputs: Vec<ValueInfo>,
}

/// 파일 하나.
#[derive(Debug, Clone)]
pub struct Model {
    /// 파일 형식의 판(IR version).
    pub ir_version: i64,
    /// 표준 연산들의 판(opset). 같은 이름의 연산도 판에 따라 뜻이 다르다.
    pub opset: i64,
    /// 파일을 만든 프로그램.
    pub producer: String,
    /// 그래프.
    pub graph: Graph,
    file_len: usize,
}
// ANCHOR_END: types

fn invalid(offset: usize, message: impl Into<String>) -> Error {
    Error::Parse {
        format: FORMAT,
        offset,
        message: message.into(),
    }
}

type Owner = Arc<dyn AsRef<[u8]> + Send + Sync>;

impl Model {
    /// 파일을 메모리에 비춰 읽는다.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::parse(Arc::new(Mmap::open(path)?))
    }

    // ANCHOR: parse
    /// 바이트를 해석한다. `owner`는 그 바이트를 가진 것이고, 가중치 텐서가
    /// 이것을 빌려 쓴다.
    pub fn parse(owner: Owner) -> Result<Self> {
        let bytes = (*owner).as_ref();
        let mut model = Self {
            ir_version: 0,
            opset: 0,
            producer: String::new(),
            graph: Graph::default(),
            file_len: bytes.len(),
        };
        let mut graph = None;
        for field in Message::new(bytes, FORMAT) {
            let field = field?;
            match field.number {
                1 => model.ir_version = field.int(FORMAT)?,
                2 => model.producer = field.string(FORMAT)?.to_string(),
                3 => {
                    model.producer.push(' ');
                    model.producer.push_str(field.string(FORMAT)?);
                }
                7 => graph = Some(field),
                8 => {
                    // OperatorSetIdProto: 묶음의 이름(1)과 판(2).
                    let (mut domain, mut version) = ("", 0);
                    for inner in field.message(FORMAT)? {
                        let inner = inner?;
                        match inner.number {
                            1 => domain = inner.string(FORMAT)?,
                            2 => version = inner.int(FORMAT)?,
                            _ => {}
                        }
                    }
                    if domain.is_empty() || domain == "ai.onnx" {
                        model.opset = version;
                    }
                }
                _ => {}
            }
        }
        if !(1..=IR_VERSION).contains(&model.ir_version) {
            return Err(Error::Unsupported(format!(
                "ONNX IR {}판 ({IR_VERSION}판까지 읽습니다)",
                model.ir_version
            )));
        }
        let graph = graph.ok_or_else(|| invalid(0, "그래프가 없습니다"))?;
        model.graph = parse_graph(&graph, &owner)?;
        Ok(model)
    }
    // ANCHOR_END: parse

    /// 파일 없이, 손으로 지은 그래프로 모델을 만든다. 시험할 때 쓴다.
    pub fn from_graph(graph: Graph, opset: i64) -> Self {
        Self {
            ir_version: IR_VERSION,
            opset,
            producer: String::new(),
            graph,
            file_len: 0,
        }
    }

    /// 파일 전체의 크기(바이트).
    pub fn file_len(&self) -> usize {
        self.file_len
    }
}

fn parse_graph(field: &Field<'_>, owner: &Owner) -> Result<Graph> {
    let mut graph = Graph::default();
    for inner in field.message(FORMAT)? {
        let inner = inner?;
        match inner.number {
            1 => graph.nodes.push(parse_node(&inner, owner)?),
            2 => graph.name = inner.string(FORMAT)?.to_string(),
            5 => graph.initializers.push(parse_tensor(&inner, owner)?),
            11 => graph.inputs.push(parse_value_info(&inner)?),
            12 => graph.outputs.push(parse_value_info(&inner)?),
            15 => return Err(Error::Unsupported("성긴(sparse) 가중치".into())),
            _ => {}
        }
    }
    Ok(graph)
}

// ANCHOR: node
fn parse_node(field: &Field<'_>, owner: &Owner) -> Result<Node> {
    let mut node = Node::default();
    for inner in field.message(FORMAT)? {
        let inner = inner?;
        match inner.number {
            1 => node.inputs.push(inner.string(FORMAT)?.to_string()),
            2 => node.outputs.push(inner.string(FORMAT)?.to_string()),
            3 => node.name = inner.string(FORMAT)?.to_string(),
            4 => node.op_type = inner.string(FORMAT)?.to_string(),
            5 => node.attributes.push(parse_attribute(&inner, owner)?),
            7 => node.domain = inner.string(FORMAT)?.to_string(),
            _ => {}
        }
    }
    Ok(node)
}

fn parse_attribute(field: &Field<'_>, owner: &Owner) -> Result<(String, Attribute)> {
    let mut name = String::new();
    let mut kind = 0;
    let (mut int, mut float, mut string, mut tensor) = (0, 0.0, String::new(), None);
    let (mut ints, mut floats) = (Vec::new(), Vec::new());
    for inner in field.message(FORMAT)? {
        let inner = inner?;
        match inner.number {
            1 => name = inner.string(FORMAT)?.to_string(),
            2 => float = inner.float(FORMAT)?,
            3 => int = inner.int(FORMAT)?,
            4 => string = inner.string(FORMAT)?.to_string(),
            5 => tensor = Some(parse_tensor(&inner, owner)?.1),
            7 => inner.floats(&mut floats, FORMAT)?,
            8 => inner.ints(&mut ints, FORMAT)?,
            20 => kind = inner.int(FORMAT)?,
            _ => {}
        }
    }
    // 어느 필드에 값이 들었는지는 `type` 필드(20번)가 말해 준다.
    let value = match kind {
        1 => Attribute::Float(float),
        2 => Attribute::Int(int),
        3 => Attribute::String(string),
        4 => Attribute::Tensor(
            tensor
                .ok_or_else(|| invalid(field.offset, format!("속성 {name:?}에 텐서가 없습니다")))?,
        ),
        6 => Attribute::Floats(floats),
        7 => Attribute::Ints(ints),
        other => Attribute::Other(other),
    };
    Ok((name, value))
}
// ANCHOR_END: node

// ANCHOR: tensor
/// `TensorProto` 하나를 읽는다. 이름과 텐서를 돌려준다.
///
/// 값은 두 가지로 적힐 수 있다. 자료형마다 따로 있는 되풀이 필드
/// (`float_data`, `int64_data`)에 숫자로 적거나, `raw_data`에 리틀 엔디언
/// 바이트로 적거나. 큰 가중치는 뒤쪽이다. `raw_data`는 복사하지 않고 파일을
/// 빌려 쓴다. 다만 Protocol Buffers는 바이트의 시작을 4바이트 경계에
/// 맞춰 주지 않으므로, 맞지 않으면 [`Tensor::from_bytes`]가 복사한다.
fn parse_tensor(field: &Field<'_>, owner: &Owner) -> Result<(String, Tensor)> {
    let mut name = String::new();
    let (mut dims, mut kind) = (Vec::new(), 0);
    let (mut floats, mut ints) = (Vec::new(), Vec::new());
    let mut raw = None;
    for inner in field.message(FORMAT)? {
        let inner = inner?;
        match inner.number {
            1 => inner.ints(&mut dims, FORMAT)?,
            2 => kind = inner.int(FORMAT)?,
            4 => inner.floats(&mut floats, FORMAT)?,
            // int32_data(5)와 int64_data(7).
            5 | 7 => inner.ints(&mut ints, FORMAT)?,
            8 => name = inner.string(FORMAT)?.to_string(),
            9 => raw = Some((inner.offset, inner.bytes(FORMAT)?)),
            13 => {
                return Err(Error::Unsupported(format!(
                    "값을 다른 파일에 둔 텐서 {name:?}"
                )));
            }
            14 if inner.int(FORMAT)? != 0 => {
                return Err(Error::Unsupported("값을 다른 파일에 둔 텐서".into()));
            }
            _ => {}
        }
    }
    let bad = |message: String| invalid(field.offset, format!("텐서 {name:?}: {message}"));
    let shape = dims
        .iter()
        .map(|&dim| usize::try_from(dim))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|_| bad("형상에 음수가 있습니다".into()))?;
    // 32비트를 넘는 값은 32비트의 끝값으로 자른다. "끝까지"를 뜻하려고
    // 64비트의 가장 큰 수를 적어 두는 파일이 있다(`Slice`의 끝).
    let narrow = |value: i64| -> Result<i32> {
        Ok(value.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32)
    };
    let tensor = match (kind, raw) {
        (data_type::FLOAT, Some((offset, bytes))) => Tensor::from_bytes(
            Arc::clone(owner),
            offset..offset + bytes.len(),
            DType::F32,
            &shape,
        ),
        (data_type::FLOAT, None) => Tensor::from_vec(floats, &shape),
        (data_type::INT32, Some((offset, bytes))) => Tensor::from_bytes(
            Arc::clone(owner),
            offset..offset + bytes.len(),
            DType::I32,
            &shape,
        ),
        // 이 책의 텐서에는 64비트 정수가 없다. 형상이나 축의 번호로 쓰이는
        // 값들이므로 32비트로 줄여 담는다.
        (data_type::INT64, Some((_, bytes))) => {
            let (chunks, rest) = bytes.as_chunks::<8>();
            if !rest.is_empty() {
                return Err(bad("바이트 수가 8의 배수가 아닙니다".into()));
            }
            let values = chunks
                .iter()
                .map(|&chunk| narrow(i64::from_le_bytes(chunk)))
                .collect::<Result<Vec<_>>>()?;
            Tensor::from_vec(values, &shape)
        }
        (data_type::INT32 | data_type::INT64, None) => {
            let values = ints.into_iter().map(narrow).collect::<Result<Vec<_>>>()?;
            Tensor::from_vec(values, &shape)
        }
        (other, _) => {
            return Err(Error::Unsupported(format!(
                "자료형이 {other}번인 텐서 {name:?}"
            )));
        }
    };
    let tensor = tensor.map_err(|error| bad(error.to_string()))?;
    Ok((name, tensor))
}
// ANCHOR_END: tensor

/// `ValueInfoProto`: 이름(1)과 자료형(2). 자료형 안에 텐서(1)가, 그 안에
/// 원소의 자료형(1)과 형상(2)이, 형상 안에 축들(1)이, 축 안에 길이(1)나
/// 이름(2)이 있다.
fn parse_value_info(field: &Field<'_>) -> Result<ValueInfo> {
    let mut info = ValueInfo::default();
    for inner in field.message(FORMAT)? {
        let inner = inner?;
        match inner.number {
            1 => info.name = inner.string(FORMAT)?.to_string(),
            2 => {
                for kind in inner.message(FORMAT)? {
                    let kind = kind?;
                    if kind.number != 1 {
                        return Err(Error::Unsupported(format!(
                            "텐서가 아닌 값 {:?}",
                            info.name
                        )));
                    }
                    for tensor in kind.message(FORMAT)? {
                        let tensor = tensor?;
                        match tensor.number {
                            1 => info.elem_type = tensor.int(FORMAT)?,
                            2 => info.shape = parse_shape(&tensor)?,
                            _ => {}
                        }
                    }
                }
            }
            _ => {}
        }
    }
    Ok(info)
}

fn parse_shape(field: &Field<'_>) -> Result<Vec<Option<usize>>> {
    let mut shape = Vec::new();
    for dim in field.message(FORMAT)? {
        let dim = dim?;
        if dim.number != 1 {
            continue;
        }
        let mut length = None;
        for inner in dim.message(FORMAT)? {
            let inner = inner?;
            if inner.number == 1 {
                length = usize::try_from(inner.int(FORMAT)?).ok();
            }
        }
        shape.push(length);
    }
    Ok(shape)
}

impl Node {
    /// 이름으로 속성을 찾는다.
    pub fn attribute(&self, name: &str) -> Option<&Attribute> {
        self.attributes
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value)
    }

    fn mismatch(&self, name: &str, expected: &str) -> Error {
        invalid(
            0,
            format!(
                "{} 마디 {:?}의 속성 {name:?}이 {expected}가 아닙니다",
                self.op_type, self.name
            ),
        )
    }

    /// 정수 속성. 없으면 `default`다.
    pub fn int(&self, name: &str, default: i64) -> Result<i64> {
        match self.attribute(name) {
            None => Ok(default),
            Some(Attribute::Int(value)) => Ok(*value),
            Some(_) => Err(self.mismatch(name, "정수")),
        }
    }

    /// 실수 속성. 없으면 `default`다.
    pub fn float(&self, name: &str, default: f32) -> Result<f32> {
        match self.attribute(name) {
            None => Ok(default),
            Some(Attribute::Float(value)) => Ok(*value),
            Some(_) => Err(self.mismatch(name, "실수")),
        }
    }

    /// 음수가 아닌 정수 여러 개인 속성. 없으면 `None`이다.
    pub fn sizes(&self, name: &str) -> Result<Option<Vec<usize>>> {
        match self.attribute(name) {
            None => Ok(None),
            Some(Attribute::Ints(values)) => values
                .iter()
                .map(|&value| usize::try_from(value))
                .collect::<std::result::Result<Vec<_>, _>>()
                .map(Some)
                .map_err(|_| self.mismatch(name, "음수가 아닌 정수들")),
            Some(_) => Err(self.mismatch(name, "정수들")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 시험용 파일을 손으로 짓는 도우미들. 필드 하나를 바이트로 적는다.
    fn varint(mut value: u64, out: &mut Vec<u8>) {
        while value >= 0x80 {
            out.push((value & 0x7f) as u8 | 0x80);
            value >>= 7;
        }
        out.push(value as u8);
    }

    fn int(number: u32, value: i64) -> Vec<u8> {
        let mut out = Vec::new();
        varint(u64::from(number) << 3, &mut out);
        varint(value.cast_unsigned(), &mut out);
        out
    }

    fn bytes(number: u32, value: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        varint(u64::from(number) << 3 | 2, &mut out);
        varint(value.len() as u64, &mut out);
        out.extend_from_slice(value);
        out
    }

    fn float(number: u32, value: f32) -> Vec<u8> {
        let mut out = Vec::new();
        varint(u64::from(number) << 3 | 5, &mut out);
        out.extend(value.to_le_bytes());
        out
    }

    fn value_info(name: &str, shape: &[i64]) -> Vec<u8> {
        let dims: Vec<u8> = shape
            .iter()
            .flat_map(|&dim| {
                // 음수는 "이름만 있는 축"으로 적는다.
                let dim = if dim < 0 { bytes(2, b"n") } else { int(1, dim) };
                bytes(1, &dim)
            })
            .collect();
        let tensor = [int(1, data_type::FLOAT), bytes(2, &dims)].concat();
        [bytes(1, name.as_bytes()), bytes(2, &bytes(1, &tensor))].concat()
    }

    /// `y = Relu(x) + w`를 하는 작은 모델.
    fn small_model(ir_version: i64) -> Vec<u8> {
        let weights: Vec<u8> = [1.0_f32, -2.0, 3.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let initializer = [
            int(1, 3),
            int(2, data_type::FLOAT),
            bytes(8, b"w"),
            bytes(9, &weights),
        ]
        .concat();
        let relu = [
            bytes(1, b"x"),
            bytes(2, b"h"),
            bytes(3, b"relu"),
            bytes(4, b"Relu"),
        ]
        .concat();
        let attributes = [
            [bytes(1, b"alpha"), float(2, 0.5), int(20, 1)].concat(),
            [bytes(1, b"axis"), int(3, -1), int(20, 2)].concat(),
            [bytes(1, b"pads"), int(8, 1), int(8, 2), int(20, 7)].concat(),
            [bytes(1, b"mode"), bytes(4, b"same"), int(20, 3)].concat(),
            // 텐서 속성. 값을 int64_data에 적는다.
            [
                bytes(1, b"value"),
                bytes(
                    5,
                    &[int(1, 2), int(2, data_type::INT64), int(7, 7), int(7, -8)].concat(),
                ),
                int(20, 4),
            ]
            .concat(),
            [bytes(1, b"body"), int(20, 5)].concat(),
        ];
        let mut add = [
            bytes(1, b"h"),
            bytes(1, b"w"),
            bytes(2, b"y"),
            bytes(4, b"Add"),
        ]
        .concat();
        for attribute in &attributes {
            add.extend(bytes(5, attribute));
        }
        let graph = [
            bytes(1, &relu),
            bytes(1, &add),
            bytes(2, b"small"),
            bytes(5, &initializer),
            bytes(11, &value_info("x", &[-1, 3])),
            bytes(12, &value_info("y", &[-1, 3])),
        ]
        .concat();
        let opset = [bytes(1, b""), int(2, 18)].concat();
        let other = [bytes(1, b"com.example"), int(2, 3)].concat();
        [
            int(1, ir_version),
            bytes(2, b"test"),
            bytes(3, b"1.0"),
            bytes(7, &graph),
            bytes(8, &opset),
            bytes(8, &other),
        ]
        .concat()
    }

    #[test]
    fn reads_a_small_model() {
        let model = Model::parse(Arc::new(small_model(8))).unwrap();
        assert_eq!((model.ir_version, model.opset), (8, 18));
        assert_eq!(model.producer, "test 1.0");
        let graph = &model.graph;
        assert_eq!(graph.name, "small");
        assert_eq!(graph.inputs[0].name, "x");
        assert_eq!(graph.inputs[0].shape, [None, Some(3)]);
        assert_eq!(graph.outputs[0].elem_type, data_type::FLOAT);
        let (name, tensor) = &graph.initializers[0];
        assert_eq!(name, "w");
        assert_eq!(tensor.to_vec::<f32>().unwrap(), [1.0, -2.0, 3.0]);

        let [relu, add] = &graph.nodes[..] else {
            panic!("마디가 둘이어야 한다");
        };
        assert_eq!(
            (relu.op_type.as_str(), relu.name.as_str()),
            ("Relu", "relu")
        );
        assert_eq!(add.inputs, ["h", "w"]);
        assert_eq!(add.float("alpha", 0.0).unwrap(), 0.5);
        assert_eq!(add.float("beta", 2.0).unwrap(), 2.0);
        assert_eq!(add.int("axis", 0).unwrap(), -1);
        assert_eq!(add.sizes("pads").unwrap().unwrap(), [1, 2]);
        assert!(add.sizes("strides").unwrap().is_none());
        assert!(add.int("alpha", 0).is_err());
        assert!(matches!(add.attribute("mode"), Some(Attribute::String(mode)) if mode == "same"));
        let Some(Attribute::Tensor(value)) = add.attribute("value") else {
            panic!("텐서 속성이어야 한다");
        };
        assert_eq!(value.to_vec::<i32>().unwrap(), [7, -8]);
        assert!(matches!(add.attribute("body"), Some(Attribute::Other(5))));
    }

    #[test]
    fn rejects_what_it_cannot_read() {
        // 모르는 판.
        assert!(matches!(
            Model::parse(Arc::new(small_model(IR_VERSION + 1))),
            Err(Error::Unsupported(_))
        ));
        // 그래프가 없다.
        assert!(Model::parse(Arc::new(int(1, 8))).is_err());
        // 어디서 잘라도 오류이지 공황이 아니다.
        let whole = small_model(8);
        for len in 0..whole.len() {
            let _ = Model::parse(Arc::new(whole[..len].to_vec()));
        }
        // 형상과 값의 수가 다른 텐서.
        let tensor = [int(1, 4), int(2, data_type::FLOAT), bytes(9, &[0; 12])].concat();
        let graph = bytes(5, &tensor);
        let model = [int(1, 8), bytes(7, &graph)].concat();
        assert!(Model::parse(Arc::new(model)).is_err());
        // 값을 다른 파일에 둔 텐서.
        let tensor = [int(2, data_type::FLOAT), int(14, 1)].concat();
        let model = [int(1, 8), bytes(7, &bytes(5, &tensor))].concat();
        assert!(matches!(
            Model::parse(Arc::new(model)),
            Err(Error::Unsupported(_))
        ));
    }
}
