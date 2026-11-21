//! 그래프의 조각을 Core ML에 넘긴다 (45장).
//!
//! Core ML은 Apple의 기기에서 모델을 돌리는 운영체제의 기능이다. 모델을
//! 주면 CPU, GPU, 뉴럴 엔진(Apple의 NPU) 가운데 어디에서 돌릴지를 스스로
//! 정한다. 뉴럴 엔진을 쓰는 길은 이것뿐이다.
//!
//! 이 모듈은 둘로 나뉜다.
//!
//! - **옮기기**([`lower`]): 조각을 Core ML의 모델 파일(`.mlmodel`)로 적는다.
//!   파일은 Protocol Buffers 메시지 하나다. 순수한 Rust이고 어느 기계에서나
//!   돈다.
//! - **돌리기**([`CoreMl`]): 그 파일을 Core ML에 주고 실행한다. Core ML은
//!   Swift와 Objective-C로만 부를 수 있어서, Swift로 쓴 작은 동적
//!   라이브러리(`apps/coreml`)를 실행할 때 올려서 부른다. macOS에서만 된다.
//!
//! 필드의 번호는 coremltools 저장소의 `mlmodel/format/*.proto`에서 왔다
//! (커밋 `a207b1a3452e`).

use std::path::PathBuf;

use badak_formats::protobuf::Writer;

use crate::delegate::Segment;
use crate::error::{Error, Result};
use crate::graph::{Activation, Graph, Node, Op, Padding};

fn unsupported(message: impl Into<String>) -> Error {
    Error::Unsupported(message.into())
}

/// 조각의 입력이나 출력 하나가 Core ML 쪽에서 어떤 모습인지.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Port {
    /// 그래프에서의 값의 번호.
    pub value: usize,
    /// Core ML에서의 이름.
    pub name: String,
    /// Core ML에서의 형상. 늘 네 축이다. `(1, 채널, 높이, 너비)`.
    pub shape: [usize; 4],
    /// 그래프의 값이 `(1, 높이, 너비, 채널)`의 순서여서 넘길 때 축을 바꿔야
    /// 하는지.
    pub channels_last: bool,
}

/// Core ML의 모델 파일로 옮긴 조각.
#[derive(Debug, Clone)]
pub struct Lowered {
    /// `.mlmodel` 파일의 바이트.
    pub model: Vec<u8>,
    /// 입력들. [`Segment::inputs`]와 순서가 같다.
    pub inputs: Vec<Port>,
    /// 출력들. [`Segment::outputs`]와 순서가 같다.
    pub outputs: Vec<Port>,
    /// 적은 층의 수.
    pub layers: usize,
}

/// 맨 앞의 `(1, 3, 높이, 너비)`를 `(1, 높이, 너비, 3)`으로 바꾸는 축 바꾸기.
/// Core ML은 채널이 앞에 오는 레이아웃을 쓰므로, 이 노드는 할 일이 없어진다.
fn is_layout_transpose(graph: &Graph, node: &Node) -> bool {
    let Op::Transpose { perm } = &node.op else {
        return false;
    };
    let input = node.inputs[0];
    let readers = graph
        .nodes
        .iter()
        .filter(|other| other.inputs.contains(&input))
        .count();
    perm == &[0, 2, 3, 1] && graph.inputs.contains(&input) && readers == 1
}

fn activation_ok(activation: Activation) -> bool {
    matches!(
        activation,
        Activation::None | Activation::Relu | Activation::Relu6 | Activation::ReluN1To1
    )
}

// ANCHOR: supports
/// 이 노드를 Core ML의 층으로 옮길 수 있는지.
///
/// 연산의 이름만 보지 않는다. 같은 합성곱이라도 필터가 가중치가 아니라
/// 계산되는 값이면 옮기지 못하고, 패딩도 높이와 너비에만 하는 것만
/// 옮긴다. 실전의 델리게이트도 이렇게 노드마다 조건을 본다.
pub fn supports(graph: &Graph, node: &Node) -> bool {
    let constant = |position: usize| {
        node.inputs
            .get(position)
            .is_some_and(|&value| graph.values[value].constant.is_some())
    };
    // 조각을 드나들 수 있는 값은 `(1, 높이, 너비, 채널)`이나 `(1, 길이)`다.
    let plain = |&value: &usize| {
        let shape = &graph.values[value].shape;
        graph.values[value].constant.is_some() || (matches!(shape.len(), 2 | 4) && shape[0] == 1)
    };
    if !node.inputs.iter().chain(&node.outputs).all(plain) {
        return false;
    }
    let rank = graph.values[node.inputs[0]].shape.len();
    match &node.op {
        Op::Transpose { .. } => is_layout_transpose(graph, node),
        Op::Conv2d { activation, .. } | Op::DepthwiseConv2d { activation, .. } => {
            constant(1) && (node.inputs.len() == 2 || constant(2)) && activation_ok(*activation)
        }
        Op::FullyConnected { activation, .. } => {
            rank == 2
                && constant(1)
                && (node.inputs.len() == 2 || constant(2))
                && activation_ok(*activation)
        }
        Op::Add { activation } => {
            !constant(0)
                && !constant(1)
                && graph.values[node.inputs[0]].shape == graph.values[node.inputs[1]].shape
                && activation_ok(*activation)
        }
        Op::Pad { amounts, .. } => rank == 4 && amounts[0] == [0, 0] && amounts[3] == [0, 0],
        Op::Mean { axes, .. } => {
            let mut axes = axes.clone();
            axes.sort_unstable();
            rank == 4 && axes == [1, 2]
        }
        Op::MaxPool2d { .. } | Op::AveragePool2d { .. } => rank == 4,
        Op::Relu | Op::Clip { .. } => true,
        _ => false,
    }
}
// ANCHOR_END: supports

/// 층 하나를 적는다. `fill`이 층의 종류에 따른 내용을 채운다.
fn layer(
    network: &mut Writer,
    name: &str,
    inputs: &[&str],
    output: &str,
    kind: u32,
    params: &Writer,
) {
    let mut layer = Writer::new();
    layer.string(1, name);
    for input in inputs {
        layer.string(2, input);
    }
    layer.string(3, output);
    layer.message(kind, params);
    network.message(1, &layer);
}

/// 가장자리에 덧댈 길이들. 높이가 먼저, 너비가 다음이다.
fn borders(amounts: [[usize; 2]; 2]) -> Writer {
    let mut all = Writer::new();
    for [before, after] in amounts {
        let mut edge = Writer::new();
        edge.varint(1, before as u64).varint(2, after as u64);
        all.message(10, &edge);
    }
    all
}

/// 합성곱과 풀링이 함께 쓰는 패딩의 종류. `valid`와 `same`의 필드 번호를
/// 받는다.
fn padding(params: &mut Writer, padding: Padding, valid: u32, same: u32) {
    match padding {
        Padding::Valid => {
            params.message(valid, &Writer::new());
        }
        Padding::Explicit(amounts) => {
            let mut inner = Writer::new();
            inner.message(1, &borders(amounts));
            params.message(valid, &inner);
        }
        // 뒤쪽(아래와 오른쪽)에 하나 더 패딩하는 방식. `.tflite`의 SAME과 같다.
        Padding::Same => {
            let mut inner = Writer::new();
            inner.varint(1, 0);
            params.message(same, &inner);
        }
    }
}

fn weights(values: impl IntoIterator<Item = f32>) -> Writer {
    let mut params = Writer::new();
    params.floats(1, values);
    params
}

/// 조각을 옮기는 동안의 상태.
struct Lowering<'a> {
    graph: &'a Graph,
    network: Writer,
    /// 값마다 Core ML에서의 이름. 축 바꾸기 노드는 입력의 이름을 그대로
    /// 물려준다.
    blob: Vec<Option<String>>,
    layers: usize,
}

impl Lowering<'_> {
    fn name(&self, value: usize) -> Result<&str> {
        self.blob[value]
            .as_deref()
            .ok_or_else(|| unsupported("조각 밖의 값을 읽는 노드입니다"))
    }

    fn floats(&self, value: usize) -> Result<&[f32]> {
        let tensor = self.graph.values[value]
            .constant
            .as_ref()
            .ok_or_else(|| unsupported("가중치가 계산되는 값입니다"))?;
        Ok(tensor.as_slice::<f32>()?)
    }

    fn push(&mut self, name: &str, inputs: &[&str], output: &str, kind: u32, params: &Writer) {
        layer(&mut self.network, name, inputs, output, kind, params);
        self.layers += 1;
    }

    /// 층의 결과에 활성화 함수를 잇는다. 층이 적을 이름을 돌려준다.
    /// 활성화가 있으면 층은 임시 이름에 적고, 활성화의 층이 진짜 이름에 적는다.
    fn before_activation(&self, output: usize, activation: Activation) -> String {
        match activation {
            Activation::None => format!("v{output}"),
            _ => format!("v{output}_linear"),
        }
    }

    fn activate(&mut self, output: usize, activation: Activation) -> Result<()> {
        let (from, to) = (format!("v{output}_linear"), format!("v{output}"));
        let name = format!("activation{output}");
        match activation {
            Activation::None => {}
            Activation::Relu => {
                let mut params = Writer::new();
                params.message(10, &Writer::new());
                self.push(&name, &[&from], &to, 130, &params);
            }
            Activation::Relu6 | Activation::ReluN1To1 => {
                let (low, high) = if activation == Activation::Relu6 {
                    (0.0, 6.0)
                } else {
                    (-1.0, 1.0)
                };
                let mut params = Writer::new();
                params.float(1, low).float(2, high);
                self.push(&name, &[&from], &to, 660, &params);
            }
            other => return Err(unsupported(format!("활성화 함수 {other:?}"))),
        }
        Ok(())
    }

    // ANCHOR: lower_conv
    /// 합성곱. 깊이별 합성곱은 "묶음의 수가 채널의 수와 같은 합성곱"이다.
    fn convolution(&mut self, index: usize, node: &Node) -> Result<()> {
        let (padding_kind, stride, dilation, activation, depthwise) = match node.op {
            Op::Conv2d {
                padding,
                stride,
                dilation,
                activation,
            } => (padding, stride, dilation, activation, false),
            Op::DepthwiseConv2d {
                padding,
                stride,
                dilation,
                activation,
                ..
            } => (padding, stride, dilation, activation, true),
            _ => unreachable!("부르는 쪽이 합성곱만 넘긴다"),
        };
        let filter = self.floats(node.inputs[1])?;
        let &[first, height, width, last] = &self.graph.values[node.inputs[1]].shape[..] else {
            return Err(unsupported("필터가 네 축이 아닙니다"));
        };
        let input_channels = self.graph.values[node.inputs[0]].shape[3];
        // 우리 필터는 (출력, 높이, 너비, 입력)이고 깊이별은 (1, 높이, 너비,
        // 출력)이다. Core ML은 (출력, 입력 / 묶음, 높이, 너비)를 받는다.
        let (outputs, kernel_channels, groups) = if depthwise {
            (last, 1, input_channels)
        } else {
            (first, last, 1)
        };
        let spatial = height * width;
        let reordered = (0..outputs).flat_map(move |output| {
            (0..kernel_channels).flat_map(move |channel| {
                (0..spatial).map(move |position| {
                    if depthwise {
                        filter[position * outputs + output]
                    } else {
                        filter[(output * spatial + position) * kernel_channels + channel]
                    }
                })
            })
        });
        let mut params = Writer::new();
        params
            .varint(1, outputs as u64)
            .varint(2, kernel_channels as u64)
            .varint(10, groups as u64)
            .varints(20, &[height as u64, width as u64])
            .varints(30, &[stride[0] as u64, stride[1] as u64])
            .varints(40, &[dilation[0] as u64, dilation[1] as u64]);
        padding(&mut params, padding_kind, 50, 51);
        params.message(90, &weights(reordered));
        if let Some(&bias) = node.inputs.get(2) {
            params.varint(70, 1);
            params.message(91, &weights(self.floats(bias)?.iter().copied()));
        }
        let output = node.outputs[0];
        let input = self.name(node.inputs[0])?.to_owned();
        let target = self.before_activation(output, activation);
        self.push(&format!("conv{index}"), &[&input], &target, 100, &params);
        self.activate(output, activation)
    }
    // ANCHOR_END: lower_conv

    fn pooling(&mut self, index: usize, node: &Node) -> Result<()> {
        let mut params = Writer::new();
        match &node.op {
            Op::Mean { .. } => {
                params.varint(1, 1).varint(60, 1);
                padding(&mut params, Padding::Valid, 30, 31);
            }
            Op::MaxPool2d {
                padding: kind,
                stride,
                filter,
            }
            | Op::AveragePool2d {
                padding: kind,
                stride,
                filter,
            } => {
                let average = matches!(node.op, Op::AveragePool2d { .. });
                params
                    .varint(1, u64::from(average))
                    .varints(10, &[filter[0] as u64, filter[1] as u64])
                    .varints(20, &[stride[0] as u64, stride[1] as u64]);
                padding(&mut params, *kind, 30, 31);
                // 패딩한 자리는 평균에 넣지 않는다. `.tflite`가 그렇게 한다.
                params.varint(50, 1);
            }
            _ => unreachable!("부르는 쪽이 풀링만 넘긴다"),
        }
        let input = self.name(node.inputs[0])?.to_owned();
        self.push(
            &format!("pool{index}"),
            &[&input],
            &format!("v{}", node.outputs[0]),
            120,
            &params,
        );
        Ok(())
    }

    fn node(&mut self, index: usize, node: &Node) -> Result<()> {
        if !supports(self.graph, node) {
            return Err(unsupported(format!(
                "Core ML로 옮길 수 없는 노드 {}",
                node.op.name()
            )));
        }
        let output = node.outputs[0];
        let input = self.name(node.inputs[0])?.to_owned();
        let target = format!("v{output}");
        match &node.op {
            Op::Transpose { .. } => {
                self.blob[output] = Some(input);
                return Ok(());
            }
            Op::Conv2d { .. } | Op::DepthwiseConv2d { .. } => self.convolution(index, node)?,
            Op::Mean { .. } | Op::MaxPool2d { .. } | Op::AveragePool2d { .. } => {
                self.pooling(index, node)?
            }
            Op::FullyConnected { activation, .. } => {
                let &[rows, cols] = &self.graph.values[node.inputs[1]].shape[..] else {
                    return Err(unsupported("선형 층의 가중치가 행렬이 아닙니다"));
                };
                let mut params = Writer::new();
                params.varint(1, cols as u64).varint(2, rows as u64);
                params.message(20, &weights(self.floats(node.inputs[1])?.iter().copied()));
                if let Some(&bias) = node.inputs.get(2) {
                    params.varint(10, 1);
                    params.message(21, &weights(self.floats(bias)?.iter().copied()));
                }
                let target = self.before_activation(output, *activation);
                self.push(&format!("linear{index}"), &[&input], &target, 140, &params);
                self.activate(output, *activation)?;
            }
            Op::Add { activation } => {
                let other = self.name(node.inputs[1])?.to_owned();
                let target = self.before_activation(output, *activation);
                self.push(
                    &format!("add{index}"),
                    &[&input, &other],
                    &target,
                    230,
                    &Writer::new(),
                );
                self.activate(output, *activation)?;
            }
            Op::Pad { amounts, value } => {
                let mut constant = Writer::new();
                constant.float(1, *value);
                let mut params = Writer::new();
                params
                    .message(1, &constant)
                    .message(10, &borders([amounts[1], amounts[2]]));
                self.push(&format!("pad{index}"), &[&input], &target, 200, &params);
            }
            Op::Relu => {
                let mut params = Writer::new();
                params.message(10, &Writer::new());
                self.push(&format!("relu{index}"), &[&input], &target, 130, &params);
            }
            Op::Clip { min, max } => {
                let mut params = Writer::new();
                params.float(1, *min).float(2, *max);
                self.push(&format!("clip{index}"), &[&input], &target, 660, &params);
            }
            other => {
                return Err(unsupported(format!(
                    "Core ML로 옮길 수 없는 노드 {}",
                    other.name()
                )));
            }
        }
        self.blob[output] = Some(target);
        Ok(())
    }
}

/// 값 하나를 Core ML의 입력이나 출력으로 적는다.
fn feature(port: &Port) -> Writer {
    let mut array = Writer::new();
    array.varints(1, &port.shape.map(|extent| extent as u64));
    // FLOAT32.
    array.varint(2, 65568);
    let mut kind = Writer::new();
    kind.message(5, &array);
    let mut feature = Writer::new();
    feature.string(1, &port.name).message(3, &kind);
    feature
}

// ANCHOR: lower
/// 조각을 Core ML의 모델 파일로 옮긴다.
pub fn lower(graph: &Graph, segment: &Segment) -> Result<Lowered> {
    let mut lowering = Lowering {
        graph,
        network: Writer::new(),
        blob: vec![None; graph.values.len()],
        layers: 0,
    };
    let port = |value: usize, name: String, raw: bool| -> Result<Port> {
        let shape = &graph.values[value].shape;
        Ok(match shape[..] {
            // 맨 앞의 축 바꾸기가 읽는 입력. 이미 채널이 앞에 있다.
            [1, channels, height, width] if raw => Port {
                value,
                name,
                shape: [1, channels, height, width],
                channels_last: false,
            },
            [1, height, width, channels] => Port {
                value,
                name,
                shape: [1, channels, height, width],
                channels_last: true,
            },
            [1, length] => Port {
                value,
                name,
                shape: [1, length, 1, 1],
                channels_last: false,
            },
            _ => {
                return Err(unsupported(format!(
                    "형상 {shape:?}인 값은 조각을 드나들 수 없습니다"
                )));
            }
        })
    };
    let mut inputs = Vec::new();
    for &value in &segment.inputs {
        let raw = segment.nodes.iter().any(|&index| {
            is_layout_transpose(graph, &graph.nodes[index]) && graph.nodes[index].inputs[0] == value
        });
        let name = format!("v{value}");
        lowering.blob[value] = Some(name.clone());
        inputs.push(port(value, name, raw)?);
    }
    for &index in &segment.nodes {
        lowering.node(index, &graph.nodes[index])?;
    }
    let mut outputs = Vec::new();
    for &value in &segment.outputs {
        let name = lowering.name(value)?.to_owned();
        if inputs.iter().any(|input| input.name == name) {
            return Err(unsupported("조각의 출력이 입력 그대로입니다"));
        }
        outputs.push(port(value, name, false)?);
    }

    // 모델 = 버전 + 설명(입력과 출력) + 신경망(층들).
    let mut description = Writer::new();
    for input in &inputs {
        description.message(1, &feature(input));
    }
    for output in &outputs {
        description.message(10, &feature(output));
    }
    // 입력의 형상을 적힌 그대로 쓴다(EXACT_ARRAY_MAPPING). 버전 4부터 된다.
    lowering.network.varint(5, 1);
    let mut model = Writer::new();
    model
        .varint(1, 4)
        .message(2, &description)
        .message(500, &lowering.network);
    Ok(Lowered {
        model: model.into_bytes(),
        inputs,
        outputs,
        layers: lowering.layers,
    })
}
// ANCHOR_END: lower

// ANCHOR: layout
/// `(높이, 너비, 채널)`의 값을 `(채널, 높이, 너비)`의 순서로 옮겨 적는다.
pub fn to_channels_first(port: &Port, from: &[f32], into: &mut [f32]) {
    let [_, channels, height, width] = port.shape;
    let plane = height * width;
    for (position, pixel) in from.chunks_exact(channels).enumerate() {
        for (channel, &value) in pixel.iter().enumerate() {
            into[channel * plane + position] = value;
        }
    }
}

/// [`to_channels_first`]의 반대.
pub fn to_channels_last(port: &Port, from: &[f32], into: &mut [f32]) {
    let [_, channels, height, width] = port.shape;
    let plane = height * width;
    for (position, pixel) in into.chunks_exact_mut(channels).enumerate() {
        for (channel, value) in pixel.iter_mut().enumerate() {
            *value = from[channel * plane + position];
        }
    }
}
// ANCHOR_END: layout

/// Core ML이 쓸 수 있는 계산 장치의 범위. 값은 Core ML의 `MLComputeUnits`와
/// 같다.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Units {
    /// CPU만.
    CpuOnly = 0,
    /// CPU와 GPU.
    CpuAndGpu = 1,
    /// 모두. Core ML이 고른다.
    All = 2,
    /// CPU와 뉴럴 엔진.
    CpuAndNeuralEngine = 3,
}

impl Units {
    /// 넷 모두.
    pub const ALL: [Self; 4] = [
        Self::CpuOnly,
        Self::CpuAndGpu,
        Self::CpuAndNeuralEngine,
        Self::All,
    ];

    /// 사람이 읽는 이름.
    pub fn label(self) -> &'static str {
        match self {
            Self::CpuOnly => "CPU만",
            Self::CpuAndGpu => "CPU와 GPU",
            Self::All => "모두",
            Self::CpuAndNeuralEngine => "CPU와 뉴럴 엔진",
        }
    }
}

/// Swift로 쓴 라이브러리가 있을 자리. `scripts/build_coreml.sh`가 만든다.
pub fn default_library() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/coreml/libbadak_coreml.dylib")
}

#[cfg(target_os = "macos")]
pub use runtime::CoreMl;

#[cfg(target_os = "macos")]
mod runtime {
    use std::ffi::{CStr, CString, c_char, c_int, c_void};
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::{
        Lowered, Units, lower, supports, to_channels_first, to_channels_last, unsupported,
    };
    use crate::delegate::{Compiled, Delegate, Segment};
    use crate::error::Result;
    use crate::graph::{Graph, Node};

    // ANCHOR: dlopen
    // 실행할 때 동적 라이브러리를 올리는 운영체제의 함수들. 6장의 `mmap`처럼
    // 직접 선언한다. `RTLD_NOW`의 값은 macOS와 Linux에서 같다.
    const RTLD_NOW: c_int = 2;

    unsafe extern "C" {
        fn dlopen(path: *const c_char, flags: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
        fn dlerror() -> *const c_char;
    }

    type Load = unsafe extern "C" fn(*const c_char, i32, *mut c_char, usize) -> *mut c_void;
    type Run = unsafe extern "C" fn(
        *mut c_void,
        i32,
        *const *const c_char,
        *const *const f32,
        *const i64,
        i32,
        *const *const c_char,
        *const *mut f32,
        *const usize,
        *mut c_char,
        usize,
    ) -> i32;
    type Free = unsafe extern "C" fn(*mut c_void);
    type Plan = unsafe extern "C" fn(*const c_char, i32, *mut c_char, usize) -> i64;

    /// 올려 둔 라이브러리의 함수들.
    struct Shim {
        load: Load,
        run: Run,
        free: Free,
        plan: Plan,
    }
    // ANCHOR_END: dlopen

    fn last_dl_error() -> String {
        // SAFETY: `dlerror`는 빈 포인터이거나 0으로 끝나는 글을 돌려준다.
        unsafe {
            let message = dlerror();
            if message.is_null() {
                "알 수 없는 오류".to_owned()
            } else {
                CStr::from_ptr(message).to_string_lossy().into_owned()
            }
        }
    }

    impl Shim {
        fn open(path: &Path) -> Result<Self> {
            let text = CString::new(path.to_string_lossy().as_bytes())
                .map_err(|_| unsupported("라이브러리의 길에 0이 들어 있습니다"))?;
            // SAFETY: 0으로 끝나는 길을 준다. 실패하면 빈 포인터가 온다.
            let handle = unsafe { dlopen(text.as_ptr(), RTLD_NOW) };
            if handle.is_null() {
                return Err(unsupported(format!(
                    "Core ML 라이브러리를 올리지 못했습니다 ({}). scripts/build_coreml.sh 를 먼저 돌리십시오",
                    last_dl_error()
                )));
            }
            let symbol = |name: &CStr| -> Result<*mut c_void> {
                // SAFETY: 올라간 라이브러리의 손잡이와 0으로 끝나는 이름이다.
                let pointer = unsafe { dlsym(handle, name.as_ptr()) };
                if pointer.is_null() {
                    return Err(unsupported(format!("라이브러리에 {name:?}가 없습니다")));
                }
                Ok(pointer)
            };
            // SAFETY: 네 함수의 타입은 `apps/coreml/BadakCoreML.swift`의
            // `@_cdecl` 선언과 맞춘 것이다. 라이브러리는 내리지 않는다.
            unsafe {
                Ok(Self {
                    load: std::mem::transmute::<*mut c_void, Load>(symbol(c"badak_coreml_load")?),
                    run: std::mem::transmute::<*mut c_void, Run>(symbol(c"badak_coreml_run")?),
                    free: std::mem::transmute::<*mut c_void, Free>(symbol(c"badak_coreml_free")?),
                    plan: std::mem::transmute::<*mut c_void, Plan>(symbol(c"badak_coreml_plan")?),
                })
            }
        }
    }

    fn c_text(buffer: &[u8]) -> String {
        let end = buffer
            .iter()
            .position(|&byte| byte == 0)
            .unwrap_or(buffer.len());
        String::from_utf8_lossy(&buffer[..end]).into_owned()
    }

    /// 조각을 Core ML에 넘기는 델리게이트.
    pub struct CoreMl {
        shim: Arc<Shim>,
        units: Units,
        /// 모델 파일을 적어 둘 폴더.
        scratch: PathBuf,
        /// 옮길 수 있어도 넘기지 않을 연산의 이름들. 실험에 쓴다.
        refused: Vec<String>,
        written: AtomicUsize,
    }

    impl std::fmt::Debug for CoreMl {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("CoreMl")
                .field("units", &self.units)
                .field("refused", &self.refused)
                .finish()
        }
    }

    impl CoreMl {
        /// `library`를 올리고, 모델 파일은 `scratch`에 적는 델리게이트.
        pub fn new(
            library: impl AsRef<Path>,
            units: Units,
            scratch: impl Into<PathBuf>,
        ) -> Result<Self> {
            let scratch = scratch.into();
            std::fs::create_dir_all(&scratch).map_err(badak_formats::Error::from)?;
            Ok(Self {
                shim: Arc::new(Shim::open(library.as_ref())?),
                units,
                scratch,
                refused: Vec::new(),
                written: AtomicUsize::new(0),
            })
        }

        /// 이름이 `names`에 든 연산은 넘기지 않는다. "그 연산을 못 돌리는
        /// 가속기"를 흉내 낸다.
        pub fn refusing(mut self, names: &[&str]) -> Self {
            self.refused = names.iter().map(|name| (*name).to_owned()).collect();
            self
        }

        /// 같은 라이브러리를 쓰되 계산 장치의 범위가 다른 델리게이트.
        pub fn with_units(&self, units: Units) -> Self {
            Self {
                shim: Arc::clone(&self.shim),
                units,
                scratch: self.scratch.clone(),
                refused: self.refused.clone(),
                written: AtomicUsize::new(0),
            }
        }

        fn write(&self, lowered: &Lowered) -> Result<PathBuf> {
            let count = self.written.fetch_add(1, Ordering::Relaxed);
            let path = self.scratch.join(format!(
                "segment_{}_{}_{count}.mlmodel",
                std::process::id(),
                self.units as i32
            ));
            std::fs::write(&path, &lowered.model).map_err(badak_formats::Error::from)?;
            Ok(path)
        }

        /// 조각의 층마다 Core ML이 어느 장치에서 돌리려 하는지 묻는다.
        /// `(층의 이름, 장치의 이름)`들을 돌려준다.
        pub fn plan(&self, graph: &Graph, segment: &Segment) -> Result<Vec<(String, String)>> {
            let path = self.write(&lower(graph, segment)?)?;
            let text = CString::new(path.to_string_lossy().as_bytes()).expect("길에 0이 없다");
            let mut buffer = vec![0_u8; 1 << 16];
            // SAFETY: 0으로 끝나는 길과 쓸 수 있는 자리를 준다.
            let status = unsafe {
                (self.shim.plan)(
                    text.as_ptr(),
                    self.units as i32,
                    buffer.as_mut_ptr().cast(),
                    buffer.len(),
                )
            };
            let _ = std::fs::remove_file(&path);
            let text = c_text(&buffer);
            if status < 0 {
                return Err(unsupported(format!(
                    "Core ML의 계획을 읽지 못했습니다: {text}"
                )));
            }
            Ok(text
                .lines()
                .filter_map(|line| line.split_once('\t'))
                .map(|(name, device)| (name.to_owned(), device.to_owned()))
                .collect())
        }
    }

    // ANCHOR: coreml_compiled
    /// Core ML이 돌릴 준비를 마친 조각.
    struct Loaded {
        shim: Arc<Shim>,
        model: *mut c_void,
        lowered: Lowered,
        names: Vec<CString>,
        /// 축을 바꿔 적어 둘 자리. 입력마다 하나, 출력마다 하나.
        staged_inputs: Vec<Vec<f32>>,
        staged_outputs: Vec<Vec<f32>>,
    }

    impl Compiled for Loaded {
        fn run(&mut self, inputs: &[&[f32]], outputs: &mut [Vec<f32>]) -> Result<()> {
            // 1. 우리 레이아웃(채널이 뒤)를 Core ML의 레이아웃(채널이 앞)로.
            let mut input_pointers = Vec::with_capacity(inputs.len());
            for ((port, input), staged) in self
                .lowered
                .inputs
                .iter()
                .zip(inputs)
                .zip(&mut self.staged_inputs)
            {
                if port.channels_last {
                    to_channels_first(port, input, staged);
                    input_pointers.push(staged.as_ptr());
                } else {
                    input_pointers.push(input.as_ptr());
                }
            }
            let shapes: Vec<i64> = self
                .lowered
                .inputs
                .iter()
                .flat_map(|port| port.shape.map(|extent| extent as i64))
                .collect();
            let output_pointers: Vec<*mut f32> = self
                .lowered
                .outputs
                .iter()
                .zip(outputs.iter_mut())
                .zip(&mut self.staged_outputs)
                .map(|((port, output), staged)| {
                    if port.channels_last {
                        staged.as_mut_ptr()
                    } else {
                        output.as_mut_ptr()
                    }
                })
                .collect();
            let lens: Vec<usize> = self
                .lowered
                .outputs
                .iter()
                .map(|port| port.shape.iter().product())
                .collect();
            let names: Vec<*const c_char> = self.names.iter().map(|name| name.as_ptr()).collect();
            let (input_names, output_names) = names.split_at(inputs.len());
            let mut error = [0_u8; 512];
            // 2. 실행. 포인터들은 이 호출 동안 살아 있고 길이는 형상과 맞다.
            // SAFETY: 위와 같다. 라이브러리는 입력을 읽고 출력의 자리에
            // `lens`만큼만 쓴다.
            let status = unsafe {
                (self.shim.run)(
                    self.model,
                    inputs.len() as i32,
                    input_names.as_ptr(),
                    input_pointers.as_ptr(),
                    shapes.as_ptr(),
                    outputs.len() as i32,
                    output_names.as_ptr(),
                    output_pointers.as_ptr(),
                    lens.as_ptr(),
                    error.as_mut_ptr().cast(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(unsupported(format!(
                    "Core ML이 실행하지 못했습니다: {}",
                    c_text(&error)
                )));
            }
            // 3. 결과를 우리 레이아웃으로.
            for ((port, output), staged) in self
                .lowered
                .outputs
                .iter()
                .zip(outputs)
                .zip(&self.staged_outputs)
            {
                if port.channels_last {
                    to_channels_last(port, staged, output);
                }
            }
            Ok(())
        }
    }
    // ANCHOR_END: coreml_compiled

    impl Drop for Loaded {
        fn drop(&mut self) {
            // SAFETY: `load`가 준 손잡이를 한 번만 돌려준다.
            unsafe { (self.shim.free)(self.model) };
        }
    }

    impl Delegate for CoreMl {
        fn name(&self) -> String {
            format!("Core ML ({})", self.units.label())
        }

        fn supports(&self, graph: &Graph, node: &Node) -> bool {
            !self.refused.iter().any(|name| name == node.op.name()) && supports(graph, node)
        }

        fn compile(&self, graph: &Graph, segment: &Segment) -> Result<Box<dyn Compiled>> {
            let lowered = lower(graph, segment)?;
            let path = self.write(&lowered)?;
            let text = CString::new(path.to_string_lossy().as_bytes()).expect("길에 0이 없다");
            let mut error = [0_u8; 512];
            // SAFETY: 0으로 끝나는 길과 쓸 수 있는 자리를 준다.
            let model = unsafe {
                (self.shim.load)(
                    text.as_ptr(),
                    self.units as i32,
                    error.as_mut_ptr().cast(),
                    error.len(),
                )
            };
            let _ = std::fs::remove_file(&path);
            if model.is_null() {
                return Err(unsupported(format!(
                    "Core ML이 모델을 받지 않았습니다: {}",
                    c_text(&error)
                )));
            }
            let staged = |ports: &[super::Port]| -> Vec<Vec<f32>> {
                ports
                    .iter()
                    .map(|port| {
                        if port.channels_last {
                            vec![0.0; port.shape.iter().product()]
                        } else {
                            Vec::new()
                        }
                    })
                    .collect()
            };
            let names = lowered
                .inputs
                .iter()
                .chain(&lowered.outputs)
                .map(|port| CString::new(port.name.as_str()).expect("이름에 0이 없다"))
                .collect();
            Ok(Box::new(Loaded {
                shim: Arc::clone(&self.shim),
                model,
                staged_inputs: staged(&lowered.inputs),
                staged_outputs: staged(&lowered.outputs),
                names,
                lowered,
            }))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::delegate::{Options, partition};
    use badak_core::tensor::Tensor;
    use badak_formats::protobuf::{Field, Message};

    const FORMAT: &str = "coreml";

    fn fields(bytes: &[u8]) -> Vec<Field<'_>> {
        Message::new(bytes, FORMAT)
            .collect::<badak_formats::Result<_>>()
            .unwrap()
    }

    /// 합성곱(ReLU6) → 평균 → 선형 층.
    fn tiny() -> Graph {
        let mut graph = Graph::new("tiny");
        let x = graph.value("x", &[1, 4, 4, 2]);
        graph.inputs.push(x);
        // 필터 (출력 3, 높이 1, 너비 1, 입력 2).
        let filter = graph.constant(
            "filter",
            Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 1, 1, 2]).unwrap(),
        );
        let bias = graph.constant(
            "bias",
            Tensor::from_vec(vec![0.5_f32, -0.5, 0.0], &[3]).unwrap(),
        );
        let conv = graph.value("conv", &[1, 4, 4, 3]);
        graph.node(
            Op::Conv2d {
                padding: Padding::Same,
                stride: [1, 1],
                dilation: [1, 1],
                activation: Activation::Relu6,
            },
            &[x, filter, bias],
            &[conv],
        );
        let pooled = graph.value("pooled", &[1, 3]);
        graph.node(
            Op::Mean {
                axes: vec![1, 2],
                keep_dims: false,
            },
            &[conv],
            &[pooled],
        );
        let weight = graph.constant(
            "weight",
            Tensor::from_vec(vec![1.0_f32; 6], &[2, 3]).unwrap(),
        );
        let y = graph.value("y", &[1, 2]);
        graph.node(
            Op::FullyConnected {
                activation: Activation::None,
                keep_num_dims: false,
            },
            &[pooled, weight],
            &[y],
        );
        graph.outputs.push(y);
        graph
    }

    #[test]
    fn lowers_a_small_graph_to_a_model_that_parses() {
        let graph = tiny();
        let supported: Vec<bool> = graph
            .nodes
            .iter()
            .map(|node| supports(&graph, node))
            .collect();
        assert_eq!(supported, [true, true, true]);
        let segments = partition(&graph, &supported, Options::default()).unwrap();
        let lowered = lower(&graph, &segments[0]).unwrap();
        // 합성곱, 자르기, 풀링, 선형 층.
        assert_eq!(lowered.layers, 4);
        assert_eq!(lowered.inputs[0].shape, [1, 2, 4, 4]);
        assert!(lowered.inputs[0].channels_last);
        assert_eq!(lowered.outputs[0].shape, [1, 2, 1, 1]);

        let model = fields(&lowered.model);
        assert_eq!(
            model.iter().map(|field| field.number).collect::<Vec<_>>(),
            [1, 2, 500]
        );
        assert_eq!(model[0].int(FORMAT).unwrap(), 4);
        let network = fields(model[2].bytes(FORMAT).unwrap());
        let layers: Vec<&Field<'_>> = network.iter().filter(|field| field.number == 1).collect();
        assert_eq!(layers.len(), 4);
        // 첫 층: 이름, 입력, 출력, 합성곱(100번).
        let first = fields(layers[0].bytes(FORMAT).unwrap());
        assert_eq!(
            first.iter().map(|field| field.number).collect::<Vec<_>>(),
            [1, 2, 3, 100]
        );
        assert_eq!(first[1].string(FORMAT).unwrap(), "v0");
        let conv = fields(first[3].bytes(FORMAT).unwrap());
        let weight = conv.iter().find(|field| field.number == 90).unwrap();
        let mut values = Vec::new();
        fields(weight.bytes(FORMAT).unwrap())[0]
            .floats(&mut values, FORMAT)
            .unwrap();
        // 창이 1 × 1이면 (출력, 입력)의 순서가 그대로다.
        assert_eq!(values, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        // 둘째 층은 자르기(660번)이고 첫 층의 임시 이름을 읽는다.
        let second = fields(layers[1].bytes(FORMAT).unwrap());
        assert_eq!(second.last().unwrap().number, 660);
        assert_eq!(
            second[1].string(FORMAT).unwrap(),
            first[2].string(FORMAT).unwrap()
        );
    }

    #[test]
    fn reorders_filters_to_channels_first() {
        // 필터 (출력 1, 높이 1, 너비 2, 입력 2): [[a0 a1] [b0 b1]] → 입력마다
        // 모아서 [a0 b0 a1 b1].
        let mut graph = Graph::new("reorder");
        let x = graph.value("x", &[1, 1, 2, 2]);
        graph.inputs.push(x);
        let filter = graph.constant(
            "f",
            Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 2, 2]).unwrap(),
        );
        let y = graph.value("y", &[1, 1, 1, 1]);
        graph.node(
            Op::Conv2d {
                padding: Padding::Valid,
                stride: [1, 1],
                dilation: [1, 1],
                activation: Activation::None,
            },
            &[x, filter],
            &[y],
        );
        graph.outputs.push(y);
        let segments = partition(&graph, &[true], Options::default()).unwrap();
        let lowered = lower(&graph, &segments[0]).unwrap();
        let model = fields(&lowered.model);
        let network = fields(model[2].bytes(FORMAT).unwrap());
        let layer = fields(network[0].bytes(FORMAT).unwrap());
        let conv = fields(layer[3].bytes(FORMAT).unwrap());
        let weight = conv.iter().find(|field| field.number == 90).unwrap();
        let mut values = Vec::new();
        fields(weight.bytes(FORMAT).unwrap())[0]
            .floats(&mut values, FORMAT)
            .unwrap();
        assert_eq!(values, [1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn layout_conversion_round_trips() {
        let port = Port {
            value: 0,
            name: "v0".into(),
            shape: [1, 3, 2, 2],
            channels_last: true,
        };
        let last: Vec<f32> = (0..12).map(|index| index as f32).collect();
        let mut first = vec![0.0; 12];
        to_channels_first(&port, &last, &mut first);
        assert_eq!(
            first,
            [0.0, 3.0, 6.0, 9.0, 1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0]
        );
        let mut back = vec![0.0; 12];
        to_channels_last(&port, &first, &mut back);
        assert_eq!(back, last);
    }

    #[test]
    fn says_no_to_what_it_cannot_lower() {
        let mut graph = tiny();
        // 평균을 높이만 따라 내면 옮기지 못한다.
        graph.nodes[1].op = Op::Mean {
            axes: vec![1],
            keep_dims: false,
        };
        assert!(!supports(&graph, &graph.nodes[1]));
        graph.nodes[0].op = Op::Sigmoid;
        assert!(!supports(&graph, &graph.nodes[0]));
        let segment = Segment {
            delegated: true,
            nodes: vec![0],
            inputs: vec![0],
            outputs: vec![3],
        };
        assert!(lower(&graph, &segment).is_err());
    }
}
