//! Llama 계열 언어 모델 (9장부터).
//!
//! Llama, SmolLM2를 비롯해 같은 구조를 쓰는 모델을 다룬다. 이 장에서는
//! 설정과 가중치를 읽고 블록을 이루는 부품을 만든다. 블록을 쌓아 순전파를
//! 조립하는 일은 10장에서 한다.

pub mod attention;
pub mod cache;
pub mod generate;
pub mod lora;
pub mod model;
pub mod paged;
pub mod perplexity;
pub mod rope;
pub mod speculative;

use std::path::{Path, PathBuf};

use badak_core::packed::{Format, Matrix, Packed};
use badak_core::{Tensor, ops};
use badak_formats::json::{self, Value};
use badak_formats::safetensors::SafeTensors;
use badak_formats::{Error, Result};

fn invalid(message: impl Into<String>) -> Error {
    Error::Parse {
        format: "Llama 모델",
        offset: 0,
        message: message.into(),
    }
}

// ANCHOR: config
/// 모델의 크기와 구조를 정하는 숫자들. `config.json`에서 읽는다.
#[derive(Debug, Clone, PartialEq)]
pub struct LlamaConfig {
    /// 단어 사전의 크기.
    pub vocab_size: usize,
    /// 토큰 하나를 나타내는 벡터의 길이.
    pub hidden_size: usize,
    /// 블록의 수.
    pub num_layers: usize,
    /// 어텐션 헤드의 수.
    pub num_heads: usize,
    /// 키와 값을 만드는 헤드의 수. `num_heads`의 약수다.
    pub num_kv_heads: usize,
    /// MLP 가운데 층의 길이.
    pub intermediate_size: usize,
    /// RMS 정규화에서 0으로 나누는 것을 막는 수.
    pub rms_norm_eps: f32,
    /// 위치를 회전으로 나타낼 때의 기준 주기.
    pub rope_theta: f32,
    /// 긴 글을 위한 주기 조정. 없는 모델도 있다.
    pub rope_scaling: Option<RopeScaling>,
    /// 글의 시작을 알리는 토큰의 번호.
    pub bos_token_id: Option<u32>,
    /// 글의 끝을 알리는 토큰의 번호들. 여럿일 수 있다.
    pub eos_token_ids: Vec<u32>,
}

/// Llama 3의 주기 조정에 쓰는 값들.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RopeScaling {
    /// 긴 주기를 늘리는 배수.
    pub factor: f32,
    /// 이 값으로 나눈 기준 길이보다 긴 주기는 `factor`배로 늘린다.
    pub low_freq_factor: f32,
    /// 이 값으로 나눈 기준 길이보다 짧은 주기는 그대로 둔다.
    pub high_freq_factor: f32,
    /// 학습할 때의 기준 글 길이.
    pub original_max_position: f32,
}
// ANCHOR_END: config

impl LlamaConfig {
    /// `config.json`을 읽는다.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let root = json::parse(&std::fs::read_to_string(path)?)?;
        let number = |key: &str| {
            root.get(key)
                .and_then(Value::as_f64)
                .ok_or_else(|| invalid(format!("config.json에 {key}가 없습니다")))
        };
        let size = |key: &str| {
            root.get(key)
                .and_then(Value::as_usize)
                .ok_or_else(|| invalid(format!("config.json에 {key}가 없습니다")))
        };
        let rope_scaling = match root.get("rope_scaling") {
            None | Some(Value::Null) => None,
            Some(scaling) => {
                let field = |key: &str| {
                    scaling
                        .get(key)
                        .and_then(Value::as_f64)
                        .ok_or_else(|| invalid(format!("rope_scaling에 {key}가 없습니다")))
                };
                let kind = scaling.get("rope_type").and_then(Value::as_str);
                if kind != Some("llama3") {
                    return Err(Error::Unsupported(format!("rope_scaling의 종류 {kind:?}")));
                }
                Some(RopeScaling {
                    factor: field("factor")? as f32,
                    low_freq_factor: field("low_freq_factor")? as f32,
                    high_freq_factor: field("high_freq_factor")? as f32,
                    original_max_position: field("original_max_position_embeddings")? as f32,
                })
            }
        };
        let config = Self {
            vocab_size: size("vocab_size")?,
            hidden_size: size("hidden_size")?,
            num_layers: size("num_hidden_layers")?,
            num_heads: size("num_attention_heads")?,
            num_kv_heads: size("num_key_value_heads")?,
            intermediate_size: size("intermediate_size")?,
            rms_norm_eps: number("rms_norm_eps")? as f32,
            rope_theta: number("rope_theta")? as f32,
            rope_scaling,
            bos_token_id: root
                .get("bos_token_id")
                .and_then(Value::as_usize)
                .map(|id| id as u32),
            // 번호 하나일 수도, 번호의 배열일 수도 있다.
            eos_token_ids: match root.get("eos_token_id") {
                Some(Value::Array(ids)) => ids
                    .iter()
                    .filter_map(Value::as_usize)
                    .map(|id| id as u32)
                    .collect(),
                Some(id) => id.as_usize().map(|id| id as u32).into_iter().collect(),
                None => Vec::new(),
            },
        };
        // 뒤의 코드가 나눗셈에 쓰는 관계들을 여기서 확인해 둔다.
        if config.num_heads == 0
            || config.num_kv_heads == 0
            || !config.hidden_size.is_multiple_of(config.num_heads)
            || !config.num_heads.is_multiple_of(config.num_kv_heads)
            || !config.head_dim().is_multiple_of(2)
        {
            return Err(invalid("헤드의 수와 벡터의 길이가 서로 맞지 않습니다"));
        }
        Ok(config)
    }

    /// 헤드 하나가 다루는 벡터의 길이.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    /// 키·값 헤드 하나를 함께 쓰는 질의 헤드의 수.
    pub fn heads_per_kv(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }
}

// ANCHOR: block_weights
/// 블록 하나의 가중치.
pub struct BlockWeights {
    /// 어텐션 앞의 정규화.
    pub attention_norm: Tensor,
    /// 질의, 키, 값, 출력을 만드는 선형 층. 편향은 없다. 20장부터 행렬
    /// 가중치는 저장된 방식이 여럿일 수 있어 [`Matrix`]에 담는다.
    pub q_proj: Matrix,
    /// 키.
    pub k_proj: Matrix,
    /// 값.
    pub v_proj: Matrix,
    /// 출력.
    pub o_proj: Matrix,
    /// MLP 앞의 정규화.
    pub mlp_norm: Tensor,
    /// MLP의 세 선형 층.
    pub gate_proj: Matrix,
    /// 위로 펼치는 층.
    pub up_proj: Matrix,
    /// 아래로 모으는 층.
    pub down_proj: Matrix,
    /// MLP 전체를 대신 계산하는 것 (23장). 있으면 위의 세 층을 쓰지 않는다.
    pub mlp_override: Option<std::sync::Arc<dyn ops::Operator>>,
}
// ANCHOR_END: block_weights

// ANCHOR: weight_format
/// 행렬 가중치를 메모리에 두는 방식 (20장).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    /// `f32`로 펼친다. 19장까지의 방식이다.
    F32,
    /// 파일에 저장된 그대로 둔다. BF16 파일이면 파일을 매핑한 메모리를 가리키는
    /// BF16 텐서가 된다.
    Stored,
    /// 읽으면서 값 32개마다 8비트 정수로 양자화한다.
    Q8_0,
    /// 읽으면서 값 32개마다 4비트 정수로 양자화한다.
    Q4_0,
}

impl WeightFormat {
    /// 환경 변수 `BADAK_WEIGHTS`(`f32`, `stored`, `q8_0`, `q4_0`)로 정한
    /// 방식. 없으면 저장된 그대로다.
    pub fn from_environment() -> Self {
        match std::env::var("BADAK_WEIGHTS").as_deref() {
            Ok("f32") => Self::F32,
            Ok("q8_0") => Self::Q8_0,
            Ok("q4_0") => Self::Q4_0,
            _ => Self::Stored,
        }
    }
}
// ANCHOR_END: weight_format

// ANCHOR: pack
impl WeightFormat {
    /// 이 방식으로 미리 양자화해 둔 가중치를 담는 파일의 이름. 양자화하지
    /// 않는 방식이면 `None`.
    pub fn packed_file(self) -> Option<&'static str> {
        match self {
            Self::Q8_0 => Some("model_q8_0.safetensors"),
            Self::Q4_0 => Some("model_q4_0.safetensors"),
            Self::F32 | Self::Stored => None,
        }
    }
}

/// 모델의 가중치를 양자화해 파일로 적어 둔다 (22장). 적은 파일의 경로를
/// 돌려준다.
///
/// 형식은 6장의 safetensors 그대로다. 행렬 가중치는 묶음의 바이트를
/// `(행의 수, 행 하나의 바이트 수)` 모양의 `U8` 텐서로 적고, 나머지
/// (정규화의 가중치)는 원래대로 적는다. 텐서의 이름은 원래 파일과 같다.
pub fn pack(directory: impl AsRef<Path>, format: WeightFormat) -> Result<PathBuf> {
    let directory = directory.as_ref();
    let file = format
        .packed_file()
        .ok_or_else(|| invalid("양자화하는 방식이 아닙니다"))?;
    let path = directory.join(file);
    pack_to(directory, format, &path)?;
    Ok(path)
}

/// [`pack`]과 같되, 적을 파일의 경로를 직접 정한다.
pub fn pack_to(directory: impl AsRef<Path>, format: WeightFormat, path: &Path) -> Result<()> {
    let directory = directory.as_ref();
    let (file, block_format) = match format {
        WeightFormat::Q8_0 => ("Q8_0", Format::Q8_0),
        WeightFormat::Q4_0 => ("Q4_0", Format::Q4_0),
        _ => return Err(invalid("양자화하는 방식이 아닙니다")),
    };
    let source = SafeTensors::open(directory.join("model.safetensors"))?;
    let mut tensors = Vec::new();
    for name in source.names() {
        let stored = source.tensor_as_stored(name)?;
        let tensor = match *stored.shape() {
            [rows, cols] => {
                let values = ops::to_f32(&stored)?;
                let packed = Packed::quantize(block_format, values.as_slice()?, rows, cols)?;
                packed.storage().clone()
            }
            _ => stored,
        };
        tensors.push((name, tensor));
    }
    let named: Vec<(&str, &Tensor)> = tensors
        .iter()
        .map(|(name, tensor)| (*name, tensor))
        .collect();
    badak_formats::safetensors::write(path, &named, &[("badak.format", file)])?;
    Ok(())
}
// ANCHOR_END: pack

// ANCHOR: weights
/// 디스크의 모델 디렉터리. 설정과, 메모리에 매핑한 가중치 파일을 쥐고 있다.
pub struct LlamaFiles {
    /// 모델의 설정.
    pub config: LlamaConfig,
    weights: SafeTensors,
    directory: PathBuf,
    /// 행렬 가중치를 꺼낼 때의 방식. 처음에는 `f32`다.
    format: WeightFormat,
    /// 미리 양자화해 둔 파일. 있으면 거기서 꺼낸다 (22장).
    packed: Option<SafeTensors>,
}

impl LlamaFiles {
    /// `config.json`과 `model.safetensors`가 있는 디렉터리를 연다.
    ///
    /// 가중치의 값은 아직 읽지 않는다. 텐서를 꺼낼 때 그 텐서만 읽는다.
    pub fn open(directory: impl AsRef<Path>) -> Result<Self> {
        let directory = directory.as_ref().to_path_buf();
        Ok(Self {
            config: LlamaConfig::from_file(directory.join("config.json"))?,
            weights: SafeTensors::open(directory.join("model.safetensors"))?,
            directory,
            format: WeightFormat::F32,
            packed: None,
        })
    }

    /// 행렬 가중치를 꺼내는 방식을 바꾼다 (20장).
    ///
    /// 22장: 그 방식으로 미리 양자화해 둔 파일([`pack`])이 디렉터리에 있으면
    /// 그 파일을 매핑해 쓴다.
    pub fn with_format(mut self, format: WeightFormat) -> Self {
        self.format = format;
        self.packed = format
            .packed_file()
            .map(|file| self.directory.join(file))
            .filter(|path| path.exists())
            .and_then(|path| SafeTensors::open(path).ok());
        self
    }

    /// 모델 디렉터리의 경로.
    pub fn directory(&self) -> &Path {
        &self.directory
    }

    /// 가중치 파일. 텐서의 이름과 형상을 살펴볼 때 쓴다.
    pub fn weights(&self) -> &SafeTensors {
        &self.weights
    }

    /// 이름으로 텐서를 읽고 형상을 확인한다.
    fn tensor(&self, name: &str, shape: &[usize]) -> Result<Tensor> {
        let tensor = self.weights.tensor(name)?;
        if tensor.shape() != shape {
            return Err(invalid(format!(
                "{name}의 형상이 {:?}인데 설정에 따르면 {shape:?}여야 합니다",
                tensor.shape()
            )));
        }
        Ok(tensor)
    }

    // ANCHOR: matrix
    /// 이름으로 행렬 가중치를 꺼낸다. 정해 둔 방식대로 메모리에 둔다 (20장).
    fn matrix(&self, name: &str, shape: &[usize]) -> Result<Matrix> {
        let stored = match self.format {
            WeightFormat::F32 => return Ok(self.tensor(name, shape)?.into()),
            _ => self.weights.tensor_as_stored(name)?,
        };
        if stored.shape() != shape {
            return Err(invalid(format!(
                "{name}의 형상이 {:?}인데 설정에 따르면 {shape:?}여야 합니다",
                stored.shape()
            )));
        }
        let format = match self.format {
            WeightFormat::Q8_0 => Format::Q8_0,
            WeightFormat::Q4_0 => Format::Q4_0,
            _ => return Ok(stored.into()),
        };
        // 22장: 미리 양자화해 둔 파일이 있으면 그 바이트를 그대로 빌려 쓴다.
        if let Some(file) = &self.packed {
            let packed = Packed::from_tensor(format, file.tensor(name)?)?;
            if packed.shape() != shape {
                return Err(invalid(format!(
                    "양자화해 둔 파일의 {name}의 형상이 {:?}인데 {shape:?}여야 합니다",
                    packed.shape()
                )));
            }
            return Ok(packed.into());
        }
        // 양자화는 `f32` 값에서 한다. 이 텐서 하나만 잠깐 펼친다.
        let values = ops::to_f32(&stored)?;
        let packed = Packed::quantize(format, values.as_slice()?, shape[0], shape[1])?;
        Ok(packed.into())
    }
    // ANCHOR_END: matrix

    /// 토큰 임베딩 표. `(단어 사전 크기, 벡터 길이)`. 늘 `f32`다.
    pub fn embedding(&self) -> Result<Tensor> {
        let c = &self.config;
        self.tensor("model.embed_tokens.weight", &[c.vocab_size, c.hidden_size])
    }

    /// 토큰 임베딩 표를 정해 둔 방식대로 꺼낸다 (20장).
    pub fn embedding_matrix(&self) -> Result<Matrix> {
        let c = &self.config;
        self.matrix("model.embed_tokens.weight", &[c.vocab_size, c.hidden_size])
    }

    /// 마지막 정규화의 가중치.
    pub fn final_norm(&self) -> Result<Tensor> {
        self.tensor("model.norm.weight", &[self.config.hidden_size])
    }

    /// `index`번째 블록의 가중치.
    pub fn block(&self, index: usize) -> Result<BlockWeights> {
        let c = &self.config;
        let (d, kv, m) = (
            c.hidden_size,
            c.num_kv_heads * c.head_dim(),
            c.intermediate_size,
        );
        let name = |part: &str| format!("model.layers.{index}.{part}.weight");
        // 정규화의 가중치는 벡터라서 늘 `f32`로 읽는다.
        let get = |part: &str, shape: &[usize]| self.tensor(&name(part), shape);
        let matrix = |part: &str, shape: &[usize]| self.matrix(&name(part), shape);
        Ok(BlockWeights {
            attention_norm: get("input_layernorm", &[d])?,
            q_proj: matrix("self_attn.q_proj", &[d, d])?,
            k_proj: matrix("self_attn.k_proj", &[kv, d])?,
            v_proj: matrix("self_attn.v_proj", &[kv, d])?,
            o_proj: matrix("self_attn.o_proj", &[d, d])?,
            mlp_norm: get("post_attention_layernorm", &[d])?,
            gate_proj: matrix("mlp.gate_proj", &[m, d])?,
            up_proj: matrix("mlp.up_proj", &[m, d])?,
            down_proj: matrix("mlp.down_proj", &[d, m])?,
            mlp_override: None,
        })
    }
}
// ANCHOR_END: weights

// ANCHOR: mlp
/// 블록의 MLP. 문(gate)이 달린 구조로, SwiGLU라고 부른다.
///
/// ```text
/// down( silu(gate(x)) × up(x) )
/// ```
pub fn mlp(x: &Tensor, weights: &BlockWeights) -> Result<Tensor> {
    // 14장: 프로파일러가 켜져 있으면 이 함수에 쓴 시간을 적는다.
    let _scope = badak_core::profile::scope("mlp");
    // 23장: 통째로 맡긴 곳이 있으면 거기서 계산한다.
    if let Some(operator) = &weights.mlp_override {
        return Ok(operator.apply(x)?);
    }
    let gate = ops::linear(x, &weights.gate_proj, None)?;
    let up = ops::linear(x, &weights.up_proj, None)?;
    // 24장: 활성화 함수와 곱셈을 한 번에 한다. 처음에는 `silu`의 결과를 텐서로
    // 만든 뒤 `mul`로 곱했다. 같은 연산을 같은 순서로 하므로 비트는 같다.
    let hidden = ops::zip(&gate, &up, |gate, up| badak_core::nn::silu(gate) * up)?;
    Ok(ops::linear(&hidden, &weights.down_proj, None)?)
}
// ANCHOR_END: mlp

// ANCHOR: block
/// 블록 하나. 어텐션과 MLP가 각각 정규화를 앞세우고, 결과를 입력에 더한다.
///
/// ```text
/// h    = x + 어텐션(정규화(x))
/// 출력 = h + MLP(정규화(h))
/// ```
pub fn block(
    x: &Tensor,
    weights: &BlockWeights,
    config: &LlamaConfig,
    frequencies: &[f32],
) -> Result<Tensor> {
    let normed = ops::rms_norm(x, &weights.attention_norm, config.rms_norm_eps)?;
    let attended = attention::attention(&normed, weights, config, frequencies)?;
    let h = ops::add(x, &attended.output)?;

    let normed = ops::rms_norm(&h, &weights.mlp_norm, config.rms_norm_eps)?;
    Ok(ops::add(&h, &mlp(&normed, weights)?)?)
}
// ANCHOR_END: block
