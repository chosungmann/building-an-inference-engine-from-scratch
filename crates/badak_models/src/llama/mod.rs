//! Llama 계열 언어 모델 (9장부터).
//!
//! Llama, SmolLM2를 비롯해 같은 구조를 쓰는 모델을 다룬다. 이 장에서는
//! 설정과 가중치를 읽고 블록을 이루는 부품을 만든다. 블록을 쌓아 순전파를
//! 조립하는 일은 10장에서 한다.

pub mod attention;
pub mod model;
pub mod rope;

use std::path::{Path, PathBuf};

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
    /// 질의, 키, 값, 출력을 만드는 선형 층. 편향은 없다.
    pub q_proj: Tensor,
    /// 키.
    pub k_proj: Tensor,
    /// 값.
    pub v_proj: Tensor,
    /// 출력.
    pub o_proj: Tensor,
    /// MLP 앞의 정규화.
    pub mlp_norm: Tensor,
    /// MLP의 세 선형 층.
    pub gate_proj: Tensor,
    /// 위로 펼치는 층.
    pub up_proj: Tensor,
    /// 아래로 모으는 층.
    pub down_proj: Tensor,
}
// ANCHOR_END: block_weights

// ANCHOR: weights
/// 디스크의 모델 디렉터리. 설정과, 메모리에 비춘 가중치 파일을 쥐고 있다.
pub struct LlamaFiles {
    /// 모델의 설정.
    pub config: LlamaConfig,
    weights: SafeTensors,
    directory: PathBuf,
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
        })
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

    /// 토큰 임베딩 표. `(단어 사전 크기, 벡터 길이)`.
    pub fn embedding(&self) -> Result<Tensor> {
        let c = &self.config;
        self.tensor("model.embed_tokens.weight", &[c.vocab_size, c.hidden_size])
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
        let get = |part: &str, shape: &[usize]| {
            self.tensor(&format!("model.layers.{index}.{part}.weight"), shape)
        };
        Ok(BlockWeights {
            attention_norm: get("input_layernorm", &[d])?,
            q_proj: get("self_attn.q_proj", &[d, d])?,
            k_proj: get("self_attn.k_proj", &[kv, d])?,
            v_proj: get("self_attn.v_proj", &[kv, d])?,
            o_proj: get("self_attn.o_proj", &[d, d])?,
            mlp_norm: get("post_attention_layernorm", &[d])?,
            gate_proj: get("mlp.gate_proj", &[m, d])?,
            up_proj: get("mlp.up_proj", &[m, d])?,
            down_proj: get("mlp.down_proj", &[d, m])?,
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
    let gate = ops::silu(&ops::linear(x, &weights.gate_proj, None)?)?;
    let up = ops::linear(x, &weights.up_proj, None)?;
    Ok(ops::linear(
        &ops::mul(&gate, &up)?,
        &weights.down_proj,
        None,
    )?)
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
