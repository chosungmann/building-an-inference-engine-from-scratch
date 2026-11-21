//! LoRA 어댑터 (41장).
//!
//! 모델을 조금 고쳐 쓰고 싶을 때 가중치 전체를 다시 학습시키는 대신,
//! 선형 층마다 **작은 행렬 둘**을 덧붙여 그것만 학습시키는 방법이다.
//! 원래의 가중치 `W`(`out × in`)는 그대로 두고, `B`(`out × r`)와
//! `A`(`r × in`)를 곱한 것을 더한다.
//!
//! ```text
//! y = W x + (alpha / r) · B (A x)
//! ```
//!
//! `r`은 8이나 16 같은 작은 수다. `in`과 `out`이 수백에서 수천이므로
//! 어댑터는 원래 가중치의 백분의 일쯤이다.
//!
//! 추론 엔진이 할 일은 둘 가운데 하나다. 어댑터를 가중치에 **합쳐
//! 버리거나**(읽을 때 한 번 계산하고, 그 뒤로는 보통의 모델이다),
//! **따로 둔 채로** 스텝마다 위의 식을 계산한다(조금 느리지만, 원래
//! 가중치를 건드리지 않으므로 어댑터를 갈아 끼울 수 있다).

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use badak_core::Tensor;
use badak_core::ops::LinearWeight;
use badak_core::packed::{Delegated, Matrix};
use badak_formats::json::{self, Value};
use badak_formats::safetensors::SafeTensors;
use badak_formats::{Error, Result};

use super::model::LlamaModel;

fn invalid(message: impl Into<String>) -> Error {
    Error::Parse {
        format: "LoRA 어댑터",
        offset: 0,
        message: message.into(),
    }
}

/// 층 하나에 덧붙는 행렬 둘.
#[derive(Debug, Clone)]
struct Pair {
    /// `A`. `(r, in)`.
    down: Vec<f32>,
    /// `B`. `(out, r)`.
    up: Vec<f32>,
    rank: usize,
    rows: usize,
    cols: usize,
}

// ANCHOR: adapter
/// 읽어 들인 어댑터.
#[derive(Debug, Clone)]
pub struct Adapter {
    /// (블록의 번호, 층의 이름)마다의 행렬 둘.
    pairs: BTreeMap<(usize, String), Pair>,
    /// 더할 때 곱하는 수. `alpha / r`.
    pub scale: f32,
    /// 작은 쪽의 크기 `r`.
    pub rank: usize,
}

impl Adapter {
    /// PEFT가 쓴 디렉터리(`adapter_config.json`, `adapter_model.safetensors`)를
    /// 읽는다.
    pub fn load(directory: impl AsRef<Path>) -> Result<Self> {
        let directory = directory.as_ref();
        let config = std::fs::read_to_string(directory.join("adapter_config.json"))?;
        let config = json::parse(&config)?;
        let number = |name: &str| {
            config
                .get(name)
                .and_then(Value::as_f64)
                .ok_or_else(|| invalid(format!("adapter_config.json에 {name}이 없습니다")))
        };
        let (rank, alpha) = (number("r")? as usize, number("lora_alpha")? as f32);
        if rank == 0 {
            return Err(invalid("r이 0입니다"));
        }

        // 이름의 꼴: `…layers.3.self_attn.q_proj.lora_A.weight`.
        let file = SafeTensors::open(directory.join("adapter_model.safetensors"))?;
        let mut halves: BTreeMap<(usize, String), [Option<Tensor>; 2]> = BTreeMap::new();
        for name in file.names() {
            let parts: Vec<&str> = name.split('.').collect();
            let which = match parts.iter().position(|part| part.starts_with("lora_")) {
                Some(index) if parts[index] == "lora_A" => (index, 0),
                Some(index) if parts[index] == "lora_B" => (index, 1),
                _ => return Err(invalid(format!("모르는 이름입니다: {name}"))),
            };
            let layer = parts
                .iter()
                .position(|part| *part == "layers")
                .and_then(|index| parts.get(index + 1)?.parse::<usize>().ok())
                .ok_or_else(|| invalid(format!("블록의 번호가 없습니다: {name}")))?;
            let module = parts[which.0 - 1].to_owned();
            halves.entry((layer, module)).or_default()[which.1] = Some(file.tensor(name)?);
        }
        let mut pairs = BTreeMap::new();
        for (key, [down, up]) in halves {
            let (Some(down), Some(up)) = (down, up) else {
                return Err(invalid(format!("{key:?}의 행렬 둘 가운데 하나가 없습니다")));
            };
            let (&[r, cols], &[rows, r2]) = (down.shape(), up.shape()) else {
                return Err(invalid(format!("{key:?}의 행렬이 2차원이 아닙니다")));
            };
            if r != rank || r2 != rank {
                return Err(invalid(format!(
                    "{key:?}의 크기가 r={rank}과 맞지 않습니다"
                )));
            }
            let pair = Pair {
                down: down.contiguous().to_vec::<f32>()?,
                up: up.contiguous().to_vec::<f32>()?,
                rank,
                rows,
                cols,
            };
            pairs.insert(key, pair);
        }
        // PEFT에는 r의 제곱근으로 나누는 변형도 있다(`use_rslora`).
        let root = matches!(config.get("use_rslora"), Some(Value::Bool(true)));
        let divisor = if root {
            (rank as f32).sqrt()
        } else {
            rank as f32
        };
        Ok(Self {
            pairs,
            scale: alpha / divisor,
            rank,
        })
    }

    /// 어댑터가 덧붙는 층의 수.
    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    /// 덧붙는 층이 없는가.
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    /// 어댑터의 값의 수.
    pub fn parameter_count(&self) -> usize {
        self.pairs
            .values()
            .map(|pair| pair.down.len() + pair.up.len())
            .sum()
    }
    // ANCHOR_END: adapter

    /// 모델의 층마다 `change`로 가중치를 바꿔 끼운다. 바꾼 층의 수를 돌려준다.
    fn apply(
        &self,
        model: &mut LlamaModel,
        change: impl Fn(&Matrix, &Pair, f32) -> Result<Matrix>,
    ) -> Result<usize> {
        let mut changed = 0;
        model.replace_matrices(|name, layer, matrix| {
            let pair = layer.and_then(|layer| self.pairs.get(&(layer, name.to_owned())));
            let Some(pair) = pair else {
                return Ok(matrix.clone());
            };
            if matrix.shape() != [pair.rows, pair.cols] {
                return Err(invalid(format!(
                    "{name}의 형상 {:?}이 어댑터의 ({}, {})과 다릅니다",
                    matrix.shape(),
                    pair.rows,
                    pair.cols
                )));
            }
            changed += 1;
            change(matrix, pair, self.scale)
        })?;
        if changed != self.pairs.len() {
            return Err(invalid("어댑터의 층 가운데 모델에 없는 것이 있습니다"));
        }
        Ok(changed)
    }

    // ANCHOR: merge
    /// 어댑터를 가중치에 **합친다**. `W' = W + scale · B A`.
    ///
    /// 합친 뒤로는 보통의 모델이다. 스텝마다 드는 값이 없다. 대신 합친
    /// 가중치를 `f32`로 새로 만들어 쥐어야 하고(파일을 비춘 것을 그대로
    /// 쓸 수 없다), 원래의 가중치로 돌아가려면 다시 읽어야 한다.
    pub fn merge(&self, model: &mut LlamaModel) -> Result<usize> {
        self.apply(model, |matrix, pair, scale| {
            let mut merged = matrix.to_f32()?.contiguous().to_vec::<f32>()?;
            for (row, values) in merged.chunks_exact_mut(pair.cols).enumerate() {
                for (index, &up) in pair.up[row * pair.rank..][..pair.rank].iter().enumerate() {
                    let down = &pair.down[index * pair.cols..][..pair.cols];
                    for (value, &down) in values.iter_mut().zip(down) {
                        *value += scale * up * down;
                    }
                }
            }
            Ok(Matrix::Dense(Tensor::from_vec(
                merged,
                &[pair.rows, pair.cols],
            )?))
        })
    }
    // ANCHOR_END: merge

    /// 어댑터를 **따로 둔 채로** 붙인다. 원래의 가중치는 그대로다.
    pub fn attach(&self, model: &mut LlamaModel) -> Result<usize> {
        self.apply(model, |matrix, pair, scale| {
            let beside = Beside {
                base: matrix.clone(),
                pair: pair.clone(),
                scale,
            };
            Ok(Matrix::Delegated(Delegated::new(
                matrix.clone(),
                Arc::new(beside),
            )))
        })
    }
}

// ANCHOR: beside
/// 원래의 가중치 옆에 어댑터를 둔 선형 층.
struct Beside {
    base: Matrix,
    pair: Pair,
    scale: f32,
}

impl LinearWeight for Beside {
    fn dimensions(&self) -> Option<(usize, usize)> {
        Some((self.pair.rows, self.pair.cols))
    }

    fn apply(&self, x: &[f32], count: usize, y: &mut [f32]) -> badak_core::Result<()> {
        // 1. 원래의 층. 3부의 커널이 그대로 돈다.
        self.base.apply(x, count, y)?;
        // 2. 어댑터. 벡터마다 `A x`(r개의 수)를 구하고 `B`를 곱해 더한다.
        let Pair {
            down,
            up,
            rank,
            rows,
            cols,
        } = &self.pair;
        let mut small = vec![0.0_f32; *rank];
        for (x, y) in x.chunks_exact(*cols).zip(y.chunks_exact_mut(*rows)) {
            for (value, down) in small.iter_mut().zip(down.chunks_exact(*cols)) {
                *value = self.scale * down.iter().zip(x).map(|(a, b)| a * b).sum::<f32>();
            }
            for (out, up) in y.iter_mut().zip(up.chunks_exact(*rank)) {
                *out += up.iter().zip(&small).map(|(a, b)| a * b).sum::<f32>();
            }
        }
        Ok(())
    }
}
// ANCHOR_END: beside
