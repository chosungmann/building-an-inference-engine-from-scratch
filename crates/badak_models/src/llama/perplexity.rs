//! 퍼플렉서티: 모델이 진짜 글의 다음 토큰을 얼마나 잘 맞히는지 잰다 (21장).

use badak_core::threads;
use badak_formats::Result;

use super::cache::{KvCache, KvPrecision};
use super::model::LlamaModel;
use crate::tokenizer::TokenId;

// ANCHOR: loss
/// 점수 목록에서 `target`번 토큰의 손실, 곧 `-ln(그 토큰의 확률)`.
///
/// 소프트맥스로 확률을 구한 뒤 로그를 취하지 않고 한 번에 계산한다.
/// `ln(확률) = 점수 − 가장 큰 점수 − ln(Σ exp(점수 − 가장 큰 점수))`다.
/// 합은 `f64`로 모은다. 12만 개를 더하므로 `f32`로는 자릿수가 모자란다
/// (12.6절).
pub fn token_loss(logits: &[f32], target: usize) -> f64 {
    let largest = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let total: f64 = logits
        .iter()
        .map(|&logit| f64::from(logit - largest).exp())
        .sum();
    total.ln() - f64::from(logits[target] - largest)
}
// ANCHOR_END: loss

// ANCHOR: losses
/// 토큰마다의 손실을 모은 것.
#[derive(Debug, Clone, Default)]
pub struct Losses {
    /// 채점한 토큰마다의 손실. 글에 나온 순서다.
    pub values: Vec<f64>,
}

impl Losses {
    /// 손실의 평균. 토큰 하나를 맞히는 데 드는 정보의 양(단위는 nat)이다.
    pub fn mean(&self) -> f64 {
        self.values.iter().sum::<f64>() / self.values.len() as f64
    }

    /// 퍼플렉서티: 평균 손실에 지수 함수를 씌운 것.
    pub fn perplexity(&self) -> f64 {
        self.mean().exp()
    }

    /// 평균 손실의 표준 오차. 채점한 토큰들이 표본이라고 볼 때, 평균이
    /// 얼마나 흔들릴 수 있는지를 어림한 값이다.
    pub fn standard_error(&self) -> f64 {
        standard_error(&self.values)
    }

    /// 다른 손실들과의 차이: `(평균의 차이, 그 표준 오차)`.
    ///
    /// 같은 토큰들을 채점한 것이어야 한다. 토큰마다 차이를 먼저 구하고 그
    /// 차이들의 평균과 표준 오차를 낸다. 두 모델이 같은 토큰에서 함께
    /// 어려워하고 함께 쉬워하므로, 차이는 각각의 평균보다 훨씬 덜 흔들린다.
    pub fn difference_from(&self, base: &Self) -> (f64, f64) {
        assert_eq!(
            self.values.len(),
            base.values.len(),
            "같은 토큰들이어야 합니다"
        );
        let differences: Vec<f64> = self
            .values
            .iter()
            .zip(&base.values)
            .map(|(a, b)| a - b)
            .collect();
        let mean = differences.iter().sum::<f64>() / differences.len() as f64;
        (mean, standard_error(&differences))
    }
}

/// 값들의 평균의 표준 오차: 표준편차를 개수의 제곱근으로 나눈 것.
fn standard_error(values: &[f64]) -> f64 {
    let count = values.len() as f64;
    let mean = values.iter().sum::<f64>() / count;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (count - 1.0);
    (variance / count).sqrt()
}
// ANCHOR_END: losses

// ANCHOR: protocol
/// 채점하는 방법.
#[derive(Debug, Clone, Copy)]
pub struct Protocol {
    /// 한 번에 모델에 넣는 토큰의 수. 글을 이 길이로 끊는다.
    pub context: usize,
    /// 채점할 조각의 수. 글이 그보다 짧으면 있는 만큼만 한다.
    pub chunks: usize,
    /// 조각마다 첫 토큰 자리에 넣을 글머리 토큰.
    pub bos: Option<TokenId>,
    /// KV 캐시의 정밀도.
    pub kv: KvPrecision,
}

/// 글의 토큰들로 손실을 잰다.
///
/// llama.cpp의 `llama-perplexity`와 같은 방법이다. 글을 `context`개씩
/// 끊고, 조각마다 첫 토큰을 글머리 토큰으로 바꿔 모델에 넣는다. **뒤
/// 절반의 자리만** 채점한다. 앞 절반은 모델이 글의 흐름을 잡도록 주는
/// 문맥이다. `p`번째 자리의 점수로 `p + 1`번째 토큰을 채점하므로 조각
/// 하나에서 `context ÷ 2 − 1`개를 채점한다.
pub fn evaluate(model: &LlamaModel, tokens: &[TokenId], protocol: Protocol) -> Result<Losses> {
    let (context, first) = (protocol.context, protocol.context / 2);
    let mut losses = Losses::default();
    for chunk in tokens.chunks_exact(context).take(protocol.chunks) {
        let mut input = chunk.to_vec();
        if let Some(bos) = protocol.bos {
            input[0] = bos;
        }
        // 조각 전체를 블록에 통과시키고, 채점할 자리의 줄만 점수로 바꾼다.
        let mut cache = KvCache::with_precision(&model.config, protocol.kv);
        let hidden = model.hidden_cached(&input, &mut cache)?;
        let logits = model.logits(&hidden.slice(0, first..context - 1)?)?;
        let logits = logits.as_slice::<f32>()?;
        let vocab = model.config.vocab_size;

        // 자리마다 따로 계산할 수 있으므로 18장의 풀에 나눈다.
        let mut scored = vec![0.0_f64; context - 1 - first];
        let pool = threads::global().pool();
        pool.run_chunks(&mut scored, 1, pool.threads(), &|index, loss| {
            let target = chunk[first + index + 1] as usize;
            loss[0] = token_loss(&logits[index * vocab..(index + 1) * vocab], target);
        });
        losses.values.extend(scored);
    }
    Ok(losses)
}
// ANCHOR_END: protocol

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_loss_matches_the_definition() {
        // 점수가 모두 같으면 확률이 1/n이고 손실은 ln(n)이다.
        let even = [0.5_f32; 8];
        assert!((token_loss(&even, 3) - 8.0_f64.ln()).abs() < 1e-12);
        // 점수 둘: 확률은 e^2 / (e^2 + e^0).
        let two = [2.0_f32, 0.0];
        let probability = 2.0_f64.exp() / (2.0_f64.exp() + 1.0);
        assert!((token_loss(&two, 0) + probability.ln()).abs() < 1e-12);
        assert!((token_loss(&two, 1) + (1.0 - probability).ln()).abs() < 1e-12);
        // 점수에 같은 값을 더해도 손실은 같다. 큰 점수에서도 넘치지 않는다.
        let shifted = [1002.0_f32, 1000.0];
        assert!((token_loss(&shifted, 0) - token_loss(&two, 0)).abs() < 1e-12);
    }

    #[test]
    fn losses_summarize_and_compare() {
        let base = Losses {
            values: vec![1.0, 2.0, 3.0, 4.0],
        };
        assert_eq!(base.mean(), 2.5);
        assert!((base.perplexity() - 2.5_f64.exp()).abs() < 1e-12);
        // 표준편차 1.291, 개수 4: 표준 오차 0.6455.
        assert!((base.standard_error() - 0.645_497).abs() < 1e-6);
        // 토큰마다 0.1씩만 다르면 차이의 표준 오차는 0이다. 각각의 평균은
        // 0.65씩 흔들리는데도 그렇다.
        let other = Losses {
            values: base.values.iter().map(|v| v + 0.1).collect(),
        };
        let (difference, error) = other.difference_from(&base);
        assert!((difference - 0.1).abs() < 1e-12);
        assert!(error < 1e-12);
    }
}
