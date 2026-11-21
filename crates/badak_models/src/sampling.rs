//! 샘플링: 점수에서 다음 토큰 하나를 뽑는다 (12장).
//!
//! 점수를 차례로 손질하는 단계 넷과, 손질한 결과에서 하나를 뽑는 단계로
//! 이루어진다. 순서는 반복 벌점, 온도, top-k, top-p다. 설정에 따라 온도를
//! 맨 뒤로 보낼 수 있다.

use badak_core::prob::{argmax, softmax};
use badak_core::random::Pcg64;

use crate::tokenizer::TokenId;

// ANCHOR: config
/// 샘플링의 설정. 기본값은 "아무 손질 없이 모델의 확률대로 뽑는다"이다.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplerConfig {
    /// 온도. 1보다 작으면 유력한 토큰에 확률이 몰리고, 크면 고르게 퍼진다.
    /// 0이면 늘 점수가 가장 높은 토큰을 고른다.
    pub temperature: f32,
    /// 점수가 높은 순서로 이만큼만 남긴다. 0이면 자르지 않는다.
    pub top_k: usize,
    /// 확률이 높은 순서로, 합이 이 값에 이를 때까지만 남긴다. 1이면 자르지
    /// 않는다.
    pub top_p: f32,
    /// 이미 나온 토큰의 점수를 깎는 정도. 1이면 깎지 않는다.
    pub repetition_penalty: f32,
    /// 참이면 top-p로 후보를 추린 **뒤에** 온도를 적용한다. 거짓이면 온도를
    /// 먼저 적용한다. 엔진마다 기본값이 다르다(12.4절).
    pub temperature_last: bool,
    /// 난수의 시드.
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            temperature_last: false,
            seed: 0,
        }
    }
}
// ANCHOR_END: config

/// 후보 토큰 하나와 그 확률.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Candidate {
    /// 토큰 번호.
    pub id: TokenId,
    /// 손질을 마친 뒤의 확률.
    pub probability: f32,
}

// ANCHOR: penalty
/// 반복 벌점. 이미 나온 토큰의 점수를 낮춘다.
///
/// 점수가 양수이면 `penalty`로 나누고 음수이면 곱한다. 어느 쪽이든 점수가
/// 내려간다. 같은 토큰이 여러 번 나왔어도 한 번만 깎는다.
pub fn penalize_repetition(logits: &mut [f32], history: &[TokenId], penalty: f32) {
    let mut seen = vec![false; logits.len()];
    for &token in history {
        let token = token as usize;
        if token < logits.len() && !std::mem::replace(&mut seen[token], true) {
            let logit = &mut logits[token];
            *logit = if *logit > 0.0 {
                *logit / penalty
            } else {
                *logit * penalty
            };
        }
    }
}
// ANCHOR_END: penalty

// ANCHOR: top_k
/// 점수가 높은 `k`개를 골라, 높은 순서로 늘어놓는다. `k`가 0이면 전부다.
///
/// 돌려주는 값은 (토큰 번호, 점수)의 목록이다.
pub fn top_k(logits: &[f32], k: usize) -> Vec<(TokenId, f32)> {
    let mut candidates: Vec<(TokenId, f32)> = logits
        .iter()
        .enumerate()
        .map(|(id, &logit)| (id as TokenId, logit))
        .collect();
    let descending = |a: &(TokenId, f32), b: &(TokenId, f32)| b.1.total_cmp(&a.1);
    if k > 0 && k < candidates.len() {
        // 전체를 정렬하지 않는다. 높은 `k`개를 앞으로 모으기만 하고 버린다.
        candidates.select_nth_unstable_by(k - 1, descending);
        candidates.truncate(k);
    }
    candidates.sort_unstable_by(descending);
    candidates
}
// ANCHOR_END: top_k

// ANCHOR: top_p
/// 확률이 높은 순서로 놓인 후보에서, 확률의 합이 `p`에 이를 때까지만
/// 남긴다. 합이 `p`를 넘게 만든 후보까지 포함한다.
///
/// 돌려주는 값은 남길 후보의 수다. 적어도 하나는 남긴다.
pub fn top_p(probabilities: &[f32], p: f32) -> usize {
    let mut cumulative = 0.0;
    for (index, probability) in probabilities.iter().enumerate() {
        cumulative += probability;
        if cumulative >= p {
            return index + 1;
        }
    }
    probabilities.len().max(1)
}
// ANCHOR_END: top_p

// ANCHOR: sampler
/// 점수에서 토큰을 뽑는 도구. 설정과 난수 생성기를 쥐고 있다.
#[derive(Debug, Clone)]
pub struct Sampler {
    config: SamplerConfig,
    rng: Pcg64,
}

impl Sampler {
    /// 설정으로 샘플러를 만든다. 난수 생성기는 설정의 시드로 시작한다.
    pub fn new(config: SamplerConfig) -> Self {
        Self {
            config,
            rng: Pcg64::new(config.seed),
        }
    }

    /// 점수를 손질해, 뽑힐 수 있는 후보와 그 확률을 구한다.
    ///
    /// 후보는 확률이 높은 순서로 놓이고 확률의 합은 1이다. `history`는
    /// 지금까지의 토큰이고 반복 벌점에 쓴다.
    pub fn distribution(&self, logits: &[f32], history: &[TokenId]) -> Vec<Candidate> {
        let config = &self.config;
        let mut logits = logits.to_vec();

        // 1. 반복 벌점.
        if config.repetition_penalty != 1.0 {
            penalize_repetition(&mut logits, history, config.repetition_penalty);
        }

        // 온도가 0이면 뽑을 것이 없다. 점수가 가장 높은 토큰 하나다.
        if config.temperature <= 0.0 {
            let id = argmax(&logits).expect("점수에 유효한 값이 있다") as TokenId;
            return vec![Candidate {
                id,
                probability: 1.0,
            }];
        }

        // 2. top-k: 점수가 높은 것만 남긴다. 온도는 점수의 순서를 바꾸지
        //    않으므로, 온도보다 먼저 해도 남는 후보는 같다.
        let mut candidates = top_k(&logits, config.top_k);
        let probabilities = |candidates: &[(TokenId, f32)], temperature: f32| {
            let scores: Vec<f32> = candidates
                .iter()
                .map(|&(_, logit)| logit / temperature)
                .collect();
            softmax(&scores)
        };

        // 3. top-p: 확률의 합이 `top_p`에 이를 때까지만 남긴다. 이때의 확률을
        //    온도를 적용한 것으로 볼지, 적용하기 전의 것으로 볼지가 설정이다.
        if config.top_p < 1.0 {
            let temperature = if config.temperature_last {
                1.0
            } else {
                config.temperature
            };
            let keep = top_p(&probabilities(&candidates, temperature), config.top_p);
            candidates.truncate(keep);
        }

        // 4. 온도: 남은 후보의 점수를 온도로 나누어 확률로 바꾼다.
        let probabilities = probabilities(&candidates, config.temperature);
        candidates
            .iter()
            .zip(probabilities)
            .map(|(&(id, _), probability)| Candidate { id, probability })
            .collect()
    }

    /// 다음 토큰 하나를 뽑는다.
    pub fn sample(&mut self, logits: &[f32], history: &[TokenId]) -> TokenId {
        let candidates = self.distribution(logits, history);
        pick(&candidates, self.rng.next_f64())
    }
}

/// 0 이상 1 미만의 수 `u`로 후보 하나를 고른다.
///
/// 후보들의 확률을 길이로 삼아 0부터 1까지의 선분에 차례로 늘어놓고, `u`가
/// 떨어진 자리의 후보를 고른다.
pub fn pick(candidates: &[Candidate], u: f64) -> TokenId {
    let mut cumulative = 0.0;
    for candidate in candidates {
        cumulative += f64::from(candidate.probability);
        if u < cumulative {
            return candidate.id;
        }
    }
    // 반올림 때문에 합이 1에 조금 못 미쳐 여기까지 올 수 있다.
    candidates.last().expect("후보가 적어도 하나 있다").id
}
// ANCHOR_END: sampler

#[cfg(test)]
mod tests {
    use super::*;

    fn probabilities(config: SamplerConfig, logits: &[f32], history: &[TokenId]) -> Vec<f32> {
        // 토큰 번호 순서의 확률로 펼친다. 걸러진 토큰은 0이다.
        let mut dense = vec![0.0; logits.len()];
        for candidate in Sampler::new(config).distribution(logits, history) {
            dense[candidate.id as usize] = candidate.probability;
        }
        dense
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() < 1e-6, "{actual:?} != {expected:?}");
        }
    }

    const LOGITS: [f32; 4] = [2.0, 1.0, 0.0, -1.0];

    #[test]
    fn default_config_is_plain_softmax() {
        let got = probabilities(SamplerConfig::default(), &LOGITS, &[]);
        assert_close(&got, &softmax(&LOGITS));
    }

    #[test]
    fn temperature_sharpens_and_flattens() {
        let with = |temperature| {
            let config = SamplerConfig {
                temperature,
                ..SamplerConfig::default()
            };
            probabilities(config, &LOGITS, &[])
        };
        let (cold, plain, hot) = (with(0.5), with(1.0), with(2.0));
        assert!(cold[0] > plain[0] && plain[0] > hot[0]);
        // 온도 0.5는 점수를 두 배로 키운 것과 같다.
        assert_close(&cold, &softmax(&[4.0, 2.0, 0.0, -2.0]));
        // 온도 0은 가장 높은 토큰 하나만 남긴다.
        assert_close(&with(0.0), &[1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn top_k_keeps_the_k_largest() {
        let config = SamplerConfig {
            top_k: 2,
            ..SamplerConfig::default()
        };
        let got = probabilities(config, &[0.0, 2.0, -1.0, 1.0], &[]);
        let expected = softmax(&[2.0, 1.0]);
        assert_close(&got, &[0.0, expected[0], 0.0, expected[1]]);
    }

    #[test]
    fn top_p_keeps_the_smallest_set_reaching_p() {
        // 확률이 0.5, 0.3, 0.15, 0.05가 되는 점수.
        let logits: Vec<f32> = [0.5_f32, 0.3, 0.15, 0.05].iter().map(|p| p.ln()).collect();
        let with = |top_p| {
            let config = SamplerConfig {
                top_p,
                ..SamplerConfig::default()
            };
            probabilities(config, &logits, &[])
        };
        // 0.5만으로는 0.6에 못 미치므로 둘째까지 남긴다.
        assert_close(&with(0.6), &[0.625, 0.375, 0.0, 0.0]);
        assert_close(&with(0.4), &[1.0, 0.0, 0.0, 0.0]);
        assert_close(&with(0.9), &[0.5 / 0.95, 0.3 / 0.95, 0.15 / 0.95, 0.0]);
        assert_eq!(top_p(&[], 0.5), 1);
    }

    #[test]
    fn temperature_last_cuts_before_flattening() {
        // 확률이 0.5, 0.3, 0.15, 0.05가 되는 점수에 온도 2와 top-p 0.75.
        let logits: Vec<f32> = [0.5_f32, 0.3, 0.15, 0.05].iter().map(|p| p.ln()).collect();
        let with = |temperature_last| {
            let config = SamplerConfig {
                temperature: 2.0,
                top_p: 0.75,
                temperature_last,
                ..SamplerConfig::default()
            };
            probabilities(config, &logits, &[])
        };
        // 온도가 먼저면 분포가 펴진 뒤에 자르므로 셋이 남는다.
        assert_eq!(with(false).iter().filter(|&&p| p > 0.0).count(), 3);
        // 온도가 나중이면 원래 분포로 자르므로 둘이 남고, 그 둘에 온도가 걸린다.
        let roots = [0.5_f32.sqrt(), 0.3_f32.sqrt()];
        let total = roots[0] + roots[1];
        assert_close(&with(true), &[roots[0] / total, roots[1] / total, 0.0, 0.0]);
    }

    #[test]
    fn repetition_penalty_lowers_seen_tokens_once() {
        let mut logits = [2.0, -2.0, 1.0];
        penalize_repetition(&mut logits, &[0, 1, 0, 0, 99], 2.0);
        assert_eq!(logits, [1.0, -4.0, 1.0]);
    }

    #[test]
    fn pick_follows_the_cumulative_line() {
        let candidates =
            [(7, 0.5), (3, 0.3), (9, 0.2)].map(|(id, probability)| Candidate { id, probability });
        assert_eq!(pick(&candidates, 0.0), 7);
        assert_eq!(pick(&candidates, 0.499), 7);
        assert_eq!(pick(&candidates, 0.5), 3);
        assert_eq!(pick(&candidates, 0.81), 9);
        assert_eq!(pick(&candidates, 0.999_999_99), 9);
    }

    #[test]
    fn samples_follow_the_distribution() {
        let mut sampler = Sampler::new(SamplerConfig {
            seed: 3,
            ..SamplerConfig::default()
        });
        let expected = softmax(&LOGITS);
        let count = 200_000;
        let mut hits = [0_u32; 4];
        for _ in 0..count {
            hits[sampler.sample(&LOGITS, &[]) as usize] += 1;
        }
        for (hit, probability) in hits.iter().zip(expected) {
            let frequency = f64::from(*hit) / f64::from(count);
            assert!(
                (frequency - f64::from(probability)).abs() < 0.005,
                "{hits:?}"
            );
        }
    }

    #[test]
    fn same_seed_gives_the_same_tokens() {
        let draw = |seed| {
            let mut sampler = Sampler::new(SamplerConfig {
                seed,
                ..SamplerConfig::default()
            });
            (0..50)
                .map(|_| sampler.sample(&LOGITS, &[]))
                .collect::<Vec<_>>()
        };
        assert_eq!(draw(11), draw(11));
        assert_ne!(draw(11), draw(12));
    }
}
