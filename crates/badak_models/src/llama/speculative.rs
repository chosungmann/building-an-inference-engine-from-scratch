//! 추측 디코딩 (41장).
//!
//! 큰 모델로 토큰 하나를 쓰는 값은 가중치를 한 번 읽는 값이다(15장).
//! 그리고 40장에서 본 대로, 그 김에 토큰 여럿을 함께 계산하는 것은 거의
//! 공짜다. 그러니 **다음에 올 토큰 몇 개를 미리 짐작해 두고**, 큰 모델이
//! 순전파 한 번으로 그 짐작들을 한꺼번에 채점하게 한다. 짐작이 맞은
//! 데까지는 순전파 한 번에 토큰 여럿을 얻는다. 틀린 데부터는 버린다.
//!
//! 채점은 큰 모델이 하므로 **결과는 큰 모델이 혼자 쓴 것과 같다.**
//! 달라지는 것은 속도뿐이다.

use badak_formats::Result;

use super::cache::KvCache;
use super::model::LlamaModel;
use crate::tokenizer::TokenId;

/// 점수가 가장 높은 토큰.
fn argmax(logits: &[f32]) -> TokenId {
    let mut best = 0;
    for (index, &logit) in logits.iter().enumerate() {
        if logit > logits[best] {
            best = index;
        }
    }
    best as TokenId
}

/// 두 토큰열이 앞에서부터 몇 개 같은가.
fn common(a: &[TokenId], b: &[TokenId]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

// ANCHOR: proposer
/// 다음에 올 토큰들을 짐작하는 것.
pub trait Proposer {
    /// `tokens` 뒤에 올 토큰을 `count`개까지 짐작한다. 짐작할 것이 없으면
    /// 빈 목록을 돌려준다.
    fn propose(&mut self, tokens: &[TokenId], count: usize) -> Result<Vec<TokenId>>;
}

/// 작은 모델로 짐작한다. 점수가 가장 높은 토큰을 이어 쓴다.
pub struct DraftModel<'a> {
    model: &'a LlamaModel,
    cache: KvCache,
    /// 캐시에 든 토큰들.
    fed: Vec<TokenId>,
    /// 지금까지 돈 순전파의 수.
    pub forwards: usize,
}

impl<'a> DraftModel<'a> {
    /// 초안을 쓸 모델을 정한다.
    pub fn new(model: &'a LlamaModel) -> Self {
        Self {
            cache: KvCache::new(&model.config),
            model,
            fed: Vec::new(),
            forwards: 0,
        }
    }
}

impl Proposer for DraftModel<'_> {
    fn propose(&mut self, tokens: &[TokenId], count: usize) -> Result<Vec<TokenId>> {
        // 지난번에 쓴 초안 가운데 버려진 것을 캐시에서 걷어 낸다. 마지막 토큰
        // 하나는 늘 다시 넣는다. 그래야 다음 토큰의 점수가 나온다.
        let keep = common(&self.fed, tokens).min(tokens.len().saturating_sub(1));
        self.cache.truncate(keep);
        self.fed.truncate(keep);

        let mut input = tokens[keep..].to_vec();
        let mut draft = Vec::with_capacity(count);
        for _ in 0..count {
            let logits = self.model.forward_cached(&input, &mut self.cache)?;
            self.forwards += 1;
            self.fed.extend_from_slice(&input);
            let next = argmax(logits.as_slice()?);
            draft.push(next);
            if self.model.config.eos_token_ids.contains(&next) {
                break;
            }
            input = vec![next];
        }
        Ok(draft)
    }
}

/// 모델 없이 짐작한다. 지금까지의 글에서 **방금 쓴 몇 토큰과 같은 자리**를
/// 찾아, 그 뒤에 왔던 토큰들을 그대로 내놓는다.
///
/// 글이 제 앞부분을 되풀이할 때 맞는다. 문서를 고쳐 쓰거나 간추리거나
/// 인용하는 일이 그렇다.
pub struct Lookup {
    /// 맞춰 볼 토큰의 수. 긴 것부터 줄여 가며 찾는다.
    pub longest: usize,
    /// 이보다 짧게 맞은 것은 쓰지 않는다.
    pub shortest: usize,
}

impl Default for Lookup {
    fn default() -> Self {
        Self {
            longest: 4,
            shortest: 2,
        }
    }
}

impl Proposer for Lookup {
    fn propose(&mut self, tokens: &[TokenId], count: usize) -> Result<Vec<TokenId>> {
        for length in (self.shortest..=self.longest.min(tokens.len().saturating_sub(1))).rev() {
            let tail = &tokens[tokens.len() - length..];
            // 가장 가까운(가장 뒤의) 자리부터 찾는다. 끝 자신은 뺀다.
            let found = (0..tokens.len() - length)
                .rev()
                .find(|&start| &tokens[start..start + length] == tail);
            if let Some(start) = found {
                let after = start + length;
                return Ok(tokens[after..(after + count).min(tokens.len())].to_vec());
            }
        }
        Ok(Vec::new())
    }
}
// ANCHOR_END: proposer

/// 추측 디코딩 한 번의 기록.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Record {
    /// 쓴 토큰들.
    pub tokens: Vec<TokenId>,
    /// 큰 모델의 순전파의 수.
    pub rounds: usize,
    /// 짐작한 토큰의 수.
    pub proposed: usize,
    /// 그 가운데 맞은 토큰의 수.
    pub accepted: usize,
}

// ANCHOR: speculate
/// 글을 이어 쓴다. `proposer`가 짐작하고 `target`이 채점한다. 결과는
/// `target`이 점수가 가장 높은 토큰을 골라 혼자 쓴 것과 같다.
pub fn generate_speculative(
    target: &LlamaModel,
    proposer: &mut dyn Proposer,
    prompt: &[TokenId],
    max_new_tokens: usize,
    lookahead: usize,
    cache: &mut KvCache,
    mut on_token: impl FnMut(TokenId),
) -> Result<Record> {
    let mut tokens = prompt.to_vec();
    let mut record = Record::default();
    let end = |token: TokenId| target.config.eos_token_ids.contains(&token);
    while record.tokens.len() < max_new_tokens {
        // 1. 짐작한다. 남은 토큰 수보다 많이 짐작할 까닭은 없다.
        let room = max_new_tokens - record.tokens.len();
        let draft = proposer.propose(&tokens, lookahead.min(room.saturating_sub(1)))?;
        record.proposed += draft.len();

        // 2. 채점한다. 캐시에 없는 토큰들과 짐작들을 **한 번에** 넣는다.
        //    줄마다 "이 토큰 다음에 올 토큰"의 점수가 나온다. 마지막
        //    `draft.len() + 1`줄이 필요한 것이다.
        let known = cache.len();
        let mut input = tokens[known..].to_vec();
        input.extend_from_slice(&draft);
        let hidden = target.hidden_cached(&input, cache)?;
        record.rounds += 1;
        let rows = hidden.shape()[0];
        let scored = target.logits(&hidden.slice(0, rows - draft.len() - 1..rows)?)?;
        let width = scored.shape()[1];
        let scored = scored.as_slice::<f32>()?;
        let choice = |row: usize| argmax(&scored[row * width..(row + 1) * width]);

        // 3. 짐작이 큰 모델의 선택과 같은 데까지 받아들인다. 그다음 토큰은
        //    큰 모델이 고른 것을 쓴다. 짐작이 하나도 안 맞아도 토큰 하나는
        //    얻는다.
        let mut kept = 0;
        while kept < draft.len() && draft[kept] == choice(kept) && !end(draft[kept]) {
            kept += 1;
        }
        record.accepted += kept;
        let fresh: Vec<TokenId> = draft[..kept]
            .iter()
            .copied()
            .chain([choice(kept)])
            .collect();

        // 4. 틀린 짐작들을 캐시에서 걷어 낸다.
        cache.truncate(tokens.len() + kept);
        for token in fresh {
            if end(token) {
                return Ok(record);
            }
            tokens.push(token);
            record.tokens.push(token);
            on_token(token);
        }
    }
    Ok(record)
}
// ANCHOR_END: speculate

// ANCHOR: judge
/// 확률로 뽑을 때의 채점 (추측 샘플링).
///
/// 초안이 분포 `q`에서 토큰 `drafted`를 뽑았고, 큰 모델의 분포는 `p`다.
/// `min(1, p/q)`의 확률로 그 토큰을 받아들인다. 받아들이지 않으면
/// `max(0, p − q)`를 다시 고르게 만든 분포에서 새로 뽑는다. 이렇게 고른
/// 토큰은 **`p`에서 바로 뽑은 것과 같은 분포**를 따른다.
///
/// `accept`와 `pick`은 0 이상 1 미만의 난수다. 돌려주는 것은 (받아들였는가,
/// 토큰)이다.
pub fn judge(p: &[f32], q: &[f32], drafted: usize, accept: f64, pick: f64) -> (bool, usize) {
    let ratio = if q[drafted] > 0.0 {
        f64::from(p[drafted]) / f64::from(q[drafted])
    } else {
        1.0
    };
    if accept < ratio {
        return (true, drafted);
    }
    // 큰 모델이 초안보다 더 주고 싶어 한 만큼만 남긴 분포.
    let extra: Vec<f64> = p
        .iter()
        .zip(q)
        .map(|(&p, &q)| (f64::from(p) - f64::from(q)).max(0.0))
        .collect();
    let total: f64 = extra.iter().sum();
    let mut left = pick * total;
    for (token, &mass) in extra.iter().enumerate() {
        if left < mass {
            return (false, token);
        }
        left -= mass;
    }
    // 반올림 때문에 끝까지 왔으면 남긴 것이 있는 마지막 토큰.
    let last = extra
        .iter()
        .rposition(|&mass| mass > 0.0)
        .unwrap_or(drafted);
    (false, last)
}
// ANCHOR_END: judge

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_copies_what_followed_the_same_tokens_before() {
        let mut lookup = Lookup::default();
        // … 7 8 9 다음에 1 2 3이 왔었다. 다시 7 8 9를 썼다.
        let tokens = [5, 7, 8, 9, 1, 2, 3, 4, 6, 7, 8, 9];
        assert_eq!(lookup.propose(&tokens, 3).unwrap(), [1, 2, 3]);
        assert_eq!(
            lookup.propose(&tokens, 10).unwrap(),
            [1, 2, 3, 4, 6, 7, 8, 9]
        );
        // 같은 것이 여러 번 나왔으면 가장 가까운 것을 따른다.
        let tokens = [1, 2, 10, 1, 2, 20, 1, 2];
        assert_eq!(lookup.propose(&tokens, 1).unwrap(), [20]);
        // 맞는 자리가 없거나 한 토큰만 맞으면 짐작하지 않는다.
        assert!(lookup.propose(&[1, 2, 3, 4], 3).unwrap().is_empty());
        assert!(lookup.propose(&[1, 9, 2, 9], 3).unwrap().is_empty());
        assert!(lookup.propose(&[], 3).unwrap().is_empty());
    }

    #[test]
    fn judged_tokens_follow_the_target_distribution() {
        // 초안의 분포와 큰 모델의 분포가 꽤 다르다.
        let p = [0.5_f32, 0.1, 0.0, 0.4];
        let q = [0.2_f32, 0.6, 0.2, 0.0];
        // 난수 셋(초안이 뽑는 것, 받아들일지, 다시 뽑는 것)을 고르게 훑으며
        // 결과의 분포를 센다.
        let steps = 200;
        let mut counts = [0.0_f64; 4];
        let mut accepted = 0.0;
        for a in 0..steps {
            // 초안이 q에서 뽑는다.
            let u = (a as f64 + 0.5) / steps as f64;
            let mut left = u;
            let drafted = q
                .iter()
                .position(|&mass| {
                    let hit = left < f64::from(mass);
                    left -= f64::from(mass);
                    hit
                })
                .unwrap_or(1);
            for b in 0..steps {
                for c in 0..steps {
                    let (kept, token) = judge(
                        &p,
                        &q,
                        drafted,
                        (b as f64 + 0.5) / steps as f64,
                        (c as f64 + 0.5) / steps as f64,
                    );
                    counts[token] += 1.0;
                    accepted += f64::from(u8::from(kept));
                }
            }
        }
        let total = (steps * steps * steps) as f64;
        for (count, &wanted) in counts.iter().zip(&p) {
            assert!(
                (count / total - f64::from(wanted)).abs() < 0.01,
                "{counts:?}"
            );
        }
        // 받아들이는 비율은 두 분포가 겹치는 만큼이다: Σ min(p, q) = 0.3.
        assert!((accepted / total - 0.3).abs() < 0.01);
    }
}
