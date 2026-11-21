//! 페이지로 나눈 KV 캐시와, 여러 글을 한 번에 계산하는 블록 (40장).
//!
//! 11장의 [`KvCache`](super::cache::KvCache)는 글 하나의 키와 값을 한
//! 줄로 이어 담는다. 글이 하나일 때는 그것이 가장 빠르다. 글이 여럿이
//! 되면 두 가지가 아쉬워진다. 같은 앞부분으로 시작하는 글들이 그
//! 앞부분의 키와 값을 저마다 따로 들고 있어야 하고, 글마다 얼마나
//! 길어질지 몰라 자리를 넉넉히 잡아 두어야 한다.
//!
//! 여기서는 캐시를 **토큰 열여섯 개짜리 페이지**로 나눈다. 글은 페이지의
//! 목록을 쥔다. 앞부분이 같은 글들은 같은 페이지를 함께 가리키고, 누가
//! 그 페이지에 쓰려 할 때에만 베낀다. 운영체제가 메모리를 다루는 방법
//! 그대로다.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use badak_core::prob::softmax_in_place;
use badak_core::simd::{self, Backend, Walk};
use badak_core::{Tensor, arena, ops};
use badak_formats::{Error, Result};

use super::{BlockWeights, LlamaConfig, mlp, rope};
use crate::tokenizer::TokenId;

/// 페이지 하나에 담는 토큰의 수.
pub const PAGE_TOKENS: usize = 16;

/// 풀이 페이지들과 함께 쥐는 것.
struct Shared {
    /// 돌려받은 페이지의 자리. 다음에 다시 내준다.
    free: Mutex<Vec<Vec<f32>>>,
    /// 지금 쓰이고 있는 페이지의 수.
    used: AtomicUsize,
    /// 가장 많이 쓰였을 때의 수.
    peak: AtomicUsize,
    /// 내줄 수 있는 페이지의 한도.
    limit: usize,
    layers: usize,
    kv_heads: usize,
    dim: usize,
}

impl Shared {
    /// 페이지 하나의 길이(수의 개수). 블록마다, 키와 값마다, 헤드마다
    /// 토큰 `PAGE_TOKENS`개의 벡터가 들어간다.
    fn page_len(&self) -> usize {
        self.layers * 2 * self.kv_heads * PAGE_TOKENS * self.dim
    }
}

// ANCHOR: pool
/// 페이지를 내주고 돌려받는 곳. 한도가 있다.
#[derive(Clone)]
pub struct PagePool(Arc<Shared>);

/// 페이지 하나. 토큰 `PAGE_TOKENS`개의, 모든 블록의 키와 값.
///
/// 버려지면 자리가 풀로 돌아간다.
pub struct Page {
    data: Vec<f32>,
    pool: Arc<Shared>,
}

impl Drop for Page {
    fn drop(&mut self) {
        self.pool.used.fetch_sub(1, Ordering::AcqRel);
        if let Ok(mut free) = self.pool.free.lock() {
            free.push(std::mem::take(&mut self.data));
        }
    }
}

impl PagePool {
    /// 페이지를 `limit`개까지 내주는 풀을 만든다.
    pub fn new(config: &LlamaConfig, limit: usize) -> Self {
        Self(Arc::new(Shared {
            free: Mutex::new(Vec::new()),
            used: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            limit,
            layers: config.num_layers,
            kv_heads: config.num_kv_heads,
            dim: config.head_dim(),
        }))
    }

    /// 지금 쓰이고 있는 페이지의 수.
    pub fn used(&self) -> usize {
        self.0.used.load(Ordering::Acquire)
    }

    /// 가장 많이 쓰였을 때의 수.
    pub fn peak(&self) -> usize {
        self.0.peak.load(Ordering::Acquire)
    }

    /// 더 내줄 수 있는 페이지의 수.
    pub fn available(&self) -> usize {
        self.0.limit.saturating_sub(self.used())
    }

    /// 페이지 하나가 차지하는 메모리(바이트).
    pub fn page_bytes(&self) -> usize {
        self.0.page_len() * size_of::<f32>()
    }

    /// 페이지 하나를 내준다. 한도에 닿았으면 오류다.
    fn take(&self) -> Result<Page> {
        let shared = &self.0;
        let used = shared.used.fetch_add(1, Ordering::AcqRel) + 1;
        if used > shared.limit {
            shared.used.fetch_sub(1, Ordering::AcqRel);
            return Err(Error::Unsupported("KV 캐시의 페이지가 모자랍니다".into()));
        }
        shared.peak.fetch_max(used, Ordering::AcqRel);
        let recycled = shared.free.lock().ok().and_then(|mut free| free.pop());
        Ok(Page {
            data: recycled.unwrap_or_else(|| vec![0.0; shared.page_len()]),
            pool: Arc::clone(shared),
        })
    }
}
// ANCHOR_END: pool

// ANCHOR: cache
/// 글 하나의 KV 캐시. 페이지의 목록이다.
pub struct PagedCache {
    pool: PagePool,
    pages: Vec<Arc<Page>>,
    /// 지금까지 넣은 토큰의 수.
    tokens: usize,
}

impl PagedCache {
    /// 빈 캐시.
    pub fn new(pool: &PagePool) -> Self {
        Self {
            pool: pool.clone(),
            pages: Vec::new(),
            tokens: 0,
        }
    }

    /// 캐시에 든 토큰의 수.
    pub fn len(&self) -> usize {
        self.tokens
    }

    /// 캐시가 비었는가.
    pub fn is_empty(&self) -> bool {
        self.tokens == 0
    }

    /// 이 캐시가 가리키는 페이지의 수.
    pub fn pages(&self) -> usize {
        self.pages.len()
    }

    /// 그 가운데 다른 캐시도 함께 가리키는 페이지의 수.
    pub fn shared_pages(&self) -> usize {
        self.pages
            .iter()
            .filter(|page| Arc::strong_count(page) > 1)
            .count()
    }

    /// 앞의 `tokens`개를 **함께 쓰는** 새 캐시를 만든다. 아무것도 베끼지
    /// 않는다. 페이지를 가리키는 것만 늘어난다.
    pub fn fork(&self, tokens: usize) -> Self {
        let tokens = tokens.min(self.tokens);
        Self {
            pool: self.pool.clone(),
            pages: self.pages[..tokens.div_ceil(PAGE_TOKENS)].to_vec(),
            tokens,
        }
    }

    /// 앞의 `tokens`개만 남긴다. 남지 않는 페이지는 놓는다.
    pub fn truncate(&mut self, tokens: usize) {
        if tokens < self.tokens {
            self.tokens = tokens;
            self.pages.truncate(tokens.div_ceil(PAGE_TOKENS));
        }
    }

    /// 토큰 `count`개를 더 넣으려면 풀에서 새로 받아야 하는 페이지의 수.
    pub fn pages_needed(&self, count: usize) -> usize {
        let fresh = (self.tokens + count).div_ceil(PAGE_TOKENS) - self.pages.len();
        // 반쯤 찬 마지막 페이지를 남과 함께 쓰고 있으면 베낄 자리도 든다.
        let copied = !self.tokens.is_multiple_of(PAGE_TOKENS)
            && Arc::strong_count(&self.pages[self.tokens / PAGE_TOKENS]) > 1;
        fresh + usize::from(copied)
    }

    /// 토큰 `count`개를 넣을 자리를 마련한다. 쓰게 될 페이지는 모두 이
    /// 캐시만 가리키게 된다.
    fn prepare(&mut self, count: usize) -> Result<()> {
        // 반쯤 찬 마지막 페이지를 남과 함께 쓰고 있으면, 쓰기 전에 베낀다.
        if !self.tokens.is_multiple_of(PAGE_TOKENS) {
            let last = &mut self.pages[self.tokens / PAGE_TOKENS];
            if Arc::get_mut(last).is_none() {
                let mut copy = self.pool.take()?;
                copy.data.copy_from_slice(&last.data);
                *last = Arc::new(copy);
            }
        }
        while self.pages.len() * PAGE_TOKENS < self.tokens + count {
            self.pages.push(Arc::new(self.pool.take()?));
        }
        Ok(())
    }
    // ANCHOR_END: cache

    /// 페이지 안에서 (블록, 키인가 값인가, 헤드)의 벡터들이 시작하는 자리.
    fn offset(&self, layer: usize, value: bool, head: usize) -> usize {
        let shape = &self.pool.0;
        ((layer * 2 + usize::from(value)) * shape.kv_heads + head) * PAGE_TOKENS * shape.dim
    }

    /// 토큰 하나의 키나 값을 적는다. [`prepare`](Self::prepare) 뒤에 부른다.
    fn write(&mut self, layer: usize, value: bool, head: usize, token: usize, vector: &[f32]) {
        let start = self.offset(layer, value, head) + (token % PAGE_TOKENS) * vector.len();
        let page = Arc::get_mut(&mut self.pages[token / PAGE_TOKENS])
            .expect("쓰기 전에 prepare가 페이지를 혼자 쓰게 해 두었다");
        page.data[start..start + vector.len()].copy_from_slice(vector);
    }

    /// 앞의 `positions`개 토큰이 걸친 페이지마다, 그 페이지에서 볼 토큰의
    /// 수와 (블록, 키인가 값인가, 헤드)의 벡터들을 내준다.
    fn pieces(
        &self,
        layer: usize,
        value: bool,
        head: usize,
        positions: usize,
    ) -> impl Iterator<Item = (usize, &[f32])> {
        let (start, dim) = (self.offset(layer, value, head), self.pool.0.dim);
        self.pages
            .iter()
            .enumerate()
            .take(positions.div_ceil(PAGE_TOKENS))
            .map(move |(index, page)| {
                let count = (positions - index * PAGE_TOKENS).min(PAGE_TOKENS);
                (count, &page.data[start..start + count * dim])
            })
    }

    // ANCHOR: kernels
    /// 질의 `count`개를 앞의 키 `positions`개와 내적한다. 결과는
    /// `scratch`의 앞쪽 `(count, positions)`에 담긴다. 그 뒤의
    /// `count × PAGE_TOKENS`는 페이지 하나의 결과를 잠깐 담는 자리다.
    ///
    /// 11장의 캐시는 키가 한 줄로 이어져 있어 커널을 한 번 불렀다.
    /// 여기서는 페이지마다 한 번씩 부른다.
    fn scores(
        &self,
        layer: usize,
        head: usize,
        queries: &[f32],
        count: usize,
        positions: usize,
        scratch: &mut [f32],
    ) {
        let (scores, piece) = scratch.split_at_mut(count * positions);
        let dim = queries.len() / count;
        let (backend, walk) = (Backend::detect(), Walk::for_vectors(count));
        let mut done = 0;
        for (keys_here, keys) in self.pieces(layer, false, head, positions) {
            let piece = &mut piece[..count * keys_here];
            simd::linear_with(backend, walk, queries, count, keys, keys_here, dim, piece);
            for (row, values) in piece.chunks_exact(keys_here).enumerate() {
                scores[row * positions + done..][..keys_here].copy_from_slice(values);
            }
            done += keys_here;
        }
    }

    /// 앞의 값 `positions`개를 비중대로 섞어 `mixed`에 더한다.
    fn mix(
        &self,
        layer: usize,
        head: usize,
        weights: &[f32],
        count: usize,
        positions: usize,
        mixed: &mut [f32],
    ) {
        let dim = mixed.len() / count;
        let mut position = 0;
        for (_, values) in self.pieces(layer, true, head, positions) {
            for value in values.chunks_exact(dim) {
                for (row, sums) in mixed.chunks_exact_mut(dim).enumerate() {
                    let weight = weights[row * positions + position];
                    for (sum, value) in sums.iter_mut().zip(value) {
                        *sum += weight * value;
                    }
                }
                position += 1;
            }
        }
    }
    // ANCHOR_END: kernels
}

/// 한 번의 순전파에 함께 넣을 글 하나. 새 토큰들과 그 글의 캐시.
pub struct Step<'a> {
    /// 캐시에 아직 없는 토큰들. 비어 있으면 안 된다.
    pub ids: &'a [TokenId],
    /// 이 글의 캐시.
    pub cache: &'a mut PagedCache,
}

/// 묶음 안에서 글 하나가 차지하는 줄들.
#[derive(Debug, Clone, Copy)]
pub(super) struct Span {
    /// 첫 줄.
    pub start: usize,
    /// 줄의 수(새 토큰의 수).
    pub count: usize,
    /// 캐시에 이미 든 토큰의 수.
    pub past: usize,
}

/// 묶음에 든 글들의 캐시에 새 토큰의 자리를 마련하고, 글마다의 줄을 정한다.
pub(super) fn prepare(steps: &mut [Step<'_>]) -> Result<Vec<Span>> {
    let mut spans = Vec::with_capacity(steps.len());
    let mut start = 0;
    for step in steps {
        assert!(
            !step.ids.is_empty(),
            "새 토큰이 없는 글은 묶음에 넣지 않는다"
        );
        step.cache.prepare(step.ids.len())?;
        spans.push(Span {
            start,
            count: step.ids.len(),
            past: step.cache.tokens,
        });
        start += step.ids.len();
    }
    Ok(spans)
}

/// 순전파가 끝난 뒤에 새 토큰을 넣었다고 기록한다.
pub(super) fn advance(steps: &mut [Step<'_>]) {
    for step in steps {
        step.cache.tokens += step.ids.len();
    }
}

// ANCHOR: attention_batch
/// 여러 글의 새 토큰들을 한 묶음으로 받는 자기 어텐션.
///
/// `x`의 줄은 글들의 새 토큰을 차례로 이어 붙인 것이다. **선형 층은 그
/// 줄들이 어느 글의 것인지 모른다.** 줄마다 따로 계산하기 때문이다.
/// 그래서 질의·키·값을 만드는 층과 결과를 모으는 층은 묶음 전체에 한 번만
/// 부른다. 글을 가려야 하는 곳은 가운데뿐이다. 토큰은 **제 글의** 앞
/// 토큰들만 본다.
fn attention_batch(
    x: &Tensor,
    weights: &BlockWeights,
    config: &LlamaConfig,
    frequencies: &[f32],
    layer: usize,
    caches: &mut [&mut PagedCache],
    spans: &[Span],
) -> Result<Tensor> {
    let _scope = badak_core::profile::scope("attention");
    let rows = x.shape()[0];
    let (heads, kv_heads, dim) = (config.num_heads, config.num_kv_heads, config.head_dim());

    // 1. 모든 글의 새 토큰에서 질의, 키, 값을 한 번에 만든다.
    let split = |projected: Tensor, count: usize| projected.reshape(&[rows, count, dim]);
    let q = split(ops::linear(x, &weights.q_proj, None)?, heads)?;
    let k = split(ops::linear(x, &weights.k_proj, None)?, kv_heads)?;
    let v = split(ops::linear(x, &weights.v_proj, None)?, kv_heads)?;

    // 2. 글마다: 질의와 키를 **그 글에서의 위치**만큼 돌리고, 키와 값을 그
    //    글의 캐시에 적는다.
    let mut queries = vec![0.0_f32; rows * heads * dim];
    for (span, cache) in spans.iter().zip(caches.iter_mut()) {
        let range = span.start..span.start + span.count;
        let q = rope::apply(&q.slice(0, range.clone())?, frequencies, span.past)?;
        queries[range.start * heads * dim..range.end * heads * dim]
            .copy_from_slice(q.as_slice::<f32>()?);
        let k = rope::apply(&k.slice(0, range.clone())?, frequencies, span.past)?;
        let v = v.slice(0, range)?.contiguous();
        let (k, v) = (k.as_slice::<f32>()?, v.as_slice::<f32>()?);
        for token in 0..span.count {
            for head in 0..kv_heads {
                let at = (token * kv_heads + head) * dim;
                cache.write(layer, false, head, span.past + token, &k[at..at + dim]);
                cache.write(layer, true, head, span.past + token, &v[at..at + dim]);
            }
        }
    }

    // 3. 줄마다: 제 글의 캐시에서 앞의 키와 값을 읽어 섞는다. 일의 단위는
    //    11장과 같이 (새 토큰, 키·값 헤드)다.
    let owners: Vec<(usize, usize)> = spans
        .iter()
        .enumerate()
        .flat_map(|(index, span)| (0..span.count).map(move |token| (index, token)))
        .collect();
    let caches: Vec<&PagedCache> = caches.iter().map(|cache| &**cache).collect();
    let scale = 1.0 / (dim as f32).sqrt();
    let shared = config.heads_per_kv();
    let unit = |index: usize, output: &mut [f32]| {
        let (row, kv_head) = (index / kv_heads, index % kv_heads);
        let (owner, token) = owners[row];
        let positions = spans[owner].past + token + 1;
        let start = (row * heads + kv_head * shared) * dim;
        let queries = &queries[start..start + shared * dim];
        arena::with_scratch(shared * (positions + PAGE_TOKENS), |scratch| {
            caches[owner].scores(layer, kv_head, queries, shared, positions, scratch);
            let attention = &mut scratch[..shared * positions];
            for scores in attention.chunks_exact_mut(positions) {
                for score in scores.iter_mut() {
                    *score *= scale;
                }
                softmax_in_place(scores);
            }
            caches[owner].mix(layer, kv_head, attention, shared, positions, output);
        });
    };
    let threads = badak_core::threads::global();
    let work: usize = spans
        .iter()
        .map(|span| simd::work(&[span.past + span.count, span.count, heads, dim]))
        .fold(0, usize::saturating_add);
    let count = if work < simd::MIN_PARALLEL_WORK {
        1
    } else {
        threads.for_vectors(rows)
    };
    let mixed = Tensor::build(&[rows, heads * dim], |output| {
        threads
            .pool()
            .run_chunks(output, shared * dim, count, &unit);
        Ok(())
    })?;

    // 4. 결과를 모으는 층도 묶음 전체에 한 번.
    Ok(ops::linear(&mixed, &weights.o_proj, None)?)
}
// ANCHOR_END: attention_batch

/// 여러 글의 새 토큰들을 한 묶음으로 받는 블록.
pub(super) fn block_batch(
    x: &Tensor,
    weights: &BlockWeights,
    config: &LlamaConfig,
    frequencies: &[f32],
    layer: usize,
    caches: &mut [&mut PagedCache],
    spans: &[Span],
) -> Result<Tensor> {
    let normed = ops::rms_norm(x, &weights.attention_norm, config.rms_norm_eps)?;
    let attended = attention_batch(&normed, weights, config, frequencies, layer, caches, spans)?;
    let h = ops::add(x, &attended)?;

    let normed = ops::rms_norm(&h, &weights.mlp_norm, config.rms_norm_eps)?;
    Ok(ops::add(&h, &mlp(&normed, weights)?)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> LlamaConfig {
        LlamaConfig {
            vocab_size: 8,
            hidden_size: 8,
            num_layers: 2,
            num_heads: 2,
            num_kv_heads: 1,
            intermediate_size: 16,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            rope_scaling: None,
            bos_token_id: None,
            eos_token_ids: Vec::new(),
        }
    }

    /// 토큰 `range`의 키와 값에 `mark + 토큰 번호`를 적는다.
    fn fill(cache: &mut PagedCache, count: usize, mark: f32) {
        cache.prepare(count).unwrap();
        for token in cache.tokens..cache.tokens + count {
            for layer in 0..2 {
                cache.write(layer, false, 0, token, &[mark + token as f32; 4]);
                cache.write(layer, true, 0, token, &[-(mark + token as f32); 4]);
            }
        }
        cache.tokens += count;
    }

    fn keys(cache: &PagedCache, layer: usize) -> Vec<f32> {
        cache
            .pieces(layer, false, 0, cache.tokens)
            .flat_map(|(_, keys)| keys.iter().step_by(4).copied())
            .collect()
    }

    #[test]
    fn forks_share_pages_until_one_of_them_writes() {
        let pool = PagePool::new(&config(), 8);
        let mut first = PagedCache::new(&pool);
        fill(&mut first, 20, 0.0);
        assert_eq!((first.pages(), pool.used()), (2, 2));

        // 앞의 18토큰을 함께 쓴다. 페이지는 늘지 않는다.
        let mut second = first.fork(18);
        assert_eq!((second.len(), second.pages(), pool.used()), (18, 2, 2));
        assert_eq!((first.shared_pages(), second.pages_needed(1)), (2, 1));

        // 둘째가 쓰면 반쯤 찬 둘째 페이지만 베낀다. 첫 페이지는 그대로 함께 쓴다.
        fill(&mut second, 3, 100.0);
        assert_eq!(pool.used(), 3);
        assert_eq!((first.shared_pages(), second.shared_pages()), (1, 1));
        let expected: Vec<f32> = (0..20).map(|token| token as f32).collect();
        assert_eq!(keys(&first, 1), expected);
        let mut expected: Vec<f32> = (0..18).map(|token| token as f32).collect();
        expected.extend([118.0, 119.0, 120.0]);
        assert_eq!(keys(&second, 1), expected);

        // 놓으면 풀로 돌아가고, 다음에 다시 쓰인다.
        drop(second);
        assert_eq!((pool.used(), pool.peak()), (2, 3));
        first.truncate(16);
        assert_eq!((first.pages(), pool.used(), pool.available()), (1, 1, 7));
    }

    #[test]
    fn the_pool_refuses_to_grow_past_its_limit() {
        let pool = PagePool::new(&config(), 2);
        let mut cache = PagedCache::new(&pool);
        fill(&mut cache, 32, 0.0);
        assert_eq!(cache.pages_needed(1), 1);
        assert!(cache.prepare(1).is_err());
        // 거절당한 뒤에도 세던 수가 틀어지지 않는다.
        assert_eq!((pool.used(), pool.available()), (2, 0));
        cache.truncate(16);
        assert!(cache.prepare(1).is_ok());
    }
}
