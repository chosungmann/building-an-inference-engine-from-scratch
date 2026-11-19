//! 벡터 검색 (34장).
//!
//! 임베딩들을 쌓아 두고, 질의의 임베딩과 가장 가까운 것들을 찾는다.
//! 모든 벡터와 견주는 정확한 검색이다. 벡터가 수백만 개를 넘으면 일부만
//! 견주는 근사 검색을 쓰는데, 이 책은 거기까지 가지 않는다.

use badak_core::simd;

// ANCHOR: index
/// 쌓아 둔 벡터들. 모두 길이가 1이라고 본다.
pub struct Index {
    width: usize,
    /// `(벡터의 수, 폭)`.
    vectors: Vec<f32>,
}

/// 찾은 것 하나.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    /// 몇 번째로 넣은 벡터인가.
    pub index: usize,
    /// 질의와의 내적. 둘 다 길이가 1이면 코사인 유사도다.
    pub score: f32,
}

impl Index {
    /// 폭이 `width`인 빈 색인.
    pub fn new(width: usize) -> Self {
        Self {
            width,
            vectors: Vec::new(),
        }
    }

    /// 벡터들을 더한다. `vectors`는 `(개수, 폭)`이다.
    ///
    /// # Panics
    ///
    /// 길이가 폭의 배수가 아니면 공황에 빠진다.
    pub fn extend(&mut self, vectors: &[f32]) {
        assert!(
            self.width > 0 && vectors.len().is_multiple_of(self.width),
            "벡터들의 길이가 폭의 배수가 아닙니다"
        );
        self.vectors.extend_from_slice(vectors);
    }

    /// 벡터의 수.
    pub fn len(&self) -> usize {
        self.vectors.len().checked_div(self.width).unwrap_or(0)
    }

    /// 비었는가.
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// 질의와 모든 벡터의 내적. 벡터들을 가중치로 하는 선형 층에 질의
    /// 하나를 넣는 것과 같은 계산이어서 5장의 커널을 그대로 쓴다.
    pub fn scores(&self, query: &[f32]) -> Vec<f32> {
        let mut scores = vec![0.0; self.len()];
        simd::linear(query, 1, &self.vectors, self.len(), self.width, &mut scores);
        scores
    }

    /// 질의와 가장 가까운 `count`개를 가까운 순서로 돌려준다.
    pub fn search(&self, query: &[f32], count: usize) -> Vec<Hit> {
        let scores = self.scores(query);
        let mut hits: Vec<Hit> = scores
            .into_iter()
            .enumerate()
            .map(|(index, score)| Hit { index, score })
            .collect();
        let count = count.min(hits.len());
        if count == 0 {
            return Vec::new();
        }
        // 전부 정렬하지 않는다. 앞의 `count`개만 골라낸 다음 그것만 정렬한다.
        let nearer = |a: &Hit, b: &Hit| b.score.total_cmp(&a.score).then(a.index.cmp(&b.index));
        if count < hits.len() {
            hits.select_nth_unstable_by(count - 1, nearer);
            hits.truncate(count);
        }
        hits.sort_by(nearer);
        hits
    }
}
// ANCHOR_END: index

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_the_nearest_vectors_in_order() {
        let mut index = Index::new(2);
        index.extend(&[1.0, 0.0, 0.0, 1.0, 0.6, 0.8, -1.0, 0.0]);
        assert_eq!(index.len(), 4);
        let hits = index.search(&[0.8, 0.6], 2);
        assert_eq!(hits.len(), 2);
        assert_eq!((hits[0].index, hits[1].index), (2, 0));
        assert!((hits[0].score - 0.96).abs() < 1e-6);
        // 가진 것보다 많이 달라고 하면 가진 만큼만.
        assert_eq!(index.search(&[1.0, 0.0], 10).len(), 4);
        assert!(index.search(&[1.0, 0.0], 0).is_empty());
        assert!(Index::new(2).search(&[1.0, 0.0], 3).is_empty());
    }

    #[test]
    fn ties_keep_the_earlier_vector_first() {
        let mut index = Index::new(1);
        index.extend(&[1.0, 1.0, 1.0]);
        let hits = index.search(&[1.0], 2);
        assert_eq!((hits[0].index, hits[1].index), (0, 1));
    }
}
