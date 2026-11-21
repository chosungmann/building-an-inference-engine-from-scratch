//! 0이 많은 행렬 (46장).
//!
//! 가지치기는 가중치의 일부를 0으로 만든다. 0을 **적어 두지 않는** 형식과
//! 0을 **곱하지 않는** 커널이 있어야 그것이 크기와 속도로 이어진다. 이
//! 모듈에 그 둘이 있다. 형식은 가장 흔한 CSR(행마다 0이 아닌 값과 그 열의
//! 번호를 차례로 적는 것)이다.

use crate::error::{Error, Result};
use crate::ops::LinearWeight;

// ANCHOR: prune
/// 절댓값이 작은 것부터 `fraction`만큼을 0으로 만든다. 0이 된 값의 수를
/// 돌려준다.
///
/// 가장 단순한 가지치기다. "작은 가중치는 덜 중요하다"고 보는 것이다.
pub fn prune_magnitude(values: &mut [f32], fraction: f64) -> usize {
    let cut = ((values.len() as f64 * fraction.clamp(0.0, 1.0)).round() as usize).min(values.len());
    if cut == 0 {
        return 0;
    }
    // `cut`번째로 작은 절댓값을 찾는다. 다 정렬할 필요는 없다.
    let mut magnitudes: Vec<f32> = values.iter().map(|value| value.abs()).collect();
    let (_, threshold, _) = magnitudes.select_nth_unstable_by(cut - 1, f32::total_cmp);
    let threshold = *threshold;
    // 임계값과 같은 값이 여럿이면 앞에서부터 필요한 만큼만 지운다.
    let mut left = cut;
    for value in values.iter_mut() {
        if left > 0 && value.abs() <= threshold {
            *value = 0.0;
            left -= 1;
        }
    }
    cut - left
}
// ANCHOR_END: prune

// ANCHOR: sparse
/// 0이 아닌 값만 적어 둔 행렬.
#[derive(Debug, Clone, PartialEq)]
pub struct Sparse {
    rows: usize,
    cols: usize,
    /// 행마다 그 행의 값이 시작하는 위치. 길이는 `rows + 1`이다.
    starts: Vec<u32>,
    /// 값마다 그 값이 놓인 열.
    columns: Vec<u16>,
    /// 0이 아닌 값들. 행의 순서대로, 행 안에서는 열의 순서대로다.
    values: Vec<f32>,
}

impl Sparse {
    /// `(rows, cols)`의 행렬에서 0이 아닌 값만 골라 담는다.
    pub fn from_dense(dense: &[f32], rows: usize, cols: usize) -> Result<Self> {
        // 열의 번호를 16비트에 적는다.
        if cols > usize::from(u16::MAX) + 1 {
            return Err(Error::Shape {
                operation: "희소 행렬 만들기",
                left: vec![rows, cols],
                right: vec![rows, usize::from(u16::MAX) + 1],
            });
        }
        if dense.len() != rows * cols {
            return Err(Error::ElementCount {
                expected: rows * cols,
                actual: dense.len(),
            });
        }
        let mut sparse = Self {
            rows,
            cols,
            starts: Vec::with_capacity(rows + 1),
            columns: Vec::new(),
            values: Vec::new(),
        };
        for row in dense.chunks_exact(cols.max(1)) {
            sparse.starts.push(sparse.values.len() as u32);
            for (column, &value) in row.iter().enumerate() {
                if value != 0.0 {
                    sparse.columns.push(column as u16);
                    sparse.values.push(value);
                }
            }
        }
        sparse.starts.resize(rows + 1, sparse.values.len() as u32);
        Ok(sparse)
    }

    /// 0이 아닌 값의 수.
    pub fn nonzeros(&self) -> usize {
        self.values.len()
    }

    /// 이 형식이 차지하는 바이트. 값마다 6바이트(값 4, 열 번호 2), 행마다 4바이트.
    pub fn bytes(&self) -> usize {
        self.values.len() * 4 + self.columns.len() * 2 + self.starts.len() * 4
    }

    /// 0을 다시 채운 보통의 행렬.
    pub fn to_dense(&self) -> Vec<f32> {
        let mut dense = vec![0.0; self.rows * self.cols];
        for row in 0..self.rows {
            let range = self.starts[row] as usize..self.starts[row + 1] as usize;
            for (&column, &value) in self.columns[range.clone()].iter().zip(&self.values[range]) {
                dense[row * self.cols + usize::from(column)] = value;
            }
        }
        dense
    }

    /// 한 행과 벡터 하나의 내적.
    ///
    /// 열의 번호를 따라 `x`의 여기저기를 읽는다. 16장의 커널처럼 "값 넷을 한
    /// 명령으로"가 되지 않는다. 읽을 자리가 이어져 있지 않기 때문이다. 합을
    /// 네 줄기로 나눠 더하는 것까지만 한다.
    #[inline]
    fn dot(columns: &[u16], values: &[f32], x: &[f32]) -> f32 {
        let mut sums = [0.0_f32; 4];
        let (column_fours, column_rest) = columns.as_chunks::<4>();
        let (value_fours, value_rest) = values.as_chunks::<4>();
        for (columns, values) in column_fours.iter().zip(value_fours) {
            for lane in 0..4 {
                sums[lane] += values[lane] * x[usize::from(columns[lane])];
            }
        }
        let mut tail = 0.0;
        for (&column, &value) in column_rest.iter().zip(value_rest) {
            tail += value * x[usize::from(column)];
        }
        (sums[0] + sums[1]) + (sums[2] + sums[3]) + tail
    }
}
// ANCHOR_END: sparse

/// 보통의 커널이 일을 나누기 시작하는 크기를 이것으로 나눈 크기부터 일을
/// 나눈다. 재서 정했다(46장).
const PARALLEL_DIVISOR: usize = 32;

// ANCHOR: sparse_apply
impl LinearWeight for Sparse {
    fn dimensions(&self) -> Option<(usize, usize)> {
        Some((self.rows, self.cols))
    }

    fn apply(&self, x: &[f32], count: usize, y: &mut [f32]) -> Result<()> {
        if x.len() != count * self.cols || y.len() != count * self.rows {
            return Err(Error::Shape {
                operation: "희소 행렬 곱하기",
                left: vec![x.len(), y.len()],
                right: vec![count * self.cols, count * self.rows],
            });
        }
        let threads = crate::threads::global();
        // 곱셈 하나가 보통의 커널보다 느리므로, 더 작은 일부터 나눈다.
        let work = crate::simd::work(&[count, self.values.len()]);
        let wanted = if work < crate::simd::MIN_PARALLEL_WORK / PARALLEL_DIVISOR {
            1
        } else {
            threads.for_vectors(count)
        };
        let row = |row: usize, x: &[f32]| {
            let range = self.starts[row] as usize..self.starts[row + 1] as usize;
            Self::dot(&self.columns[range.clone()], &self.values[range], x)
        };
        if count == 1 {
            // 벡터가 하나면 행을 나눈다.
            let size = self
                .rows
                .div_ceil(wanted * crate::simd::PIECES_PER_THREAD)
                .max(1);
            threads.pool().run_chunks(y, size, wanted, &|index, chunk| {
                for (offset, out) in chunk.iter_mut().enumerate() {
                    *out = row(index * size + offset, x);
                }
            });
        } else {
            // 벡터가 여럿이면 벡터를 나눈다. 벡터마다 행렬 전체를 한 번 읽는다.
            threads
                .pool()
                .run_chunks(y, self.rows, wanted, &|index, chunk| {
                    let x = &x[index * self.cols..][..self.cols];
                    for (position, out) in chunk.iter_mut().enumerate() {
                        *out = row(position, x);
                    }
                });
        }
        Ok(())
    }
}
// ANCHOR_END: sparse_apply

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd;

    fn values(count: usize, seed: usize) -> Vec<f32> {
        (0..count)
            .map(|index| ((index * 37 + seed * 101) % 29) as f32 * 0.37 - 5.0)
            .collect()
    }

    #[test]
    fn pruning_zeroes_the_smallest_magnitudes() {
        let mut weights = vec![0.5, -3.0, 0.1, 2.0, -0.2, 0.2, -4.0, 1.0];
        assert_eq!(prune_magnitude(&mut weights, 0.5), 4);
        // 절댓값이 0.1, 0.2, 0.2, 0.5인 넷이 지워진다.
        assert_eq!(weights, [0.0, -3.0, 0.0, 2.0, 0.0, 0.0, -4.0, 1.0]);
        assert_eq!(prune_magnitude(&mut weights, 0.0), 0);
        assert_eq!(prune_magnitude(&mut weights, 1.0), 8);
        assert!(weights.iter().all(|&value| value == 0.0));
        // 같은 값이 많아도 딱 그만큼만 지운다.
        let mut same = vec![1.0_f32; 10];
        assert_eq!(prune_magnitude(&mut same, 0.3), 3);
        assert_eq!(same.iter().filter(|&&value| value == 0.0).count(), 3);
    }

    #[test]
    fn round_trips_and_counts() {
        let (rows, cols) = (7, 13);
        let mut dense = values(rows * cols, 1);
        prune_magnitude(&mut dense, 0.6);
        let sparse = Sparse::from_dense(&dense, rows, cols).unwrap();
        assert_eq!(
            sparse.nonzeros(),
            dense.iter().filter(|&&value| value != 0.0).count()
        );
        assert_eq!(sparse.to_dense(), dense);
        assert_eq!(sparse.bytes(), sparse.nonzeros() * 6 + (rows + 1) * 4);
        assert!(Sparse::from_dense(&dense, rows, cols + 1).is_err());
        assert!(Sparse::from_dense(&[], 0, 70_000).is_err());
        // 빈 행이 있어도 된다.
        let empty = Sparse::from_dense(&[0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 3, 2).unwrap();
        assert_eq!(
            (empty.nonzeros(), empty.to_dense()),
            (1, vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        );
    }

    #[test]
    fn multiplies_like_the_dense_kernel() {
        // 스레드로 나뉘는 크기와 나뉘지 않는 크기, 벡터 하나와 여럿.
        for (rows, cols, count) in [(5, 9, 1), (5, 9, 3), (600, 520, 1), (600, 520, 5)] {
            let mut weight = values(rows * cols, 2);
            prune_magnitude(&mut weight, 0.7);
            let x = values(count * cols, 3);
            let mut expected = vec![0.0; count * rows];
            simd::linear(&x, count, &weight, rows, cols, &mut expected);
            let sparse = Sparse::from_dense(&weight, rows, cols).unwrap();
            let mut actual = vec![0.0; count * rows];
            sparse.apply(&x, count, &mut actual).unwrap();
            // 더하는 순서가 달라 비트까지 같지는 않다.
            for (a, e) in actual.iter().zip(&expected) {
                assert!(
                    (a - e).abs() <= 1e-3 * e.abs().max(1.0),
                    "{rows}×{cols}×{count}: {a} {e}"
                );
            }
            assert!(sparse.apply(&x, count + 1, &mut actual).is_err());
        }
    }
}
