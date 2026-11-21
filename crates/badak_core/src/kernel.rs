//! 행렬 곱의 커널 (16장).
//!
//! 2장의 `linalg`는 정의를 그대로 옮긴 구현이다. 이 모듈은 같은 계산을
//! 프로세서가 잘 처리할 수 있는 모양으로 다시 짠 것이다.

// ANCHOR: reduce
/// 나누어 담은 합 `L`개를 하나로 모은다. 이웃한 둘씩 더하기를 되풀이한다.
///
/// 더하는 순서가 늘 같으므로 같은 입력에는 늘 같은 비트가 나온다.
pub fn reduce<const L: usize>(mut sums: [f32; L]) -> f32 {
    let mut width = L;
    while width > 1 {
        width /= 2;
        for lane in 0..width {
            sums[lane] += sums[lane + width];
        }
    }
    sums[0]
}
// ANCHOR_END: reduce

// ANCHOR: dots
/// 내적 `M × N`개를 한꺼번에 계산한다. `xs`의 벡터 `M`개와 `ws`의 벡터
/// `N`개의 모든 짝이다. 돌려주는 값의 `[r][c]`가 `xs[r]`과 `ws[c]`의 내적이다.
///
/// 내적 하나의 합을 `L`개의 변수에 나누어 담는다. `L`은 2의 거듭제곱이어야
/// 한다. 내적 하나가 계산되는 순서는 `M`과 `N`에 상관없이 같다. 그래서 어떤
/// 배치로 계산해도 같은 비트가 나온다.
pub fn dots<const M: usize, const N: usize, const L: usize>(
    xs: [&[f32]; M],
    ws: [&[f32]; N],
) -> [[f32; N]; M] {
    let len = ws[0].len();
    assert!(
        xs.iter().chain(&ws).all(|vector| vector.len() == len),
        "벡터의 길이가 다릅니다"
    );
    // 벡터마다 `L`개씩 끊어 읽는다.
    let x_chunks: [&[[f32; L]]; M] = xs.map(|x| x.as_chunks::<L>().0);
    let w_chunks: [&[[f32; L]]; N] = ws.map(|w| w.as_chunks::<L>().0);

    let mut sums = [[[0.0_f32; L]; N]; M];
    for index in 0..len / L {
        let x: [[f32; L]; M] = std::array::from_fn(|r| x_chunks[r][index]);
        let w: [[f32; L]; N] = std::array::from_fn(|c| w_chunks[c][index]);
        for r in 0..M {
            for c in 0..N {
                for lane in 0..L {
                    sums[r][c][lane] += x[r][lane] * w[c][lane];
                }
            }
        }
    }

    // `L`로 나누어떨어지지 않는 끝부분은 차례로 더한다.
    let tail = len - len % L;
    std::array::from_fn(|r| {
        std::array::from_fn(|c| {
            let rest: f32 = xs[r][tail..]
                .iter()
                .zip(&ws[c][tail..])
                .map(|(x, w)| x * w)
                .sum();
            reduce(sums[r][c]) + rest
        })
    })
}
// ANCHOR_END: dots

// ANCHOR: shape
/// 내적 하나의 합을 나누어 담는 변수의 수.
pub const LANES: usize = 16;
/// 한꺼번에 계산하는 가중치 행의 수.
pub const ROWS: usize = 4;
/// 입력 벡터가 이보다 적으면 가중치를 한 행씩 읽는다. 네 행을 함께 읽는
/// 것은 그 행들을 여러 벡터에 다시 쓸 때만 이득이다(16.5절).
pub const MIN_VECTORS_FOR_ROWS: usize = 5;
// ANCHOR_END: shape

/// 두 벡터의 내적. 2장의 `linalg::dot`과 값은 같고 더하는 순서가 다르다.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    dots::<1, 1, LANES>([a], [b])[0][0]
}

// ANCHOR: linear
/// 선형 층의 몸통: 입력 벡터 `count`개 각각에 가중치 행렬을 곱한다.
///
/// `x`는 벡터들을 이어 놓은 `(count, cols)`, `w`는 `(rows, cols)`이고 결과
/// `y`는 `(count, rows)`다. `y[t][o]`는 `t`번째 벡터와 가중치의 `o`번째 행의
/// 내적이다. 편향은 더하지 않는다.
///
/// 벡터가 몇 개이든, 한꺼번에 주든 나누어 주든 `y`의 값은 비트 단위로 같다.
pub fn linear(x: &[f32], count: usize, w: &[f32], rows: usize, cols: usize, y: &mut [f32]) {
    // 입력을 끊지 않고 한 덩어리로 다룬다. 끊어도 빨라지지 않았다(16.6절).
    linear_blocked(x, count, w, rows, cols, y, count.max(1));
}

/// [`linear`]와 같되, 입력 벡터를 `block`개씩 끊어서 다룬다. 한 덩어리를
/// 다루는 동안 가중치 전체를 한 번 훑는다.
pub fn linear_blocked(
    x: &[f32],
    count: usize,
    w: &[f32],
    rows: usize,
    cols: usize,
    y: &mut [f32],
    block: usize,
) {
    assert_eq!(x.len(), count * cols, "입력의 크기가 맞지 않습니다");
    assert_eq!(w.len(), rows * cols, "가중치의 크기가 맞지 않습니다");
    assert_eq!(y.len(), count * rows, "출력의 크기가 맞지 않습니다");
    if cols == 0 {
        y.fill(0.0);
        return;
    }
    let vector = |index: usize| &x[index * cols..(index + 1) * cols];
    let weight = |row: usize| &w[row * cols..(row + 1) * cols];

    for first in (0..count).step_by(block) {
        let last = (first + block).min(count);
        // 가중치를 네 행씩 읽는다. 이 네 행이 가장 빠른 캐시에 있는 동안
        // 덩어리 안의 모든 벡터와 내적한다. 벡터가 한둘뿐이면 다시 쓸 일이
        // 없으므로 건너뛰고, 아래에서 한 행씩 차례로 읽는다.
        let mut row = 0;
        while last - first >= MIN_VECTORS_FOR_ROWS && row + ROWS <= rows {
            let weights: [&[f32]; ROWS] = std::array::from_fn(|offset| weight(row + offset));
            for index in first..last {
                let [results] = dots::<1, ROWS, LANES>([vector(index)], weights);
                y[index * rows + row..index * rows + row + ROWS].copy_from_slice(&results);
            }
            row += ROWS;
        }
        // 남은 행을 한 행씩 읽는다.
        for row in row..rows {
            for index in first..last {
                y[index * rows + row] = dot(vector(index), weight(row));
            }
        }
    }
}
// ANCHOR_END: linear

#[cfg(test)]
mod tests {
    use super::*;

    fn vector(len: usize, seed: usize) -> Vec<f32> {
        (0..len)
            .map(|i| ((i * 37 + seed * 101) % 23) as f32 * 0.125 - 1.0)
            .collect()
    }

    #[test]
    fn reduce_adds_everything() {
        assert_eq!(reduce([1.0, 2.0, 3.0, 4.0]), 10.0);
        assert_eq!(reduce([5.0]), 5.0);
    }

    #[test]
    fn dots_match_the_definition() {
        for len in [0, 1, 15, 16, 17, 100] {
            let (a, b) = (vector(len, 1), vector(len, 2));
            let expected: f64 = a.iter().zip(&b).map(|(x, y)| f64::from(x * y)).sum();
            let [[got]] = dots::<1, 1, 16>([&a], [&b]);
            assert!((f64::from(got) - expected).abs() < 1e-3, "길이 {len}");
        }
    }

    // ANCHOR: same_bits
    #[test]
    fn any_grouping_gives_the_same_bits() {
        let xs: Vec<Vec<f32>> = (0..4).map(|seed| vector(203, seed)).collect();
        let ws: Vec<Vec<f32>> = (0..4).map(|seed| vector(203, seed + 10)).collect();
        let block = dots::<4, 4, 16>(
            [&xs[0], &xs[1], &xs[2], &xs[3]],
            [&ws[0], &ws[1], &ws[2], &ws[3]],
        );
        for r in 0..4 {
            for c in 0..4 {
                let [[single]] = dots::<1, 1, 16>([&xs[r]], [&ws[c]]);
                assert_eq!(block[r][c].to_bits(), single.to_bits(), "({r}, {c})");
            }
        }
    }
    // ANCHOR_END: same_bits

    #[test]
    fn linear_matches_the_definition_for_any_shape() {
        for (count, rows, cols) in [(1, 1, 1), (3, 5, 7), (2, 8, 16), (5, 9, 33), (1, 4, 100)] {
            let x = vector(count * cols, 3);
            let w = vector(rows * cols, 4);
            let mut y = vec![f32::NAN; count * rows];
            linear(&x, count, &w, rows, cols, &mut y);
            for t in 0..count {
                for o in 0..rows {
                    let expected: f64 = (0..cols)
                        .map(|i| f64::from(x[t * cols + i]) * f64::from(w[o * cols + i]))
                        .sum();
                    let got = f64::from(y[t * rows + o]);
                    assert!((got - expected).abs() < 1e-3, "({count}, {rows}, {cols})");
                }
            }
        }
    }

    // ANCHOR: split
    #[test]
    fn linear_gives_the_same_bits_however_the_input_is_split() {
        let (count, rows, cols) = (7, 11, 131);
        let x = vector(count * cols, 5);
        let w = vector(rows * cols, 6);
        let mut together = vec![0.0; count * rows];
        linear(&x, count, &w, rows, cols, &mut together);

        // 하나씩 따로 넣는다.
        let mut one_by_one = vec![0.0; count * rows];
        for t in 0..count {
            let (input, output) = (
                &x[t * cols..(t + 1) * cols],
                &mut one_by_one[t * rows..(t + 1) * rows],
            );
            linear(input, 1, &w, rows, cols, output);
        }
        // 덩어리의 크기를 바꾼다.
        let mut blocked = vec![0.0; count * rows];
        linear_blocked(&x, count, &w, rows, cols, &mut blocked, 3);

        let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&together), bits(&one_by_one));
        assert_eq!(bits(&together), bits(&blocked));
    }
    // ANCHOR_END: split

    #[test]
    fn empty_shapes_are_handled() {
        let mut y = vec![1.0; 6];
        linear(&[], 2, &[], 3, 0, &mut y);
        assert_eq!(y, vec![0.0; 6]);
        linear(&[], 0, &[1.0, 2.0], 1, 2, &mut []);
    }
}
