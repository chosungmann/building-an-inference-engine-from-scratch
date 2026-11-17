//! 벡터 명령으로 짠 커널 (17장).
//!
//! 16장의 `kernel`은 컴파일러가 벡터 명령을 쓰도록 유도했다. 이 모듈은 벡터
//! 명령을 직접 적는다. 프로세서의 종류마다 구현이 따로 있고, 실행할 때 쓸 수
//! 있는 것을 고른다.
//!
//! 모든 구현은 [`dots_portable`]과 **같은 비트**를 내야 한다. 그 함수가 이
//! 모듈의 계산을 정의한다.

// ANCHOR: contract
/// 내적 하나의 합을 나누어 담는 변수의 수.
pub const LANES: usize = 8;

/// 나누어 담은 합 여덟 개를 하나로 모은다. 순서가 정해져 있다.
///
/// 먼저 `i`번째와 `i + 4`번째를 더해 넷으로 줄이고, 그 넷을
/// `(0 + 2) + (1 + 3)`의 순서로 더한다.
#[inline(always)]
pub fn reduce(sums: [f32; LANES]) -> f32 {
    let four: [f32; 4] = std::array::from_fn(|lane| sums[lane] + sums[lane + 4]);
    (four[0] + four[2]) + (four[1] + four[3])
}

/// 내적 `M × N`개를 계산한다. 이 모듈의 계산을 정의하는 기준 구현이다.
///
/// 내적 하나는 이렇게 계산한다.
///
/// 1. 합 여덟 개를 0으로 둔다. 값을 여덟 개씩 읽어, `k`번째 값의 곱을
///    `k`번째 합에 더한다. 곱하고 더하기는 **한 번의 연산**으로 한다
///    (`mul_add`). 곱셈의 결과를 따로 반올림하지 않는다는 뜻이다.
/// 2. 합 여덟 개를 [`reduce`]로 모은다.
/// 3. 여덟으로 나누어떨어지지 않는 끝부분은 그 합에 차례로, 역시 한 번의
///    연산으로 곱하고 더한다.
pub fn dots_portable<const M: usize, const N: usize>(
    xs: [&[f32]; M],
    ws: [&[f32]; N],
) -> [[f32; N]; M] {
    let len = ws[0].len();
    assert!(
        xs.iter().chain(&ws).all(|vector| vector.len() == len),
        "벡터의 길이가 다릅니다"
    );
    let full = len - len % LANES;
    std::array::from_fn(|r| {
        std::array::from_fn(|c| {
            let (x, w) = (xs[r], ws[c]);
            let mut sums = [0.0_f32; LANES];
            let (x_chunks, w_chunks) = (x.as_chunks::<LANES>().0, w.as_chunks::<LANES>().0);
            for (x, w) in x_chunks.iter().zip(w_chunks) {
                for lane in 0..LANES {
                    sums[lane] = x[lane].mul_add(w[lane], sums[lane]);
                }
            }
            let mut sum = reduce(sums);
            for (x, w) in x[full..].iter().zip(&w[full..]) {
                sum = x.mul_add(*w, sum);
            }
            sum
        })
    })
}
// ANCHOR_END: contract

// ANCHOR: backend
/// 커널의 구현들.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// ARM의 NEON. 64비트 ARM 프로세서에는 늘 있다.
    Neon,
    /// x86의 AVX2와 FMA. 있는지는 실행할 때 프로세서에 물어본다.
    Avx2,
    /// 벡터 명령을 직접 쓰지 않는 기준 구현.
    Portable,
}

impl Backend {
    /// 이 기계에서 쓸 수 있는 가장 빠른 구현.
    pub fn detect() -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            Self::Neon
        }
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                Self::Avx2
            } else {
                Self::Portable
            }
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            Self::Portable
        }
    }
}
// ANCHOR_END: backend

// ANCHOR: tuning
/// 가중치 행렬을 걷는 방법. 입력 벡터의 수에 따라 가장 빠른 것이 다르다.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Walk {
    /// 행렬을 네 부분으로 나누어, 부분마다 한 행씩 네 행을 함께 읽는다.
    /// 메모리에서 길게 이어지는 줄기 넷을 동시에 읽게 된다. 벡터가 하나일 때.
    FourStreams,
    /// 이웃한 두 행을 함께 읽는다. 벡터가 몇 개일 때.
    TwoRows,
    /// 이웃한 네 행을 함께 읽는다. 벡터가 많을 때.
    FourRows,
}

impl Walk {
    /// 입력 벡터의 수에 맞는 방법. 경계는 측정으로 정했다(17.5절).
    pub fn for_vectors(count: usize) -> Self {
        match count {
            0 | 1 => Self::FourStreams,
            2..8 => Self::TwoRows,
            _ => Self::FourRows,
        }
    }
}
// ANCHOR_END: tuning

// ANCHOR: drive
/// 선형 층의 몸통. `$dots`가 내적 `M × N`개를 계산하는 함수다.
///
/// 함수가 아니라 매크로인 데는 이유가 있다. 이 반복문은 구현마다 한 벌씩
/// 컴파일되어야 하고, AVX2용은 "AVX2를 쓸 수 있다"고 표시된 함수 안에
/// 있어야 커널이 그 안으로 합쳐진다(17.6절).
macro_rules! drive {
    ($dots:path, $x:expr, $count:expr, $w:expr, $rows:expr, $cols:expr, $y:expr, $walk:expr) => {{
        let (x, count, w, rows, cols, y): (&[f32], usize, &[f32], usize, usize, &mut [f32]) =
            ($x, $count, $w, $rows, $cols, $y);
        let vector = |index: usize| &x[index * cols..(index + 1) * cols];
        let weight = |row: usize| &w[row * cols..(row + 1) * cols];
        // 입력 벡터는 둘씩 짝짓는다. 홀수이면 마지막 하나가 남는다.
        let paired = count - count % 2;

        let mut row = 0;
        if $walk == Walk::FourStreams {
            // 행렬을 네 부분으로 나눈다. 부분마다 한 행씩, 네 행 × 입력 벡터 하나.
            let part = rows / 4;
            for step in 0..part {
                let weights: [&[f32]; 4] =
                    std::array::from_fn(|stream| weight(stream * part + step));
                for index in 0..count {
                    let [results] = $dots([vector(index)], weights);
                    for (stream, value) in results.into_iter().enumerate() {
                        y[index * rows + stream * part + step] = value;
                    }
                }
            }
            row = 4 * part;
        }
        if $walk == Walk::FourRows {
            // 이웃한 가중치 네 행 × 입력 벡터 둘.
            while row + 4 <= rows {
                let weights: [&[f32]; 4] = std::array::from_fn(|offset| weight(row + offset));
                for index in (0..paired).step_by(2) {
                    let [first, second] = $dots([vector(index), vector(index + 1)], weights);
                    y[index * rows + row..][..4].copy_from_slice(&first);
                    y[(index + 1) * rows + row..][..4].copy_from_slice(&second);
                }
                if paired < count {
                    let [last] = $dots([vector(paired)], weights);
                    y[paired * rows + row..][..4].copy_from_slice(&last);
                }
                row += 4;
            }
        }
        // 이웃한 가중치 두 행 × 입력 벡터 둘. 위에서 다루지 않고 남은 행도 여기서.
        while row + 2 <= rows {
            let weights: [&[f32]; 2] = [weight(row), weight(row + 1)];
            for index in (0..paired).step_by(2) {
                let [first, second] = $dots([vector(index), vector(index + 1)], weights);
                y[index * rows + row..][..2].copy_from_slice(&first);
                y[(index + 1) * rows + row..][..2].copy_from_slice(&second);
            }
            if paired < count {
                let [last] = $dots([vector(paired)], weights);
                y[paired * rows + row..][..2].copy_from_slice(&last);
            }
            row += 2;
        }
        // 행이 홀수 개이면 마지막 한 행.
        if row < rows {
            let weights: [&[f32]; 1] = [weight(row)];
            for index in 0..count {
                let [[value]] = $dots([vector(index)], weights);
                y[index * rows + row] = value;
            }
        }
    }};
}
// ANCHOR_END: drive

// 위의 매크로를 쓰는 모듈들. 매크로를 정의한 뒤에 두어야 한다.
#[cfg(target_arch = "x86_64")]
pub mod avx2;
#[cfg(target_arch = "aarch64")]
pub mod neon;

fn check(x: &[f32], count: usize, w: &[f32], rows: usize, cols: usize, y: &[f32]) {
    assert_eq!(x.len(), count * cols, "입력의 크기가 맞지 않습니다");
    assert_eq!(w.len(), rows * cols, "가중치의 크기가 맞지 않습니다");
    assert_eq!(y.len(), count * rows, "출력의 크기가 맞지 않습니다");
}

// ANCHOR: linear
/// 선형 층의 몸통: 입력 벡터 `count`개 각각에 가중치 행렬을 곱한다.
///
/// `x`는 `(count, cols)`, `w`는 `(rows, cols)`, 결과 `y`는 `(count, rows)`다.
/// 이 기계에서 쓸 수 있는 가장 빠른 구현을 골라 쓰고, 일이 충분히 크면 여러
/// 스레드에 나눈다. 어느 구현을 쓰든, 벡터를 몇 개씩 넣든, 스레드가 몇
/// 개이든 `y`의 값은 비트 단위로 같다.
pub fn linear(x: &[f32], count: usize, w: &[f32], rows: usize, cols: usize, y: &mut [f32]) {
    let threads = crate::threads::global();
    // 작은 일은 나누지 않는다.
    let wanted = if count * rows * cols < MIN_PARALLEL_WORK {
        1
    } else {
        threads.for_vectors(count)
    };
    linear_threaded(
        threads.pool(),
        wanted,
        PIECES_PER_THREAD,
        x,
        count,
        w,
        rows,
        cols,
        y,
    );
}
// ANCHOR_END: linear

// ANCHOR: threaded
/// 곱셈이 이보다 적은 일은 나누지 않는다. 나누는 수고가 더 든다(18.5절).
pub const MIN_PARALLEL_WORK: usize = 1 << 18;
/// 스레드 하나에 돌아갈 조각의 수. 조각이 스레드보다 많아야 빠른 스레드가
/// 느린 스레드의 몫을 덜어 줄 수 있다(18.6절).
pub const PIECES_PER_THREAD: usize = 4;

/// 여러 스레드에 일을 나누는 방법.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Split {
    /// 가중치의 행을 나눈다. 스레드마다 가중치의 한 구간만 읽는다.
    Rows,
    /// 입력 벡터를 나눈다. 스레드마다 가중치 전체를 읽는다. 견주어 보려고
    /// 남겨 둔 방법이다(18.4절).
    Vectors,
}

/// [`linear`]와 같되, 풀과 스레드의 수와 조각의 수를 직접 정한다.
#[allow(clippy::too_many_arguments)]
pub fn linear_threaded(
    pool: &crate::threads::ThreadPool,
    threads: usize,
    pieces_per_thread: usize,
    x: &[f32],
    count: usize,
    w: &[f32],
    rows: usize,
    cols: usize,
    y: &mut [f32],
) {
    let split = Split::Rows;
    linear_split(
        pool,
        threads,
        pieces_per_thread,
        split,
        x,
        count,
        w,
        rows,
        cols,
        y,
    );
}

// ANCHOR_END: threaded

// ANCHOR: split
/// [`linear_threaded`]와 같되, 나누는 방법도 직접 정한다.
#[allow(clippy::too_many_arguments)]
pub fn linear_split(
    pool: &crate::threads::ThreadPool,
    threads: usize,
    pieces_per_thread: usize,
    split: Split,
    x: &[f32],
    count: usize,
    w: &[f32],
    rows: usize,
    cols: usize,
    y: &mut [f32],
) {
    check(x, count, w, rows, cols, y);
    let backend = Backend::detect();
    let threads = threads.clamp(1, pool.threads());
    if threads == 1 {
        linear_with(
            backend,
            Walk::for_vectors(count),
            x,
            count,
            w,
            rows,
            cols,
            y,
        );
        return;
    }
    let pieces = threads * pieces_per_thread.max(1);
    let walk = Walk::for_vectors(count);

    if split == Split::Vectors && count > 1 {
        // 벡터를 나눈다. 조각 하나가 벡터 `piece`개와, 결과에서 그 벡터들의 줄을
        // 맡는다. 조각마다 가중치 전체를 한 번 걷는다.
        let piece = count.div_ceil(pieces).max(8).next_multiple_of(2);
        pool.run_chunks(y, piece * rows, threads, &|index, output| {
            let vectors = output.len() / rows;
            let inputs = &x[index * piece * cols..][..vectors * cols];
            let walk = Walk::for_vectors(vectors);
            linear_with(backend, walk, inputs, vectors, w, rows, cols, output);
        });
        return;
    }

    // 행을 나눈다. 조각 하나가 가중치의 행 `piece`개를 맡는다. 네 행씩, 네
    // 줄기로 걷기 좋게 넷의 배수로 맞춘다.
    let piece = rows.div_ceil(pieces).next_multiple_of(4);
    // 조각 `index`가 맡은 가중치.
    let weights = |index: usize, piece_rows: usize| &w[index * piece * cols..][..piece_rows * cols];

    if count == 1 {
        // 벡터가 하나이면 결과가 행의 순서대로 놓인다. 조각마다 결과의 한
        // 구간을 곧바로 채운다.
        pool.run_chunks(y, piece, threads, &|index, output| {
            let piece_rows = output.len();
            let weights = weights(index, piece_rows);
            linear_with(backend, walk, x, 1, weights, piece_rows, cols, output);
        });
        return;
    }

    // 벡터가 여럿이면 결과에서 한 조각의 자리가 벡터마다 흩어져 있다. 먼저
    // 조각마다 따로 모아 두고(1단계), 그다음 제자리로 옮긴다(2단계).
    let mut scratch = vec![0.0_f32; count * rows];
    // 1단계: 조각마다 `(count, 조각의 행 수)` 모양의 결과를 채운다.
    pool.run_chunks(&mut scratch, piece * count, threads, &|index, output| {
        let piece_rows = output.len() / count;
        let weights = weights(index, piece_rows);
        linear_with(backend, walk, x, count, weights, piece_rows, cols, output);
    });
    // 2단계: 벡터마다, 조각들의 결과를 행의 순서대로 이어 붙인다.
    let group = count.div_ceil(threads);
    pool.run_chunks(y, group * rows, threads, &|index, output| {
        for (offset, row) in output.chunks_mut(rows).enumerate() {
            let vector = index * group + offset;
            for (part, source) in row.chunks_mut(piece).zip(scratch.chunks(piece * count)) {
                let piece_rows = part.len();
                part.copy_from_slice(&source[vector * piece_rows..][..piece_rows]);
            }
        }
    });
}
// ANCHOR_END: split

// ANCHOR: linear_with
/// 스레드 하나로, 구현과 가중치를 걷는 방법을 직접 골라 계산한다. 시험과
/// 측정에 쓰고, [`linear_threaded`]가 조각마다 부른다.
///
/// 이 기계에서 쓸 수 없는 구현을 고르면 기준 구현으로 계산한다.
#[allow(clippy::too_many_arguments)]
pub fn linear_with(
    backend: Backend,
    walk: Walk,
    x: &[f32],
    count: usize,
    w: &[f32],
    rows: usize,
    cols: usize,
    y: &mut [f32],
) {
    check(x, count, w, rows, cols, y);
    if cols == 0 {
        y.fill(0.0);
        return;
    }
    match backend {
        #[cfg(target_arch = "aarch64")]
        Backend::Neon => {
            // SAFETY: NEON은 64비트 ARM의 모든 프로세서에 있다.
            unsafe { neon::linear(x, count, w, rows, cols, y, walk) }
        }
        #[cfg(target_arch = "x86_64")]
        Backend::Avx2 if Backend::detect() == Backend::Avx2 => {
            // SAFETY: 바로 위에서 이 프로세서가 AVX2와 FMA를 지원함을 확인했다.
            unsafe { avx2::linear(x, count, w, rows, cols, y, walk) }
        }
        _ => drive!(dots_portable, x, count, w, rows, cols, y, walk),
    }
}
// ANCHOR_END: linear_with

#[cfg(test)]
mod tests {
    use super::*;

    fn values(count: usize, seed: usize) -> Vec<f32> {
        (0..count)
            .map(|i| ((i * 37 + seed * 101) % 29) as f32 * 0.37 - 5.0)
            .collect()
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn run(
        backend: Backend,
        walk: Walk,
        x: &[f32],
        count: usize,
        w: &[f32],
        rows: usize,
        cols: usize,
    ) -> Vec<f32> {
        let mut y = vec![f32::NAN; count * rows];
        linear_with(backend, walk, x, count, w, rows, cols, &mut y);
        y
    }

    #[test]
    fn reduce_follows_the_documented_order() {
        let sums = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert_eq!(reduce(sums), 36.0);
        // 순서가 결과를 바꾸는 값들. (0 + 4) = 1e8 - 1e8 = 0 이므로 1이 살아남는다.
        let sums = [1e8, 1.0, 0.0, 0.0, -1e8, 0.0, 0.0, 0.0];
        assert_eq!(reduce(sums), 1.0);
    }

    #[test]
    fn portable_matches_the_definition() {
        for len in [0, 1, 7, 8, 9, 100] {
            let (a, b) = (values(len, 1), values(len, 2));
            let expected: f64 = a
                .iter()
                .zip(&b)
                .map(|(x, y)| f64::from(*x) * f64::from(*y))
                .sum();
            let [[got]] = dots_portable([a.as_slice()], [b.as_slice()]);
            assert!((f64::from(got) - expected).abs() < 1e-2, "길이 {len}");
        }
    }

    // ANCHOR: same_bits
    /// 이 기계의 빠른 구현이 기준 구현과 같은 비트를 내는지 본다. 벡터의 수,
    /// 행의 수, 길이를 바꿔 가며 짝이 맞지 않는 경우와 끝부분이 남는 경우를
    /// 모두 지난다.
    #[test]
    fn every_backend_gives_the_same_bits_as_the_portable_one() {
        let fast = Backend::detect();
        for (count, rows, cols) in [
            (1, 1, 1),
            (1, 7, 8),
            (2, 4, 16),
            (3, 5, 19),
            (8, 9, 64),
            (9, 6, 131),
            (16, 13, 203),
        ] {
            let x = values(count * cols, 3);
            let w = values(rows * cols, 4);
            let reference = run(Backend::Portable, Walk::TwoRows, &x, count, &w, rows, cols);
            for walk in [Walk::FourStreams, Walk::TwoRows, Walk::FourRows] {
                for backend in [fast, Backend::Portable] {
                    let got = run(backend, walk, &x, count, &w, rows, cols);
                    assert_eq!(
                        bits(&got),
                        bits(&reference),
                        "{backend:?} {walk:?} ({count}, {rows}, {cols})"
                    );
                }
            }
        }
    }
    // ANCHOR_END: same_bits

    // ANCHOR: thread_bits
    /// 스레드와 조각의 수를 바꿔도 같은 비트가 나오는지 본다.
    #[test]
    fn any_number_of_threads_gives_the_same_bits() {
        use crate::threads::{ThreadPool, Waiting};
        let pool = ThreadPool::new(5, Waiting::Sleep);
        // 벡터가 하나일 때와 여럿일 때를 모두 본다.
        for (count, rows, cols) in [(1, 1031, 517), (37, 61, 233)] {
            let x = values(count * cols, 7);
            let w = values(rows * cols, 8);
            let reference = run(Backend::Portable, Walk::TwoRows, &x, count, &w, rows, cols);
            for threads in [1, 2, 3, 5] {
                for pieces_per_thread in [1, 4] {
                    for split in [Split::Rows, Split::Vectors] {
                        let mut y = vec![f32::NAN; count * rows];
                        linear_split(
                            &pool,
                            threads,
                            pieces_per_thread,
                            split,
                            &x,
                            count,
                            &w,
                            rows,
                            cols,
                            &mut y,
                        );
                        assert_eq!(
                            bits(&y),
                            bits(&reference),
                            "{split:?}, 스레드 {threads}개 ({count}, {rows}, {cols})"
                        );
                    }
                    let mut y = vec![f32::NAN; count * rows];
                    linear_threaded(
                        &pool,
                        threads,
                        pieces_per_thread,
                        &x,
                        count,
                        &w,
                        rows,
                        cols,
                        &mut y,
                    );
                    assert_eq!(
                        bits(&y),
                        bits(&reference),
                        "스레드 {threads}개 ({count}, {rows}, {cols})"
                    );
                }
            }
        }
    }
    // ANCHOR_END: thread_bits

    #[test]
    fn splitting_the_input_does_not_change_the_bits() {
        let (count, rows, cols) = (11, 10, 77);
        let x = values(count * cols, 5);
        let w = values(rows * cols, 6);
        let mut together = vec![0.0; count * rows];
        linear(&x, count, &w, rows, cols, &mut together);
        let mut one_by_one = vec![0.0; count * rows];
        for t in 0..count {
            linear(
                &x[t * cols..(t + 1) * cols],
                1,
                &w,
                rows,
                cols,
                &mut one_by_one[t * rows..(t + 1) * rows],
            );
        }
        assert_eq!(bits(&together), bits(&one_by_one));
    }

    // ANCHOR: fma
    #[test]
    fn fused_multiply_add_differs_from_separate_operations() {
        // 곱셈의 결과가 f32에 정확히 담기지 않는 값. 따로 곱하고 더하면 곱이
        // 반올림된 뒤에 더해진다.
        let a = [1.0 + f32::EPSILON];
        let b = [1.0 - f32::EPSILON];
        let [[fused]] = dots_portable([&a[..]], [&b[..]]);
        assert_eq!(fused, a[0].mul_add(b[0], 0.0));
        let c = [-1.0_f32];
        // (1 + e)(1 - e) - 1 = -e². 한 번의 연산으로 하면 이 작은 값이 남는다.
        let exact = a[0].mul_add(b[0], c[0]);
        let separate = a[0] * b[0] + c[0];
        assert_ne!(exact, separate);
        assert_eq!(separate, 0.0);
    }
    // ANCHOR_END: fma

    #[test]
    fn empty_shapes_are_handled() {
        let mut y = vec![1.0; 6];
        linear(&[], 2, &[], 3, 0, &mut y);
        assert_eq!(y, vec![0.0; 6]);
        linear(&[], 0, &[1.0, 2.0], 1, 2, &mut []);
    }
}
