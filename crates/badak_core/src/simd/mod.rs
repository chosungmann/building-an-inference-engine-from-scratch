//! 벡터 명령으로 짠 커널 (17장).
//!
//! 16장의 `kernel`은 컴파일러가 벡터 명령을 쓰도록 유도했다. 이 모듈은 벡터
//! 명령을 직접 적는다. 프로세서의 종류마다 구현이 따로 있고, 실행할 때 쓸 수
//! 있는 것을 고른다.
//!
//! 모든 구현은 [`dots_portable`]과 **같은 비트**를 내야 한다. 그 함수가 이
//! 모듈의 계산을 정의한다.

// ANCHOR: weight
/// 선형 층의 가중치가 될 수 있는 원소. 계산할 때 `f32`로 바꿔 읽는다 (20장).
///
/// `f32`는 그대로이고, [`Bf16`](crate::quant::Bf16)은 아래 16비트에 0을
/// 채운다. 값이 정확히 보존되므로 어느 쪽으로 저장했든 계산 결과의 비트가
/// 같다.
pub trait Weight: Copy + Send + Sync + 'static {
    /// 값 하나를 `f32`로 바꾼다.
    fn widen(self) -> f32;

    /// 값 넷을 읽어 NEON 레지스터 하나에 `f32`로 담는다.
    ///
    /// # Safety
    ///
    /// `pointer`부터 값 넷을 읽을 수 있어야 한다.
    #[cfg(target_arch = "aarch64")]
    unsafe fn load_neon(pointer: *const Self) -> core::arch::aarch64::float32x4_t;

    /// 값 여덟을 읽어 AVX 레지스터 하나에 `f32`로 담는다.
    ///
    /// # Safety
    ///
    /// `pointer`부터 값 여덟을 읽을 수 있고, 프로세서가 AVX2를 지원해야 한다.
    #[cfg(target_arch = "x86_64")]
    unsafe fn load_avx2(pointer: *const Self) -> core::arch::x86_64::__m256;
}

impl Weight for f32 {
    #[inline(always)]
    fn widen(self) -> f32 {
        self
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn load_neon(pointer: *const Self) -> core::arch::aarch64::float32x4_t {
        // SAFETY: 부르는 쪽이 값 넷을 읽을 수 있음을 보장한다.
        unsafe { core::arch::aarch64::vld1q_f32(pointer) }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn load_avx2(pointer: *const Self) -> core::arch::x86_64::__m256 {
        // SAFETY: 부르는 쪽이 값 여덟을 읽을 수 있음을 보장한다.
        unsafe { core::arch::x86_64::_mm256_loadu_ps(pointer) }
    }
}

impl Weight for crate::quant::Bf16 {
    #[inline(always)]
    fn widen(self) -> f32 {
        self.to_f32()
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn load_neon(pointer: *const Self) -> core::arch::aarch64::float32x4_t {
        use core::arch::aarch64::{vld1_u16, vmovl_u16, vreinterpretq_f32_u32, vshlq_n_u32};
        // SAFETY: 부르는 쪽이 값 넷을 읽을 수 있음을 보장한다. `Bf16`은
        // `u16` 하나와 생김새가 같다.
        // 16비트 넷을 32비트 넷으로 넓히고, 왼쪽으로 16비트 밀어 위쪽 절반에
        // 놓는다. 그 비트를 그대로 `f32`로 읽는다.
        unsafe {
            let halves = vld1_u16(pointer.cast::<u16>());
            vreinterpretq_f32_u32(vshlq_n_u32::<16>(vmovl_u16(halves)))
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn load_avx2(pointer: *const Self) -> core::arch::x86_64::__m256 {
        use core::arch::x86_64::{
            __m128i, _mm_loadu_si128, _mm256_castsi256_ps, _mm256_cvtepu16_epi32, _mm256_slli_epi32,
        };
        // SAFETY: 부르는 쪽이 값 여덟(16바이트)을 읽을 수 있고 AVX2가 있음을
        // 보장한다.
        unsafe {
            let halves = _mm_loadu_si128(pointer.cast::<__m128i>());
            _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(halves)))
        }
    }
}
// ANCHOR_END: weight

// ANCHOR: weight_f16
/// 16비트 실수(IEEE 754 binary16)도 곱하는 순간에 `f32`로 넓혀 읽는다 (24장).
///
/// KV 캐시를 16비트로 담았을 때(21.7절) 키를 통째로 펼친 사본을 만들지 않고
/// 그대로 내적한다. 16비트 실수는 모두 `f32`로 정확히 옮겨지므로, 먼저 펼쳐
/// 놓고 계산한 것과 결과의 비트가 같다.
impl Weight for crate::quant::F16 {
    #[inline(always)]
    fn widen(self) -> f32 {
        self.to_f32()
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn load_neon(pointer: *const Self) -> core::arch::aarch64::float32x4_t {
        use core::arch::aarch64::vld1q_f32;
        #[cfg(target_feature = "fp16")]
        {
            use core::arch::aarch64::{vcvt_f32_f16, vld1_u16, vreinterpret_f16_u16};
            // SAFETY: 부르는 쪽이 값 넷을 읽을 수 있음을 보장한다. `F16`은
            // `u16` 하나와 생김새가 같다. 이 갈래는 16비트 실수 명령이 있는
            // 프로세서를 대상으로 컴파일할 때만 들어간다(Apple Silicon은 늘
            // 그렇다). 넷을 한 명령으로 넓힌다.
            unsafe {
                let _ = vld1q_f32;
                vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(pointer.cast::<u16>())))
            }
        }
        #[cfg(not(target_feature = "fp16"))]
        {
            // SAFETY: 부르는 쪽이 값 넷을 읽을 수 있음을 보장한다.
            unsafe {
                let values: [f32; 4] = std::array::from_fn(|lane| (*pointer.add(lane)).to_f32());
                vld1q_f32(values.as_ptr())
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn load_avx2(pointer: *const Self) -> core::arch::x86_64::__m256 {
        // 16비트 실수를 넓히는 명령(F16C)은 AVX2와 따로 확인해야 하는
        // 기능이다. 여기서는 하나씩 바꾼다. x86에서의 속도는 재지 않았다.
        // SAFETY: 부르는 쪽이 값 여덟을 읽을 수 있고 AVX2가 있음을 보장한다.
        unsafe {
            let values: [f32; 8] = std::array::from_fn(|lane| (*pointer.add(lane)).to_f32());
            core::arch::x86_64::_mm256_loadu_ps(values.as_ptr())
        }
    }
}
// ANCHOR_END: weight_f16

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
pub fn dots_portable<W: Weight, const M: usize, const N: usize>(
    xs: [&[f32]; M],
    ws: [&[W]; N],
) -> [[f32; N]; M] {
    let len = ws[0].len();
    assert!(
        xs.iter().all(|x| x.len() == len) && ws.iter().all(|w| w.len() == len),
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
                    sums[lane] = x[lane].mul_add(w[lane].widen(), sums[lane]);
                }
            }
            let mut sum = reduce(sums);
            for (x, w) in x[full..].iter().zip(&w[full..]) {
                sum = x.mul_add(w.widen(), sum);
            }
            sum
        })
    })
}
// ANCHOR_END: contract

// ANCHOR: widen
/// 값들을 `f32`로 넓혀 `wide`에 적는다 (24장).
///
/// 16비트로 담아 둔 값을 계산에 쓰려고 펼칠 때 쓴다. 이 기계의 벡터 명령으로
/// 여러 개씩 넓힌다. 어느 길로 넓히든 값은 정확히 같다.
pub fn widen<W: Weight>(values: &[W], wide: &mut [f32]) {
    assert_eq!(values.len(), wide.len(), "길이가 다릅니다");
    #[cfg(target_arch = "aarch64")]
    let done = {
        let full = values.len() - values.len() % 4;
        for start in (0..full).step_by(4) {
            // SAFETY: `start + 4 <= full <= len`이므로 두 슬라이스 모두에서 값
            // 넷을 읽고 쓸 수 있다. NEON은 64비트 ARM에 늘 있다.
            unsafe {
                let lanes = W::load_neon(values.as_ptr().add(start));
                core::arch::aarch64::vst1q_f32(wide.as_mut_ptr().add(start), lanes);
            }
        }
        full
    };
    #[cfg(not(target_arch = "aarch64"))]
    let done = 0;
    for (wide, value) in wide[done..].iter_mut().zip(&values[done..]) {
        *wide = value.widen();
    }
}
// ANCHOR_END: widen

// ANCHOR: dot
/// 내적 하나. [`dots_portable`]의 정의대로 계산한다 (24장).
///
/// 어텐션의 점수가 이 정의를 따른다. 빠른 커널로 계산한 점수와 비트가 같다.
pub fn dot(x: &[f32], w: &[f32]) -> f32 {
    let [[value]] = dots_portable([x], [w]);
    value
}
// ANCHOR_END: dot

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
        let (x, count, w, rows, cols, y): (&[f32], usize, &[_], usize, usize, &mut [f32]) =
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
                let weights: [&[_]; 4] = std::array::from_fn(|stream| weight(stream * part + step));
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
                let weights: [&[_]; 4] = std::array::from_fn(|offset| weight(row + offset));
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
            let weights: [&[_]; 2] = [weight(row), weight(row + 1)];
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
            let weights: [&[_]; 1] = [weight(row)];
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
pub mod half;
#[cfg(target_arch = "aarch64")]
pub mod neon;

fn check<W>(x: &[f32], count: usize, w: &[W], rows: usize, cols: usize, y: &[f32]) {
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
pub fn linear<W: Weight>(
    x: &[f32],
    count: usize,
    w: &[W],
    rows: usize,
    cols: usize,
    y: &mut [f32],
) {
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
pub fn linear_threaded<W: Weight>(
    pool: &crate::threads::ThreadPool,
    threads: usize,
    pieces_per_thread: usize,
    x: &[f32],
    count: usize,
    w: &[W],
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
pub fn linear_split<W: Weight>(
    pool: &crate::threads::ThreadPool,
    threads: usize,
    pieces_per_thread: usize,
    split: Split,
    x: &[f32],
    count: usize,
    w: &[W],
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

    // 행을 나눈다. 조각 하나가 가중치의 행 몇 개를 맡는다.
    split_rows(pool, threads, pieces, count, rows, y, &|range, output| {
        let weights = &w[range.start * cols..range.end * cols];
        linear_with(backend, walk, x, count, weights, range.len(), cols, output);
    });
}
// ANCHOR_END: split

// ANCHOR: split_rows
/// 가중치의 행을 `pieces`조각으로 나누어 스레드들에 맡긴다.
///
/// `kernel(행의 구간, 결과)`가 그 구간의 행들을 입력 벡터 `count`개와 곱해
/// `(count, 구간의 행 수)` 모양의 결과를 채운다. 가중치가 어떤 형식인지는 이
/// 함수가 알 필요가 없다. 20장의 양자화된 가중치도 이 함수로 나눈다.
pub fn split_rows(
    pool: &crate::threads::ThreadPool,
    threads: usize,
    pieces: usize,
    count: usize,
    rows: usize,
    y: &mut [f32],
    kernel: &(dyn Fn(std::ops::Range<usize>, &mut [f32]) + Sync),
) {
    // 네 행씩, 네 줄기로 걷기 좋게 조각의 크기를 넷의 배수로 맞춘다.
    let piece = rows.div_ceil(pieces.max(1)).next_multiple_of(4);
    let range = |index: usize, piece_rows: usize| index * piece..index * piece + piece_rows;

    if count == 1 {
        // 벡터가 하나이면 결과가 행의 순서대로 놓인다. 조각마다 결과의 한
        // 구간을 곧바로 채운다.
        pool.run_chunks(y, piece, threads, &|index, output| {
            kernel(range(index, output.len()), output);
        });
        return;
    }

    // 벡터가 여럿이면 결과에서 한 조각의 자리가 벡터마다 흩어져 있다. 먼저
    // 조각마다 따로 모아 두고(1단계), 그다음 제자리로 옮긴다(2단계).
    // 22장: 모아 둘 자리는 스레드가 돌려 쓰는 것을 빌린다.
    crate::arena::with_scratch(count * rows, |scratch| {
        // 1단계: 조각마다 `(count, 조각의 행 수)` 모양의 결과를 채운다.
        pool.run_chunks(scratch, piece * count, threads, &|index, output| {
            kernel(range(index, output.len() / count), output);
        });
        // 2단계: 벡터마다, 조각들의 결과를 행의 순서대로 이어 붙인다.
        let scratch = &*scratch;
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
    });
}
// ANCHOR_END: split_rows

// ANCHOR: linear_with
/// 스레드 하나로, 구현과 가중치를 걷는 방법을 직접 골라 계산한다. 시험과
/// 측정에 쓰고, [`linear_threaded`]가 조각마다 부른다.
///
/// 이 기계에서 쓸 수 없는 구현을 고르면 기준 구현으로 계산한다.
#[allow(clippy::too_many_arguments)]
pub fn linear_with<W: Weight>(
    backend: Backend,
    walk: Walk,
    x: &[f32],
    count: usize,
    w: &[W],
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

    // ANCHOR: bf16_bits
    /// BF16으로 저장한 가중치로 계산해도 `f32`로 펼친 것과 같은 비트가 나오는지
    /// 본다. 값이 같으니 결과도 같아야 한다.
    #[test]
    fn bf16_weights_give_the_same_bits_as_widened_ones() {
        use crate::quant::Bf16;
        for (count, rows, cols) in [(1, 9, 5), (1, 16, 64), (3, 7, 33), (9, 13, 100)] {
            let x = values(count * cols, 3);
            let halves: Vec<Bf16> = values(rows * cols, 4)
                .into_iter()
                .map(Bf16::from_f32)
                .collect();
            let widened: Vec<f32> = halves.iter().map(|half| half.to_f32()).collect();
            let reference = run(
                Backend::Portable,
                Walk::TwoRows,
                &x,
                count,
                &widened,
                rows,
                cols,
            );
            for backend in [Backend::Portable, Backend::detect()] {
                for walk in [Walk::FourStreams, Walk::TwoRows, Walk::FourRows] {
                    let mut y = vec![f32::NAN; count * rows];
                    linear_with(backend, walk, &x, count, &halves, rows, cols, &mut y);
                    assert_eq!(bits(&y), bits(&reference), "{backend:?} {walk:?}");
                }
            }
        }
    }
    // ANCHOR_END: bf16_bits

    #[test]
    fn widen_gives_the_same_values_as_one_by_one() {
        use crate::quant::{Bf16, F16};
        let source = values(37, 5);
        let halves: Vec<F16> = source.iter().map(|&v| F16::from_f32(v)).collect();
        let brains: Vec<Bf16> = source.iter().map(|&v| Bf16::from_f32(v)).collect();
        let mut wide = vec![f32::NAN; 37];
        widen(&halves, &mut wide);
        let expected: Vec<f32> = halves.iter().map(|half| half.to_f32()).collect();
        assert_eq!(bits(&wide), bits(&expected));
        widen(&brains, &mut wide);
        let expected: Vec<f32> = brains.iter().map(|half| half.to_f32()).collect();
        assert_eq!(bits(&wide), bits(&expected));
        widen(&source, &mut wide);
        assert_eq!(bits(&wide), bits(&source));
    }

    /// 16비트 실수로 저장한 값으로 계산해도 `f32`로 펼친 것과 같은 비트가
    /// 나오는지 본다 (24장).
    #[test]
    fn f16_weights_give_the_same_bits_as_widened_ones() {
        use crate::quant::F16;
        for (count, rows, cols) in [(1, 9, 5), (4, 16, 64), (3, 7, 33), (4, 301, 64)] {
            let x = values(count * cols, 3);
            let halves: Vec<F16> = values(rows * cols, 4)
                .into_iter()
                .map(F16::from_f32)
                .collect();
            let widened: Vec<f32> = halves.iter().map(|half| half.to_f32()).collect();
            let reference = run(
                Backend::Portable,
                Walk::TwoRows,
                &x,
                count,
                &widened,
                rows,
                cols,
            );
            for backend in [Backend::Portable, Backend::detect()] {
                for walk in [Walk::FourStreams, Walk::TwoRows, Walk::FourRows] {
                    let mut y = vec![f32::NAN; count * rows];
                    linear_with(backend, walk, &x, count, &halves, rows, cols, &mut y);
                    assert_eq!(bits(&y), bits(&reference), "{backend:?} {walk:?}");
                }
            }
        }
    }

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
        linear::<f32>(&[], 2, &[], 3, 0, &mut y);
        assert_eq!(y, vec![0.0; 6]);
        linear(&[], 0, &[1.0, 2.0], 1, 2, &mut []);
    }
}
