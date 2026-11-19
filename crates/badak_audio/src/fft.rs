//! 푸리에 변환.
//!
//! 소리의 한 토막(표본 `n`개)을 "주파수마다의 세기" `n`개로 바꾼다. 정의대로
//! 계산하면 곱셈이 `n²`번이고, 빠른 푸리에 변환(FFT)은 `n log n`번이다.

use std::f64::consts::PI;

// ANCHOR: complex
/// 복소수. 실수부와 허수부.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Complex {
    /// 실수부.
    pub re: f64,
    /// 허수부.
    pub im: f64,
}

impl Complex {
    /// 길이가 1이고 각도가 `angle`(라디안)인 복소수. `cos + i·sin`이다.
    pub fn unit(angle: f64) -> Self {
        let (im, re) = angle.sin_cos();
        Self { re, im }
    }

    /// 곱. `(a + bi)(c + di) = (ac − bd) + (ad + bc)i`.
    pub fn times(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    /// 합.
    pub fn plus(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }

    /// 차.
    pub fn minus(self, other: Self) -> Self {
        Self {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }

    /// 켤레. 허수부의 부호를 바꾼 것이다.
    pub fn conjugate(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// 크기의 제곱. 그 주파수의 세기(power)다.
    pub fn power(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
}
// ANCHOR_END: complex

// ANCHOR: dft
/// 정의 그대로의 푸리에 변환.
///
/// `X[k] = Σₜ x[t] · e^(−2πi·k·t/n)`. `k`번째 출력은 입력과 "한 토막 안에서
/// `k`번 도는 회전"의 내적이다. 입력에 그 빠르기의 떨림이 들어 있으면 값이
/// 커진다. 곱셈이 `n²`번이다. 빠른 구현이 맞는지 견줄 때 쓴다.
pub fn dft(values: &[Complex]) -> Vec<Complex> {
    let n = values.len();
    (0..n)
        .map(|k| {
            values
                .iter()
                .enumerate()
                .fold(Complex::default(), |sum, (t, &value)| {
                    // k·t가 커져도 각도는 n을 주기로 되풀이된다.
                    let angle = -2.0 * PI * ((k * t) % n) as f64 / n as f64;
                    sum.plus(value.times(Complex::unit(angle)))
                })
        })
        .collect()
}
// ANCHOR_END: dft

// ANCHOR: fft
/// 빠른 푸리에 변환. 길이가 2의 거듭제곱이어야 한다. 제자리에서 계산한다.
///
/// 길이 `n`인 변환은 짝수 번째 값들의 변환과 홀수 번째 값들의 변환(각각
/// 길이 `n/2`)으로 만들 수 있다. `X[k] = E[k] + wᵏ·O[k]`,
/// `X[k + n/2] = E[k] − wᵏ·O[k]`이고 `w = e^(−2πi/n)`이다. 반씩 나누기를
/// 길이 1까지 되풀이하면 단계가 `log₂ n`개이고, 단계마다 곱셈이 `n/2`번이다.
///
/// 재귀로 부르지 않는다. 먼저 값들을 "끝까지 나눴을 때 놓일 자리"로 옮겨
/// 두고(번호의 비트를 뒤집은 자리다), 길이 2짜리부터 합쳐 올라간다.
///
/// # Panics
///
/// 길이가 2의 거듭제곱이 아니면 공황에 빠진다.
pub fn fft(values: &mut [Complex], inverse: bool) {
    let n = values.len();
    assert!(n.is_power_of_two(), "길이 {n}은 2의 거듭제곱이 아닙니다");
    if n <= 1 {
        return;
    }
    let bits = n.trailing_zeros();
    for index in 0..n {
        let reversed = index.reverse_bits() >> (usize::BITS - bits);
        if index < reversed {
            values.swap(index, reversed);
        }
    }
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut size = 2;
    while size <= n {
        let step = Complex::unit(sign * 2.0 * PI / size as f64);
        for block in values.chunks_exact_mut(size) {
            let (even, odd) = block.split_at_mut(size / 2);
            let mut twist = Complex { re: 1.0, im: 0.0 };
            for (a, b) in even.iter_mut().zip(odd) {
                let turned = b.times(twist);
                (*a, *b) = (a.plus(turned), a.minus(turned));
                twist = twist.times(step);
            }
        }
        size *= 2;
    }
    if inverse {
        let scale = 1.0 / n as f64;
        for value in values.iter_mut() {
            (value.re, value.im) = (value.re * scale, value.im * scale);
        }
    }
}
// ANCHOR_END: fft

// ANCHOR: bluestein
/// 길이가 2의 거듭제곱이 아닌 변환을 2의 거듭제곱 길이의 FFT로 계산하는
/// 방법(Bluestein의 알고리즘).
///
/// Whisper는 표본 400개짜리 토막을 변환한다. 400은 2의 거듭제곱이 아니다.
/// 정의의 지수에 있는 `k·t`를 `(k² + t² − (k − t)²) / 2`로 바꿔 쓰면 변환이
/// 합성곱 하나가 되고, 합성곱은 길이를 넉넉히 늘린 FFT 셋으로 계산할 수
/// 있다:
///
/// `X[k] = c[k] · Σₜ (x[t]·c[t]) · conj(c[k − t])`,  `c[m] = e^(−πi·m²/n)`
pub struct Bluestein {
    n: usize,
    /// `c[m]`, 길이 `n`.
    chirp: Vec<Complex>,
    /// `conj(c)`를 길이 `m`의 원 위에 놓고 변환해 둔 것.
    kernel: Vec<Complex>,
}

impl Bluestein {
    /// 길이 `n`인 변환을 준비한다.
    pub fn new(n: usize) -> Self {
        // 합성곱의 결과가 겹치지 않으려면 길이가 2n − 1 이상이어야 한다.
        let m = (2 * n).max(1).next_power_of_two();
        let chirp: Vec<Complex> = (0..n)
            .map(|index| {
                // 제곱이 커지면 정밀도가 깎이므로 2n으로 나눈 나머지를 쓴다.
                let square = (index * index) % (2 * n.max(1));
                Complex::unit(-PI * square as f64 / n as f64)
            })
            .collect();
        let mut kernel = vec![Complex::default(); m];
        for (index, value) in chirp.iter().enumerate() {
            kernel[index] = value.conjugate();
            if index > 0 {
                kernel[m - index] = value.conjugate();
            }
        }
        fft(&mut kernel, false);
        Self { n, chirp, kernel }
    }

    /// 길이.
    pub fn len(&self) -> usize {
        self.n
    }

    /// 길이가 0인지.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// 변환한다. `scratch`는 빌려 쓰는 자리다. 길이를 알아서 맞춘다.
    ///
    /// # Panics
    ///
    /// `values`의 길이가 준비한 길이와 다르면 공황에 빠진다.
    pub fn transform(&self, values: &mut [Complex], scratch: &mut Vec<Complex>) {
        assert_eq!(values.len(), self.n, "길이가 다릅니다");
        scratch.clear();
        scratch.resize(self.kernel.len(), Complex::default());
        for ((slot, value), chirp) in scratch.iter_mut().zip(values.iter()).zip(&self.chirp) {
            *slot = value.times(*chirp);
        }
        fft(scratch, false);
        for (slot, kernel) in scratch.iter_mut().zip(&self.kernel) {
            *slot = slot.times(*kernel);
        }
        fft(scratch, true);
        for ((value, slot), chirp) in values.iter_mut().zip(scratch.iter()).zip(&self.chirp) {
            *value = slot.times(*chirp);
        }
    }
}
// ANCHOR_END: bluestein

#[cfg(test)]
mod tests {
    use super::*;

    fn signal(n: usize) -> Vec<Complex> {
        (0..n)
            .map(|t| Complex {
                re: ((t * 37 + 11) % 23) as f64 * 0.1 - 1.0,
                im: ((t * 17 + 5) % 13) as f64 * 0.05,
            })
            .collect()
    }

    fn assert_close(a: &[Complex], b: &[Complex], tolerance: f64) {
        assert_eq!(a.len(), b.len());
        for (index, (x, y)) in a.iter().zip(b).enumerate() {
            assert!(
                x.minus(*y).power().sqrt() < tolerance,
                "{index}: {x:?} 대 {y:?}"
            );
        }
    }

    #[test]
    fn a_pure_tone_lands_in_one_bin() {
        // 한 토막 안에서 세 번 도는 코사인. 3번과 n − 3번에만 값이 있다.
        let n = 16;
        let mut values: Vec<Complex> = (0..n)
            .map(|t| Complex {
                re: (2.0 * PI * 3.0 * t as f64 / n as f64).cos(),
                im: 0.0,
            })
            .collect();
        fft(&mut values, false);
        for (k, value) in values.iter().enumerate() {
            let expected = if k == 3 || k == n - 3 {
                n as f64 / 2.0
            } else {
                0.0
            };
            assert!(
                (value.power().sqrt() - expected).abs() < 1e-9,
                "{k}: {value:?}"
            );
        }
    }

    #[test]
    fn fast_matches_the_definition() {
        for n in [1, 2, 4, 8, 64, 512] {
            let mut values = signal(n);
            let expected = dft(&values);
            fft(&mut values, false);
            assert_close(&values, &expected, 1e-8);
            // 되돌리면 처음 값이다.
            fft(&mut values, true);
            assert_close(&values, &signal(n), 1e-10);
        }
    }

    #[test]
    fn any_length_matches_the_definition() {
        let mut scratch = Vec::new();
        for n in [1, 2, 3, 5, 7, 12, 100, 400, 401] {
            let plan = Bluestein::new(n);
            assert_eq!((plan.len(), plan.is_empty()), (n, false));
            let mut values = signal(n);
            let expected = dft(&values);
            plan.transform(&mut values, &mut scratch);
            assert_close(&values, &expected, 1e-8);
        }
    }

    #[test]
    #[should_panic(expected = "2의 거듭제곱")]
    fn fast_rejects_other_lengths() {
        fft(&mut signal(12), false);
    }
}
