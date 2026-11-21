//! 신경망의 기본 부품: 활성화 함수, 선형 층, 다층 퍼셉트론 (3장).
//!
//! 여기의 층은 벡터 하나를 받아 벡터 하나를 내놓는다. 여러 입력을 한꺼번에
//! 처리하는 배치와 일반적인 형상은 4장에서 텐서를 만든 뒤에 다룬다.

use crate::linalg::matvec;

// ANCHOR: activations
/// ReLU. 음수를 0으로 바꾸고 양수는 그대로 둔다.
pub fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// 시그모이드. 모든 실수를 0과 1 사이로 눌러 담는다.
///
/// `x`가 큰 음수일 때 `exp(-x)`가 넘치지 않도록 부호에 따라 식을 나눈다.
pub fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// 하이퍼볼릭 탄젠트. 모든 실수를 -1과 1 사이로 눌러 담는다.
pub fn tanh(x: f32) -> f32 {
    x.tanh()
}

/// SiLU (Swish라고도 한다). `x * sigmoid(x)`.
pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}
// ANCHOR_END: activations

// ANCHOR: silu_many
/// `e^x`을 값 여러 개에 한꺼번에 계산하려고 쓰는 근사 (32장).
///
/// `f32::exp`는 값 하나를 받는 함수 호출이어서, 반복문 안에 있으면
/// 컴파일러가 반복문을 벡터 명령으로 바꾸지 못한다. 이 함수는 덧셈과
/// 곱셈과 버림뿐이라 바꿀 수 있다.
///
/// `x = n·ln2 + r`로 나눈다(`n`은 정수, `|r| ≤ ln2 / 2`). `e^x = 2ⁿ × e^r`
/// 이고, `e^r`은 좁은 구간에서 6차 다항식으로 충분하고, `2ⁿ`은 지수
/// 비트에 `n`을 적어 넣으면 된다. 다항식의 계수는 Cephes 라이브러리의
/// `expf`에서 가져왔다. 상대 오차는 `2 × 10⁻⁷`쯤이다.
#[inline]
pub fn exp_fast(x: f32) -> f32 {
    // 결과가 f32에 담기지 않는 곳은 끝값으로 자른다.
    let x = x.clamp(-87.0, 88.0);
    let n = (x * std::f32::consts::LOG2_E + 0.5).floor();
    // ln2를 둘로 나눠 빼면 `r`의 끝자리가 덜 깎인다.
    let r = x - n * 0.693_359_4 + n * 2.121_944_4e-4;
    const C: [f32; 6] = [
        1.987_569_1e-4,
        1.398_2e-3,
        8.333_452e-3,
        4.166_579_6e-2,
        1.666_666_5e-1,
        0.5,
    ];
    let polynomial = C.iter().fold(0.0, |sum, &c| sum * r + c);
    let value = polynomial * r * r + r + 1.0;
    // 2ⁿ: 지수 비트(23번째부터)에 `n + 127`을 적는다.
    let power = f32::from_bits(((n as i32 + 127) as u32) << 23);
    value * power
}

/// 값들에 SiLU를 제자리에서 적용한다. [`silu`]를 하나씩 부르는 것과 같은
/// 계산이고, 지수 함수만 근사로 바꿔 반복문이 벡터 명령으로 바뀌게 했다.
/// 결과는 [`silu`]와 `10⁻⁶`쯤의 상대 오차 안에서 같다.
pub fn silu_in_place(values: &mut [f32]) {
    for value in values.iter_mut() {
        // 아주 큰 음수에서 `x × sigmoid(x)`는 0으로 간다. 지수 함수의
        // 근사가 끝값에서 잘리므로 `x`도 같은 곳에서 잘라 둔다. −87에서
        // 값은 이미 `10⁻³⁶`이다.
        let x = value.max(-87.0);
        *value = x / (1.0 + exp_fast(-x));
    }
}

/// 값들에 GELU를 제자리에서 적용한다. [`gelu`]와 같은 식이고 지수 함수만
/// [`exp_fast`]로 바꿨다.
pub fn gelu_in_place(values: &mut [f32]) {
    const P: f32 = 0.327_591_1;
    const A: [f32; 5] = [
        0.254_829_6,
        -0.284_496_72,
        1.421_413_8,
        -1.453_152_1,
        1.061_405_4,
    ];
    for value in values.iter_mut() {
        let x = *value;
        let z = (x * std::f32::consts::FRAC_1_SQRT_2).abs();
        let t = 1.0 / (1.0 + P * z);
        let polynomial = A.iter().rev().fold(0.0, |sum, &a| (sum + a) * t);
        let erf = (1.0 - polynomial * exp_fast(-z * z)).copysign(x);
        *value = 0.5 * x * (1.0 + erf);
    }
}

/// 소프트맥스를 제자리에서 계산한다. [`crate::prob::softmax_in_place`]와
/// 같은 식이고 지수 함수만 [`exp_fast`]로 바꿨다.
pub fn softmax_fast_in_place(scores: &mut [f32]) {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for score in scores.iter_mut() {
        *score = exp_fast(*score - max);
    }
    let sum: f32 = scores.iter().sum();
    for score in scores.iter_mut() {
        *score /= sum;
    }
}
// ANCHOR_END: silu_many

// ANCHOR: gelu
/// 오차 함수 `erf`. 표준 라이브러리에 없어서 근사식으로 계산한다 (32장).
///
/// Abramowitz와 Stegun의 식 7.1.26이다. `t = 1 / (1 + p·|x|)`의 5차
/// 다항식에 `exp(−x²)`를 곱한다. 식 자체의 오차는 `1.5 × 10⁻⁷` 이하이고,
/// `f32`의 정밀도와 같은 크기다.
pub fn erf(x: f32) -> f32 {
    const P: f32 = 0.327_591_1;
    const A: [f32; 5] = [
        0.254_829_6,
        -0.284_496_72,
        1.421_413_8,
        -1.453_152_1,
        1.061_405_4,
    ];
    let magnitude = x.abs();
    let t = 1.0 / (1.0 + P * magnitude);
    // 다항식을 안쪽부터 계산한다(호너의 방법).
    let polynomial = A.iter().rev().fold(0.0, |sum, &a| (sum + a) * t);
    let value = 1.0 - polynomial * (-magnitude * magnitude).exp();
    value.copysign(x)
}

/// GELU. `x × Φ(x)`이고 `Φ`는 표준 정규 분포의 누적 분포 함수다
/// (`Φ(x) = (1 + erf(x / √2)) / 2`). ReLU처럼 음수를 누르되 0 근처에서
/// 매끄럽다. BERT와 ViT의 MLP가 쓴다.
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}
// ANCHOR_END: gelu

// ANCHOR: activation_enum
/// 층 사이에 넣을 활성화 함수의 종류.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// 아무것도 하지 않는다.
    Identity,
    /// [`relu`]
    Relu,
    /// [`sigmoid`]
    Sigmoid,
    /// [`tanh`]
    Tanh,
    /// [`silu`]
    Silu,
}

impl Activation {
    /// 값 하나에 활성화 함수를 적용한다.
    pub fn apply(self, x: f32) -> f32 {
        match self {
            Self::Identity => x,
            Self::Relu => relu(x),
            Self::Sigmoid => sigmoid(x),
            Self::Tanh => tanh(x),
            Self::Silu => silu(x),
        }
    }

    /// 벡터의 모든 원소에 제자리에서 적용한다.
    pub fn apply_in_place(self, values: &mut [f32]) {
        for value in values {
            *value = self.apply(*value);
        }
    }
}
// ANCHOR_END: activation_enum

// ANCHOR: linear
/// 선형 층 `y = W x + b`.
///
/// 길이 `in_features`인 벡터를 받아 길이 `out_features`인 벡터를 내놓는다.
#[derive(Debug, Clone, PartialEq)]
pub struct Linear {
    /// `out_features × in_features` 행렬을 행 우선으로 펼친 것.
    weight: Vec<f32>,
    /// 길이 `out_features`인 편향.
    bias: Vec<f32>,
    in_features: usize,
    out_features: usize,
}

impl Linear {
    /// 가중치와 편향으로 층을 만든다.
    ///
    /// # Panics
    ///
    /// `weight`의 길이가 `out_features * in_features`가 아니거나 `bias`의
    /// 길이가 `out_features`가 아니면 패닉한다.
    pub fn new(weight: Vec<f32>, bias: Vec<f32>, in_features: usize, out_features: usize) -> Self {
        assert_eq!(
            weight.len(),
            out_features * in_features,
            "가중치의 크기가 맞지 않습니다"
        );
        assert_eq!(bias.len(), out_features, "편향의 크기가 맞지 않습니다");
        Self {
            weight,
            bias,
            in_features,
            out_features,
        }
    }

    /// 순전파. `W x + b`를 계산한다.
    ///
    /// # Panics
    ///
    /// `x`의 길이가 `in_features`가 아니면 패닉한다.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut y = matvec(&self.weight, self.out_features, self.in_features, x);
        for (value, bias) in y.iter_mut().zip(&self.bias) {
            *value += bias;
        }
        y
    }

    /// 입력 벡터의 길이.
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// 출력 벡터의 길이.
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// 이 층의 파라미터 수. 가중치와 편향을 합친 것이다.
    pub fn parameter_count(&self) -> usize {
        self.weight.len() + self.bias.len()
    }
}
// ANCHOR_END: linear

// ANCHOR: mlp
/// 다층 퍼셉트론. 선형 층을 차례로 쌓고 그 사이에 활성화 함수를 넣는다.
#[derive(Debug, Clone, PartialEq)]
pub struct Mlp {
    layers: Vec<Linear>,
    activation: Activation,
}

impl Mlp {
    /// 층들과, 층 사이에 넣을 활성화 함수로 모델을 만든다.
    ///
    /// # Panics
    ///
    /// 층이 하나도 없거나, 이웃한 층의 출력 길이와 입력 길이가 다르면
    /// 패닉한다.
    pub fn new(layers: Vec<Linear>, activation: Activation) -> Self {
        assert!(!layers.is_empty(), "층이 하나 이상 있어야 합니다");
        for pair in layers.windows(2) {
            assert_eq!(
                pair[0].out_features(),
                pair[1].in_features(),
                "이웃한 층의 크기가 이어지지 않습니다"
            );
        }
        Self { layers, activation }
    }

    /// 순전파. 마지막 층 뒤에는 활성화 함수를 적용하지 않는다.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let (last, hidden) = self.layers.split_last().expect("층이 하나 이상 있다");
        let mut values = x.to_vec();
        for layer in hidden {
            values = layer.forward(&values);
            self.activation.apply_in_place(&mut values);
        }
        last.forward(&values)
    }

    /// 모델 전체의 파라미터 수.
    pub fn parameter_count(&self) -> usize {
        self.layers.iter().map(Linear::parameter_count).sum()
    }
}
// ANCHOR_END: mlp

// ANCHOR: xor
/// XOR을 계산하는 작은 MLP. 가중치를 손으로 정했다.
///
/// 은닉층의 첫 뉴런은 "입력의 합", 둘째 뉴런은 "입력이 둘 다 1인가"를
/// 계산하고, 출력층이 첫째에서 둘째의 두 배를 뺀다.
pub fn xor_mlp() -> Mlp {
    let hidden = Linear::new(vec![1.0, 1.0, 1.0, 1.0], vec![0.0, -1.0], 2, 2);
    let output = Linear::new(vec![1.0, -2.0], vec![0.0], 2, 1);
    Mlp::new(vec![hidden, output], Activation::Relu)
}
// ANCHOR_END: xor

#[cfg(test)]
mod tests {
    use super::*;
    use crate::approx::allclose;
    use crate::linalg::matmul;

    #[test]
    fn activations_match_known_values() {
        assert_eq!(relu(-2.0), 0.0);
        assert_eq!(relu(3.0), 3.0);
        assert_eq!(sigmoid(0.0), 0.5);
        assert_eq!(tanh(0.0), 0.0);
        assert_eq!(silu(0.0), 0.0);
        // sigmoid(ln 3) = 3 / (1 + 3) = 0.75
        assert!((sigmoid(3.0_f32.ln()) - 0.75).abs() < 1e-6);
        // silu(x) = x * sigmoid(x)
        assert!((silu(3.0_f32.ln()) - 3.0_f32.ln() * 0.75).abs() < 1e-6);
    }

    #[test]
    fn sigmoid_is_stable_at_extremes() {
        assert_eq!(sigmoid(1000.0), 1.0);
        assert_eq!(sigmoid(-1000.0), 0.0);
        assert!(sigmoid(-50.0) > 0.0);
        // sigmoid(-x) = 1 - sigmoid(x)
        assert!((sigmoid(-2.0) - (1.0 - sigmoid(2.0))).abs() < 1e-6);
    }

    #[test]
    fn linear_computes_wx_plus_b() {
        // [1 2 3]   [1]   [10]   [-2 + 10]   [ 8]
        // [4 5 6] · [0] + [20] = [-2 + 20] = [18]
        //           [-1]
        let layer = Linear::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![10.0, 20.0], 3, 2);
        assert_eq!(layer.forward(&[1.0, 0.0, -1.0]), [8.0, 18.0]);
        assert_eq!(layer.parameter_count(), 8);
    }

    #[test]
    #[should_panic(expected = "가중치의 크기")]
    fn linear_rejects_wrong_weight_size() {
        Linear::new(vec![1.0; 5], vec![0.0; 2], 3, 2);
    }

    // ANCHOR: collapse_test
    #[test]
    fn two_linear_layers_without_activation_collapse_into_one() {
        let first = Linear::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![0.5, -0.5, 1.0],
            2,
            3,
        );
        let second = Linear::new(vec![1.0, -1.0, 2.0, 0.5, 0.0, -2.0], vec![3.0, -3.0], 3, 2);
        let stacked = Mlp::new(vec![first.clone(), second.clone()], Activation::Identity);

        // 합친 층: W = W2 W1, b = W2 b1 + b2
        let weight = matmul(&second.weight, &first.weight, 2, 3, 2);
        let bias = second.forward(&first.bias);
        let merged = Linear::new(weight, bias, 2, 2);

        for x in [[1.0, 2.0], [-3.0, 0.25], [0.0, 0.0]] {
            assert!(allclose(
                &stacked.forward(&x),
                &merged.forward(&x),
                1e-6,
                1e-6
            ));
        }
    }
    // ANCHOR_END: collapse_test

    #[test]
    fn hand_built_mlp_computes_xor() {
        let model = xor_mlp();
        assert_eq!(model.forward(&[0.0, 0.0]), [0.0]);
        assert_eq!(model.forward(&[0.0, 1.0]), [1.0]);
        assert_eq!(model.forward(&[1.0, 0.0]), [1.0]);
        assert_eq!(model.forward(&[1.0, 1.0]), [0.0]);
        assert_eq!(model.parameter_count(), 9);
    }

    #[test]
    #[should_panic(expected = "이어지지 않습니다")]
    fn mlp_rejects_layers_that_do_not_chain() {
        let a = Linear::new(vec![0.0; 6], vec![0.0; 3], 2, 3);
        let b = Linear::new(vec![0.0; 4], vec![0.0; 2], 2, 2);
        Mlp::new(vec![a, b], Activation::Relu);
    }
}

#[cfg(test)]
mod gelu_tests {
    use super::*;

    #[test]
    fn erf_matches_known_values() {
        // 값은 수치 계산 책의 표에서 가져왔다.
        let known = [
            (0.0, 0.0),
            (0.5, 0.520_499_9),
            (1.0, 0.842_700_8),
            (2.0, 0.995_322_3),
            (3.5, 0.999_999_3),
        ];
        for (x, expected) in known {
            assert!((erf(x) - expected).abs() < 5e-7, "erf({x}) = {}", erf(x));
            assert!((erf(-x) + expected).abs() < 5e-7);
        }
        assert_eq!(erf(f32::INFINITY), 1.0);
        assert_eq!(erf(f32::NEG_INFINITY), -1.0);
    }

    #[test]
    fn vectorized_silu_matches_the_scalar_one() {
        let mut values: Vec<f32> = (-4000..=4000).map(|i| i as f32 * 0.01).collect();
        values.extend([
            -100.0,
            100.0,
            -87.5,
            88.5,
            0.0,
            -0.0,
            1e-8,
            f32::MAX,
            f32::MIN,
        ]);
        let expected: Vec<f32> = values.iter().map(|&x| silu(x)).collect();
        silu_in_place(&mut values);
        for (index, (got, want)) in values.iter().zip(&expected).enumerate() {
            let tolerance = 2e-6 * want.abs().max(1e-3);
            assert!((got - want).abs() <= tolerance, "{index}: {got} 대 {want}");
        }
        // GELU와 소프트맥스의 빠른 버전도 느린 버전과 같은 값을 낸다.
        let mut values: Vec<f32> = (-800..=800).map(|i| i as f32 * 0.01).collect();
        let expected: Vec<f32> = values.iter().map(|&x| gelu(x)).collect();
        gelu_in_place(&mut values);
        for (got, want) in values.iter().zip(&expected) {
            assert!((got - want).abs() <= 2e-6, "{got} 대 {want}");
        }
        let mut scores = [1.0_f32, -2.0, 0.5, 30.0, 29.0];
        let mut slow = scores;
        softmax_fast_in_place(&mut scores);
        crate::prob::softmax_in_place(&mut slow);
        for (got, want) in scores.iter().zip(&slow) {
            assert!((got - want).abs() <= 1e-6, "{got} 대 {want}");
        }
        // 지수 함수의 근사 자체도 본다.
        for x in [-80.0_f32, -10.0, -1.0, -0.3, 0.0, 0.3, 1.0, 10.0, 80.0] {
            let (got, want) = (exp_fast(x), x.exp());
            assert!(
                (got - want).abs() <= 3e-7 * want,
                "exp({x}) = {got}, {want}"
            );
        }
    }

    #[test]
    fn gelu_has_the_expected_shape() {
        assert_eq!(gelu(0.0), 0.0);
        assert!((gelu(1.0) - 0.841_344_7).abs() < 1e-6);
        assert!((gelu(-1.0) + 0.158_655_3).abs() < 1e-6);
        // 큰 양수는 그대로, 큰 음수는 0으로.
        assert!((gelu(6.0) - 6.0).abs() < 1e-6);
        assert!(gelu(-6.0).abs() < 1e-6);
        // 0 바로 아래에 가장 낮은 곳이 있다.
        assert!(gelu(-0.75) < gelu(-0.1) && gelu(-0.75) < gelu(-3.0));
    }
}
