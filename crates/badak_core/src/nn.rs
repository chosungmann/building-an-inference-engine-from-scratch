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
