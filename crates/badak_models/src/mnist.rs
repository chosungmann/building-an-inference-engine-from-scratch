//! 손글씨 숫자를 분류하는 다층 퍼셉트론 (7장).

use std::path::Path;

use badak_core::{DType, Tensor, ops};
use badak_formats::safetensors::SafeTensors;
use badak_formats::{Error, Result};

/// 사진 한 장의 한 변의 픽셀 수.
pub const SIDE: usize = 28;

// ANCHOR: model
/// 입력 784개, 은닉층 하나, 출력 10개인 분류 모델.
pub struct MnistMlp {
    fc1_weight: Tensor,
    fc1_bias: Tensor,
    fc2_weight: Tensor,
    fc2_bias: Tensor,
    /// 학습할 때 입력을 정규화한 평균과 표준편차.
    mean: f32,
    std: f32,
}

/// 순전파의 중간 결과. 층마다 정답과 대조할 때 쓴다.
pub struct Trace {
    /// 첫 선형 층의 출력.
    pub fc1: Tensor,
    /// ReLU를 지난 뒤.
    pub relu: Tensor,
    /// 둘째 선형 층의 출력. 숫자 열 개에 대한 점수다.
    pub logits: Tensor,
}
// ANCHOR_END: model

fn invalid(message: String) -> Error {
    Error::Parse {
        format: "MNIST 모델",
        offset: 0,
        message,
    }
}

impl MnistMlp {
    // ANCHOR: load
    /// safetensors 파일에서 가중치와 전처리 값을 읽는다.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = SafeTensors::open(path)?;
        let number = |key: &str| -> Result<f32> {
            file.metadata()
                .get(key)
                .and_then(|text| text.parse().ok())
                .ok_or_else(|| invalid(format!("메타데이터 {key}가 없거나 숫자가 아닙니다")))
        };
        let model = Self {
            fc1_weight: file.tensor("fc1.weight")?,
            fc1_bias: file.tensor("fc1.bias")?,
            fc2_weight: file.tensor("fc2.weight")?,
            fc2_bias: file.tensor("fc2.bias")?,
            mean: number("input_mean")?,
            std: number("input_std")?,
        };

        // 파일의 형상이 이 모델의 구조와 맞는지 확인한다.
        let (&[hidden, input], &[classes, hidden2]) =
            (model.fc1_weight.shape(), model.fc2_weight.shape())
        else {
            return Err(invalid("가중치가 2차원이 아닙니다".into()));
        };
        if input != SIDE * SIDE
            || hidden != hidden2
            || model.fc1_bias.shape() != [hidden]
            || model.fc2_bias.shape() != [classes]
        {
            return Err(invalid(format!(
                "형상이 맞지 않습니다: fc1 {:?}, fc2 {:?}",
                model.fc1_weight.shape(),
                model.fc2_weight.shape()
            )));
        }
        Ok(model)
    }
    // ANCHOR_END: load

    /// 전처리의 평균과 표준편차를 바꾼 모델을 돌려준다.
    ///
    /// 학습할 때와 다른 전처리를 하면 어떻게 되는지 실험할 때 쓴다.
    pub fn with_normalization(mut self, mean: f32, std: f32) -> Self {
        (self.mean, self.std) = (mean, std);
        self
    }

    /// 파라미터의 수.
    pub fn parameter_count(&self) -> usize {
        [
            &self.fc1_weight,
            &self.fc1_bias,
            &self.fc2_weight,
            &self.fc2_bias,
        ]
        .iter()
        .map(|tensor| tensor.len())
        .sum()
    }

    // ANCHOR: preprocess
    /// 픽셀 값(0~255)을 모델의 입력으로 바꾼다.
    ///
    /// `images`는 `(장 수, 28, 28)`인 `u8` 텐서이고, 결과는 `(장 수, 784)`인
    /// `f32` 텐서다. 학습할 때와 똑같은 계산을 해야 한다. 0과 1 사이로
    /// 바꾸고, 평균을 빼고, 표준편차로 나눈다.
    pub fn preprocess(&self, images: &Tensor) -> Result<Tensor> {
        let &[count, SIDE, SIDE] = images.shape() else {
            return Err(invalid(format!(
                "입력의 형상이 (장 수, 28, 28)이 아닙니다: {:?}",
                images.shape()
            )));
        };
        let pixels = images.contiguous();
        let values = pixels
            .as_slice::<u8>()?
            .iter()
            .map(|&pixel| (f32::from(pixel) / 255.0 - self.mean) / self.std)
            .collect();
        Ok(Tensor::from_vec(values, &[count, SIDE * SIDE])?)
    }
    // ANCHOR_END: preprocess

    // ANCHOR: forward
    /// 순전파. `(장 수, 784)`를 받아 `(장 수, 10)`의 점수를 돌려준다.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.trace(x)?.logits)
    }

    /// 순전파를 하면서 층마다의 결과를 남긴다.
    pub fn trace(&self, x: &Tensor) -> Result<Trace> {
        let fc1 = ops::linear(x, &self.fc1_weight, Some(&self.fc1_bias))?;
        let relu = ops::relu(&fc1)?;
        let logits = ops::linear(&relu, &self.fc2_weight, Some(&self.fc2_bias))?;
        Ok(Trace { fc1, relu, logits })
    }

    /// 사진들을 받아 각각 어떤 숫자인지 답한다.
    pub fn classify(&self, images: &Tensor) -> Result<Vec<i32>> {
        let logits = self.forward(&self.preprocess(images)?)?;
        Ok(ops::argmax(&logits, 1)?.to_vec::<i32>()?)
    }
    // ANCHOR_END: forward
}

// ANCHOR: draw
/// 글자로 그린 그림을 `(1, 28, 28)`인 `u8` 텐서로 바꾼다.
///
/// `#`은 가장 밝은 픽셀(255), `+`는 중간(128), 그 밖의 글자는 검은
/// 픽셀(0)이다. 줄이나 칸이 모자라면 검은 픽셀로 채운다.
pub fn image_from_art(art: &str) -> Tensor {
    let mut pixels = vec![0_u8; SIDE * SIDE];
    for (row, line) in art.lines().take(SIDE).enumerate() {
        for (column, character) in line.chars().take(SIDE).enumerate() {
            pixels[row * SIDE + column] = match character {
                '#' => 255,
                '+' => 128,
                _ => 0,
            };
        }
    }
    Tensor::from_vec(pixels, &[1, SIDE, SIDE]).expect("원소 수는 형상과 같다")
}
// ANCHOR_END: draw

/// `u8` 사진 한 장을 글자로 그린다. 눈으로 확인할 때 쓴다.
pub fn art_from_image(image: &Tensor) -> Result<String> {
    if image.dtype() != DType::U8 || image.len() != SIDE * SIDE {
        return Err(invalid("28 × 28인 u8 사진이 아닙니다".into()));
    }
    let pixels = image.to_vec::<u8>()?;
    let (rows, _) = pixels.as_chunks::<SIDE>();
    Ok(rows
        .iter()
        .map(|row| {
            row.iter()
                .map(|&pixel| match pixel {
                    0..64 => '.',
                    64..192 => '+',
                    _ => '#',
                })
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn art_round_trips() {
        let art = "#+.\n.#";
        let image = image_from_art(art);
        assert_eq!(image.shape(), [1, SIDE, SIDE]);
        assert_eq!(image.get::<u8>(&[0, 0, 0]).unwrap(), 255);
        assert_eq!(image.get::<u8>(&[0, 0, 1]).unwrap(), 128);
        assert_eq!(image.get::<u8>(&[0, 1, 1]).unwrap(), 255);
        let back = art_from_image(&image.squeeze(0).unwrap()).unwrap();
        assert!(back.starts_with("#+.."));
        assert_eq!(back.lines().count(), SIDE);
    }
}
