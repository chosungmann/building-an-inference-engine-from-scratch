//! 회전 위치 임베딩(RoPE): 토큰의 위치를 벡터의 회전으로 나타낸다.

use badak_core::{Tensor, arena};
use badak_formats::Result;

use super::LlamaConfig;

// ANCHOR: frequencies
/// 벡터의 짝마다 쓸 회전 속도. 길이는 `head_dim / 2`다.
///
/// `i`번째 짝은 토큰 하나를 지날 때마다 `theta^(-2i / head_dim)` 라디안씩
/// 돈다. 앞쪽 짝은 빠르게, 뒤쪽 짝은 느리게 돈다.
pub fn inverse_frequencies(config: &LlamaConfig) -> Vec<f32> {
    let dim = config.head_dim();
    let mut frequencies: Vec<f32> = (0..dim / 2)
        .map(|i| 1.0 / config.rope_theta.powf((2 * i) as f32 / dim as f32))
        .collect();

    // Llama 3는 느리게 도는 짝을 더 느리게 만들어 긴 글에 대비한다.
    if let Some(scaling) = config.rope_scaling {
        let low_wavelength = scaling.original_max_position / scaling.low_freq_factor;
        let high_wavelength = scaling.original_max_position / scaling.high_freq_factor;
        for frequency in &mut frequencies {
            let wavelength = 2.0 * std::f32::consts::PI / *frequency;
            if wavelength > low_wavelength {
                // 주기가 긴 짝: 통째로 느리게.
                *frequency /= scaling.factor;
            } else if wavelength >= high_wavelength {
                // 그 사이의 짝: 두 경우를 부드럽게 잇는다.
                let smooth = (scaling.original_max_position / wavelength - scaling.low_freq_factor)
                    / (scaling.high_freq_factor - scaling.low_freq_factor);
                *frequency = (1.0 - smooth) * *frequency / scaling.factor + smooth * *frequency;
            }
        }
    }
    frequencies
}
// ANCHOR_END: frequencies

// ANCHOR: apply
/// `(토큰 수, 헤드 수, head_dim)`인 텐서의 벡터들을 위치에 따라 돌린다.
///
/// 벡터를 앞 절반과 뒤 절반으로 나누어, 같은 자리의 두 값을 한 짝으로 본다.
/// 짝 `(a, b)`를 평면 위의 점으로 보고 `위치 × 회전 속도`만큼 돌린다.
/// `first_position`은 첫 토큰의 위치다.
pub fn apply(x: &Tensor, frequencies: &[f32], first_position: usize) -> Result<Tensor> {
    // 14장: 프로파일러가 켜져 있으면 이 함수에 쓴 시간을 적는다.
    let _scope = badak_core::profile::scope("rope");
    let half = frequencies.len();
    let &[_, heads, dim] = x.shape() else {
        unreachable!("호출하는 쪽이 3차원 텐서를 넘긴다");
    };
    assert_eq!(
        dim,
        2 * half,
        "벡터의 길이가 회전 속도의 수와 맞지 않습니다"
    );

    let x = x.contiguous();
    let values = x.as_slice::<f32>()?;
    // 22장: 결과가 들어갈 자리를 먼저 얻고, 사인과 코사인을 담을 자리는
    // 스레드가 돌려 쓰는 것을 빌린다. 앞 절반에 사인, 뒤 절반에 코사인을 둔다.
    Ok(Tensor::build(x.shape(), |data| {
        data.copy_from_slice(values);
        arena::with_scratch(2 * half, |rotation| {
            let (sines, cosines) = rotation.split_at_mut(half);
            for (token, vectors) in data.chunks_exact_mut(heads * dim).enumerate() {
                let position = (first_position + token) as f32;
                // 이 위치에서 짝마다 돌릴 각도의 사인과 코사인.
                for ((sin, cos), frequency) in sines.iter_mut().zip(&mut *cosines).zip(frequencies)
                {
                    (*sin, *cos) = (position * frequency).sin_cos();
                }
                for vector in vectors.chunks_exact_mut(dim) {
                    let (front, back) = vector.split_at_mut(half);
                    for ((a, b), (&sin, &cos)) in
                        front.iter_mut().zip(back).zip(sines.iter().zip(&*cosines))
                    {
                        (*a, *b) = (*a * cos - *b * sin, *b * cos + *a * sin);
                    }
                }
            }
        });
        Ok(())
    })?)
}
// ANCHOR_END: apply

#[cfg(test)]
mod tests {
    use super::*;

    fn config(theta: f32) -> LlamaConfig {
        LlamaConfig {
            vocab_size: 1,
            hidden_size: 4,
            num_layers: 1,
            num_heads: 1,
            num_kv_heads: 1,
            intermediate_size: 1,
            rms_norm_eps: 1e-5,
            rope_theta: theta,
            rope_scaling: None,
            bos_token_id: None,
            eos_token_ids: Vec::new(),
        }
    }

    #[test]
    fn frequencies_fall_from_one() {
        // head_dim 4, theta 10000: 1, 10000^(-1/2) = 0.01
        let frequencies = inverse_frequencies(&config(10_000.0));
        assert_eq!(frequencies.len(), 2);
        assert_eq!(frequencies[0], 1.0);
        assert!((frequencies[1] - 0.01).abs() < 1e-8);
    }

    #[test]
    fn position_zero_is_left_alone_and_lengths_are_kept() {
        let frequencies = inverse_frequencies(&config(10_000.0));
        let x =
            Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0], &[2, 1, 4]).unwrap();
        let y = apply(&x, &frequencies, 0).unwrap().to_vec::<f32>().unwrap();
        // 위치 0은 돌리지 않는다.
        assert_eq!(y[..4], [1.0, 2.0, 3.0, 4.0]);
        // 위치 1: 짝 (1, 3)은 1라디안, 짝 (2, 4)는 0.01라디안 돈다.
        let (sin, cos) = 1.0_f32.sin_cos();
        assert!((y[4] - (1.0 * cos - 3.0 * sin)).abs() < 1e-6);
        assert!((y[6] - (3.0 * cos + 1.0 * sin)).abs() < 1e-6);
        // 회전은 길이를 바꾸지 않는다.
        let length = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>();
        assert!((length(&y[4..]) - length(&y[..4])).abs() < 1e-4);
    }

    #[test]
    fn dot_products_depend_only_on_the_distance() {
        // 같은 두 벡터를 위치 (2, 5)와 (10, 13)에 놓으면 내적이 같아야 한다.
        let frequencies = inverse_frequencies(&config(10_000.0));
        let q = Tensor::from_vec(vec![0.3_f32, -1.2, 0.8, 0.5], &[1, 1, 4]).unwrap();
        let k = Tensor::from_vec(vec![1.1_f32, 0.4, -0.7, 0.9], &[1, 1, 4]).unwrap();
        let dot = |q_position, k_position| {
            let q = apply(&q, &frequencies, q_position)
                .unwrap()
                .to_vec::<f32>()
                .unwrap();
            let k = apply(&k, &frequencies, k_position)
                .unwrap()
                .to_vec::<f32>()
                .unwrap();
            badak_core::linalg::dot(&q, &k)
        };
        assert!((dot(5, 2) - dot(13, 10)).abs() < 1e-5);
        assert!((dot(5, 2) - dot(5, 3)).abs() > 1e-3);
    }
}
