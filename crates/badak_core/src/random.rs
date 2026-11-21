//! 난수 생성기 (12장).
//!
//! 표준 라이브러리에는 난수 생성기가 없으므로 직접 만든다. 시드가 같으면
//! 어느 기계에서든 같은 수열이 나온다.

// ANCHOR: pcg64
/// PCG64 난수 생성기 (PCG XSL-RR 128/64).
///
/// 128비트 상태를 "곱하고 더하기"로 한 스텝씩 나아가게 하고, 상태를 섞어
/// 64비트 수 하나를 꺼낸다. NumPy의 기본 생성기와 같은 알고리즘이다.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pcg64 {
    state: u128,
    /// 더하는 수. 홀수여야 한다.
    increment: u128,
}

/// 상태에 곱하는 수. PCG가 128비트 상태에 쓰는 기본값이다.
const MULTIPLIER: u128 = 0x2360_ED05_1FC6_5DA4_4385_DF64_9FCC_F645;

impl Pcg64 {
    /// 시드로 생성기를 만든다. 시드가 같으면 같은 수열이 나온다.
    pub fn new(seed: u64) -> Self {
        let mut rng = Self {
            state: 0,
            increment: 1,
        };
        rng.step();
        rng.state = rng.state.wrapping_add(u128::from(seed));
        rng.step();
        rng
    }

    /// 상태를 한 스텝 나아가게 한다.
    fn step(&mut self) {
        self.state = self
            .state
            .wrapping_mul(MULTIPLIER)
            .wrapping_add(self.increment);
    }

    /// 0 이상 2^64 미만의 고른 정수 하나.
    pub fn next_u64(&mut self) -> u64 {
        self.step();
        // 위 64비트와 아래 64비트를 XOR하고, 맨 위 6비트가 가리키는 만큼 돌린다.
        let folded = (self.state >> 64) as u64 ^ self.state as u64;
        folded.rotate_right((self.state >> 122) as u32)
    }

    /// 0 이상 1 미만의 고른 실수 하나.
    ///
    /// `f64`가 정확히 나타낼 수 있는 53비트만 쓴다. 결과는 `k / 2^53` 꼴이고
    /// 1이 되는 일은 없다.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}
// ANCHOR_END: pcg64

#[cfg(test)]
mod tests {
    use super::*;

    // 기대하는 값은 NumPy의 PCG64에서 얻었다. `reference/ch12_rng_vectors.py`.
    #[test]
    fn matches_numpy_pcg64() {
        let cases: [(u64, [u64; 2], [f64; 3]); 3] = [
            (
                0,
                [0xd4fe_b4e5_a4bc_fe09, 0xe85a_7fe0_71b0_26e6],
                [0.8320115147259805, 0.9076309130629743, 0.2279596459107528],
            ),
            (
                42,
                [0x3f04_2f64_9083_f6aa, 0x649a_f5df_0210_45f2],
                [0.24615760998905478, 0.3929895085767052, 0.10740772453548153],
            ),
            (
                u64::MAX,
                [0xfb5f_d3d9_b3d0_cca6, 0xe158_2e00_eb96_6e21],
                [0.9819309622287761, 0.8802517654654253, 0.8614996571682312],
            ),
        ];
        for (seed, integers, reals) in cases {
            let mut rng = Pcg64::new(seed);
            assert_eq!([rng.next_u64(), rng.next_u64()], integers, "시드 {seed}");
            let mut rng = Pcg64::new(seed);
            let drawn = [rng.next_f64(), rng.next_f64(), rng.next_f64()];
            assert_eq!(drawn, reals, "시드 {seed}");
        }
    }

    #[test]
    fn same_seed_same_sequence_and_different_seed_differs() {
        let (mut a, mut b, mut c) = (Pcg64::new(7), Pcg64::new(7), Pcg64::new(8));
        let first: Vec<u64> = (0..100).map(|_| a.next_u64()).collect();
        let second: Vec<u64> = (0..100).map(|_| b.next_u64()).collect();
        let third: Vec<u64> = (0..100).map(|_| c.next_u64()).collect();
        assert_eq!(first, second);
        assert_ne!(first, third);
    }

    #[test]
    fn reals_are_uniform_in_the_unit_interval() {
        let mut rng = Pcg64::new(1);
        let count = 200_000;
        let mut buckets = [0_u32; 10];
        let mut sum = 0.0;
        for _ in 0..count {
            let x = rng.next_f64();
            assert!((0.0..1.0).contains(&x));
            buckets[(x * 10.0) as usize] += 1;
            sum += x;
        }
        assert!((sum / f64::from(count) - 0.5).abs() < 0.005);
        // 칸마다 2만 개 안팎이어야 한다.
        for bucket in buckets {
            assert!((19_400..20_600).contains(&bucket), "{buckets:?}");
        }
    }
}
