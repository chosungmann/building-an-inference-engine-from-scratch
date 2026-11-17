"""12장: 난수 생성기 PCG64의 시험 값을 NumPy에서 얻는다.

실행: python reference/ch12_rng_vectors.py

badak의 `Pcg64::new(seed)`가 하는 초기화를 파이썬의 정수로 따라 한 뒤,
그 상태를 NumPy의 PCG64에 넣고 수를 뽑는다. 출력한 값은
`crates/badak_core/src/random.rs`의 테스트에 적혀 있다.
"""

from numpy.random import PCG64, Generator

MULTIPLIER = 0x2360ED051FC65DA44385DF649FCCF645
MASK = (1 << 128) - 1


def initial_state(seed):
    """`Pcg64::new`와 같은 초기화: 한 걸음, 씨앗을 더하고, 한 걸음."""
    increment = 1
    state = (0 * MULTIPLIER + increment) & MASK
    state = (state + seed) & MASK
    state = (state * MULTIPLIER + increment) & MASK
    return state, increment


def generator(seed):
    state, increment = initial_state(seed)
    bits = PCG64()
    snapshot = bits.state
    snapshot["state"] = {"state": state, "inc": increment}
    snapshot["has_uint32"] = 0
    bits.state = snapshot
    return bits


def main():
    for seed in (0, 42, 2**64 - 1):
        integers = [hex(int(x)) for x in generator(seed).random_raw(2)]
        reals = [repr(float(x)) for x in Generator(generator(seed)).random(3)]
        print(f"씨앗 {seed}: 정수 {integers}, 실수 {reals}")


if __name__ == "__main__":
    main()
