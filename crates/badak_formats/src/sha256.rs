//! SHA-256 (44장). 내려받은 모델 파일이 보낸 그대로인지 확인하는 데 쓴다.
//!
//! FIPS 180-4의 6.2절을 그대로 옮겼다. 빠르게 만들지 않았다. 폰의 CPU에는
//! 이 계산을 위한 명령이 따로 있지만 쓰지 않는다. 모델 파일은 받을 때 한
//! 번만 확인한다.

use std::io::Read;

// ANCHOR: constants
/// 처음 여덟 소수의 제곱근의 소수 부분 32비트 (FIPS 180-4, 5.3.3절).
const INITIAL: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// 처음 예순네 소수의 세제곱근의 소수 부분 32비트 (FIPS 180-4, 4.2.2절).
const ROUND: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];
// ANCHOR_END: constants

/// 64바이트 덩어리의 크기.
const BLOCK: usize = 64;

/// 바이트를 여러 번에 나누어 넣을 수 있는 SHA-256 계산기.
#[derive(Debug, Clone)]
pub struct Sha256 {
    state: [u32; 8],
    /// 아직 한 덩어리를 채우지 못한 바이트.
    pending: [u8; BLOCK],
    filled: usize,
    /// 지금까지 넣은 바이트의 수.
    total: u64,
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha256 {
    /// 빈 계산기.
    pub fn new() -> Self {
        Self {
            state: INITIAL,
            pending: [0; BLOCK],
            filled: 0,
            total: 0,
        }
    }

    /// 바이트를 더 넣는다.
    pub fn update(&mut self, mut bytes: &[u8]) {
        self.total += bytes.len() as u64;
        if self.filled > 0 {
            let take = (BLOCK - self.filled).min(bytes.len());
            self.pending[self.filled..self.filled + take].copy_from_slice(&bytes[..take]);
            self.filled += take;
            bytes = &bytes[take..];
            if self.filled < BLOCK {
                return;
            }
            let block = self.pending;
            compress(&mut self.state, &block);
            self.filled = 0;
        }
        let (blocks, rest) = bytes.as_chunks::<BLOCK>();
        for block in blocks {
            compress(&mut self.state, block);
        }
        self.pending[..rest.len()].copy_from_slice(rest);
        self.filled = rest.len();
    }

    // ANCHOR: finish
    /// 계산을 끝내고 32바이트의 해시를 돌려준다.
    pub fn finish(mut self) -> [u8; 32] {
        // 끝에 1비트 하나, 0을 채우고, 마지막 8바이트에 전체 길이(비트).
        let bits = self.total * 8;
        let mut tail = [0_u8; 2 * BLOCK];
        tail[0] = 0x80;
        let zeros = (BLOCK + 55 - self.filled) % BLOCK;
        tail[1 + zeros..1 + zeros + 8].copy_from_slice(&bits.to_be_bytes());
        let total = self.total;
        self.update(&tail[..1 + zeros + 8]);
        self.total = total;
        debug_assert_eq!(self.filled, 0);
        let mut hash = [0_u8; 32];
        for (target, word) in hash.as_chunks_mut::<4>().0.iter_mut().zip(self.state) {
            *target = word.to_be_bytes();
        }
        hash
    }
    // ANCHOR_END: finish
}

// ANCHOR: compress
/// 덩어리 하나를 상태에 섞는다 (FIPS 180-4, 6.2.2절).
fn compress(state: &mut [u32; 8], block: &[u8; BLOCK]) {
    let mut words = [0_u32; 64];
    for (word, bytes) in words.iter_mut().zip(block.as_chunks::<4>().0) {
        *word = u32::from_be_bytes(*bytes);
    }
    for index in 16..64 {
        let (before, after) = (words[index - 15], words[index - 2]);
        let small0 = before.rotate_right(7) ^ before.rotate_right(18) ^ (before >> 3);
        let small1 = after.rotate_right(17) ^ after.rotate_right(19) ^ (after >> 10);
        words[index] = words[index - 16]
            .wrapping_add(small0)
            .wrapping_add(words[index - 7])
            .wrapping_add(small1);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for index in 0..64 {
        let big1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let choose = (e & f) ^ (!e & g);
        let first = h
            .wrapping_add(big1)
            .wrapping_add(choose)
            .wrapping_add(ROUND[index])
            .wrapping_add(words[index]);
        let big0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let majority = (a & b) ^ (a & c) ^ (b & c);
        let second = big0.wrapping_add(majority);
        (h, g, f, e, d, c, b, a) = (
            g,
            f,
            e,
            d.wrapping_add(first),
            c,
            b,
            a,
            first.wrapping_add(second),
        );
    }
    for (target, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
        *target = target.wrapping_add(value);
    }
}
// ANCHOR_END: compress

/// 바이트열의 해시.
pub fn digest(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finish()
}

/// 읽을 수 있는 것(파일)을 끝까지 읽어 해시와 읽은 바이트 수를 돌려준다.
/// 파일 전체를 메모리에 올리지 않는다.
pub fn digest_reader(mut reader: impl Read) -> std::io::Result<([u8; 32], u64)> {
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 64 * 1024];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            return Ok((hasher.clone().finish(), hasher.total));
        }
        hasher.update(&buffer[..count]);
    }
}

/// 해시를 소문자 16진수 예순네 글자로 적는다.
pub fn hex(hash: &[u8; 32]) -> String {
    hash.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // FIPS 180-4의 예와 NIST의 예제 문서에 실린 값들.
    #[test]
    fn matches_the_published_examples() {
        assert_eq!(
            hex(&digest(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex(&digest(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            hex(&digest(
                b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
            )),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        let million = vec![b'a'; 1_000_000];
        assert_eq!(
            hex(&digest(&million)),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    #[test]
    fn pieces_give_the_same_hash_as_the_whole() {
        let bytes: Vec<u8> = (0..1000_u32).map(|i| (i * 7 % 256) as u8).collect();
        let whole = digest(&bytes);
        for piece in [1, 3, 55, 56, 63, 64, 65, 119, 120, 500] {
            let mut hasher = Sha256::new();
            for chunk in bytes.chunks(piece) {
                hasher.update(chunk);
            }
            assert_eq!(hasher.finish(), whole, "조각 {piece}");
        }
        // 덩어리의 경계 근처의 모든 길이.
        for len in 0..200 {
            let (hash, count) = digest_reader(&bytes[..len]).unwrap();
            assert_eq!(hash, digest(&bytes[..len]));
            assert_eq!(count, len as u64);
        }
    }
}
