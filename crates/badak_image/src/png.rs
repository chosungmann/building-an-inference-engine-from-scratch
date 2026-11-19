//! PNG 파일 쓰기 (35장).
//!
//! 모델이 만든 그림을 파일로 남기는 데 쓴다. 압축하지 않는다. PNG는
//! 픽셀을 zlib 형식으로 싸서 담는데, zlib에는 "압축하지 않고 그대로
//! 담는" 방법이 있어서 그것만 쓴다. 파일은 크지만 어느 프로그램으로든
//! 열린다.

use crate::Image;

/// PNG의 덩어리마다 붙는 검사 값(CRC-32).
fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0_u32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xedb8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// zlib의 끝에 붙는 검사 값(Adler-32).
fn adler32(bytes: &[u8]) -> u32 {
    let (mut a, mut b) = (1_u32, 0_u32);
    for &byte in bytes {
        a = (a + u32::from(byte)) % 65_521;
        b = (b + a) % 65_521;
    }
    (b << 16) | a
}

/// 덩어리 하나: 길이, 이름, 내용, 검사 값.
fn chunk(out: &mut Vec<u8>, name: &[u8; 4], body: &[u8]) {
    out.extend_from_slice(&(body.len() as u32).to_be_bytes());
    let start = out.len();
    out.extend_from_slice(name);
    out.extend_from_slice(body);
    let crc = crc32(&out[start..]);
    out.extend_from_slice(&crc.to_be_bytes());
}

// ANCHOR: encode
/// 사진을 PNG 파일의 바이트들로 바꾼다.
pub fn encode(image: &Image) -> Vec<u8> {
    // 줄마다 맨 앞에 "거르지 않음"을 뜻하는 0을 하나 붙인다.
    let stride = image.width * 3;
    let mut raw = Vec::with_capacity((stride + 1) * image.height);
    for row in image.pixels.chunks_exact(stride.max(1)) {
        raw.push(0);
        raw.extend_from_slice(row);
    }
    // zlib: 머리 둘, 그대로 담은 토막들(토막마다 65,535바이트까지), 검사 값.
    let mut packed = vec![0x78, 0x01];
    let mut blocks = raw.chunks(65_535).peekable();
    if blocks.peek().is_none() {
        packed.extend_from_slice(&[1, 0, 0, 0xff, 0xff]);
    }
    while let Some(block) = blocks.next() {
        let length = block.len() as u16;
        packed.push(u8::from(blocks.peek().is_none()));
        packed.extend_from_slice(&length.to_le_bytes());
        packed.extend_from_slice(&(!length).to_le_bytes());
        packed.extend_from_slice(block);
    }
    packed.extend_from_slice(&adler32(&raw).to_be_bytes());

    let mut out = vec![0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n'];
    let mut header = Vec::with_capacity(13);
    header.extend_from_slice(&(image.width as u32).to_be_bytes());
    header.extend_from_slice(&(image.height as u32).to_be_bytes());
    // 채널마다 8비트, 색의 종류 2(RGB), 압축·거르기·건너뛰기는 모두 0.
    header.extend_from_slice(&[8, 2, 0, 0, 0]);
    chunk(&mut out, b"IHDR", &header);
    chunk(&mut out, b"IDAT", &packed);
    chunk(&mut out, b"IEND", &[]);
    out
}
// ANCHOR_END: encode

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_values_match_known_answers() {
        // 두 검사 값의 표준 시험 값.
        assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
        assert_eq!(adler32(b"Wikipedia"), 0x11e6_0398);
        assert_eq!(crc32(b"IEND"), 0xae42_6082);
    }

    #[test]
    fn a_tiny_image_has_the_expected_layout() {
        let image = Image {
            width: 2,
            height: 1,
            pixels: vec![255, 0, 0, 0, 0, 255],
        };
        let bytes = encode(&image);
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        assert_eq!(&bytes[12..16], b"IHDR");
        assert_eq!(&bytes[16..24], [0, 0, 0, 2, 0, 0, 0, 1]);
        // IDAT의 내용: zlib 머리 2, 토막 머리 5, 줄 하나(0과 픽셀 6바이트),
        // 검사 값 4.
        let start = 8 + 25;
        assert_eq!(&bytes[start..start + 4], [0, 0, 0, 18]);
        assert_eq!(&bytes[start + 4..start + 8], b"IDAT");
        assert_eq!(
            &bytes[start + 8..start + 15],
            [0x78, 1, 1, 7, 0, 0xf8, 0xff]
        );
        assert_eq!(&bytes[start + 15..start + 22], [0, 255, 0, 0, 0, 0, 255]);
        assert_eq!(&bytes[bytes.len() - 8..bytes.len() - 4], b"IEND");
    }

    #[test]
    fn large_images_split_into_stored_blocks() {
        let image = Image {
            width: 200,
            height: 200,
            pixels: vec![7; 200 * 200 * 3],
        };
        let bytes = encode(&image);
        // 줄 200개 × 601바이트 = 120,200바이트. 토막 둘.
        let raw = 200 * 601;
        assert_eq!(bytes.len(), 8 + 25 + 12 + 2 + 2 * 5 + raw + 4 + 12);
    }
}
