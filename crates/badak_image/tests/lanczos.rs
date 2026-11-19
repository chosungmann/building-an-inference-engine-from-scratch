//! 란초스 필터로 크기를 바꾼 결과가 Pillow와 바이트까지 같은지 확인한다
//! (36장).
//!
//! 정답 파일이 없으면 건너뛴다. `sh scripts/download_ch36.sh`로 받는다.

use badak_formats::safetensors::SafeTensors;
use badak_image::{Image, lanczos};

/// `reference/ch36_make_reference.py`의 `synthetic`과 같은 사진.
fn synthetic(width: usize, height: usize) -> Image {
    let pixels = (0..height)
        .flat_map(|row| {
            (0..width).flat_map(move |column| {
                [0, 1, 2].map(|channel| {
                    ((row * 7 + column * 13 + channel * 31 + (row * column) % 251) % 256) as u8
                })
            })
        })
        .collect();
    Image {
        width,
        height,
        pixels,
    }
}

#[test]
fn lanczos_matches_pillow_byte_for_byte() {
    let path = badak_graph::models_dir().join("smolvlm/reference.safetensors");
    if !path.exists() {
        eprintln!("건너뜀: {}이 없습니다", path.display());
        return;
    }
    let reference = SafeTensors::open(path).unwrap();
    // 줄이기, 세로로 크게 줄이기, 늘리기, 한 축만.
    for (width, height, new_width, new_height) in [
        (517, 389, 341, 256),
        (300, 1000, 256, 200),
        (97, 61, 256, 161),
        (640, 480, 512, 480),
    ] {
        let name = format!("synthetic.{width}x{height}.{new_width}x{new_height}");
        let expected = reference.tensor(&name).unwrap().to_vec::<u8>().unwrap();
        let resized = lanczos::resize(&synthetic(width, height), new_width, new_height);
        let different = resized
            .pixels
            .iter()
            .zip(&expected)
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(different, 0, "{name}: {different}바이트가 다릅니다");
    }
}
