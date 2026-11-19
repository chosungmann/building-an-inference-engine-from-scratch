//! 36장: 사진 파일을 SmolVLM의 입력으로 바꿔 `f32` 값들을 그대로 파일에 쓴다.
//! Pillow로 만든 값과 견줄 때 쓴다(`reference/ch36_preprocess.py`).
//!
//! 실행: `cargo run --release -p badak_image --example ch36_squash -- 사진.jpg 출력.bin`

use badak_image::Image;
use badak_image::lanczos::squash;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let (Some(input), Some(output)) = (arguments.next(), arguments.next()) else {
        return Err("사진과 출력 파일의 경로를 주십시오".into());
    };
    let values = squash(&Image::open(input)?, 2048, 512);
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    std::fs::write(output, bytes)?;
    Ok(())
}
