//! 6장: 모델 파일의 내용을 들여다본다.
//!
//! 실행: `cargo run --release -p badak_formats --example ch06_inspect -- 파일경로`
//!
//! 확장자가 `.safetensors`이거나 `.gguf`인 파일을 받는다. 텐서의 값은 읽지
//! 않고 머리 부분만 해석하므로, 수 기가바이트짜리 파일에도 바로 답한다.

use std::path::Path;

use badak_formats::gguf::{Gguf, MetadataValue};
use badak_formats::safetensors::SafeTensors;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("파일 경로를 주십시오")?;
    let size = std::fs::metadata(&path)?.len();
    let name = Path::new(&path)
        .file_name()
        .unwrap_or_default()
        .to_string_lossy();
    println!("{name} ({size}바이트)");

    match Path::new(&path).extension().and_then(|e| e.to_str()) {
        Some("safetensors") => inspect_safetensors(&path)?,
        Some("gguf") => inspect_gguf(&path)?,
        _ => return Err("확장자가 .safetensors 또는 .gguf여야 합니다".into()),
    }
    Ok(())
}

fn inspect_safetensors(path: &str) -> badak_formats::Result<()> {
    let file = SafeTensors::open(path)?;
    println!("형식: safetensors");
    println!();
    println!("메타데이터");
    for (key, value) in file.metadata() {
        println!("  {key:<24} {value:?}");
    }
    println!();
    println!("텐서");
    println!(
        "  {:<10} {:<6} {:<10} {:<12} 값",
        "이름", "자료형", "형상", "구간"
    );
    for name in file.names() {
        let info = file.info(name).expect("names가 준 이름이다");
        println!(
            "  {:<12} {:<9} {:<12} {:<14} {}",
            name,
            info.dtype,
            format!("{:?}", info.shape),
            format!("{}..{}", info.range.start, info.range.end),
            preview(file.tensor(name))
        );
    }
    Ok(())
}

fn inspect_gguf(path: &str) -> badak_formats::Result<()> {
    let file = Gguf::open(path)?;
    println!("형식: GGUF 버전 {}", file.version());
    println!();
    println!("메타데이터");
    for (key, value) in file.metadata() {
        println!("  {key:<24} {}", describe(value));
    }
    println!();
    println!("텐서");
    println!(
        "  {:<10} {:<6} {:<10} {:<8} 값",
        "이름", "종류", "형상", "위치"
    );
    for name in file.names() {
        let info = file.info(name).expect("names가 준 이름이다");
        println!(
            "  {:<12} {:<8} {:<12} {:<10} {}",
            name,
            format!("{:?}", info.ggml_type),
            format!("{:?}", info.shape),
            info.offset,
            preview(file.tensor(name))
        );
    }
    Ok(())
}

/// 메타데이터 값을 한 줄로 적는다. 긴 배열은 앞부분만 보인다.
fn describe(value: &MetadataValue) -> String {
    match value {
        MetadataValue::String(text) => format!("{text:?}"),
        MetadataValue::Array(items) => {
            let shown: Vec<String> = items.iter().take(4).map(describe).collect();
            let more = if items.len() > 4 { ", ..." } else { "" };
            format!("[{}{more}] ({}개)", shown.join(", "), items.len())
        }
        MetadataValue::Bool(flag) => flag.to_string(),
        other => other
            .as_u64()
            .map(|n| n.to_string())
            .or_else(|| other.as_f64().map(|n| n.to_string()))
            .unwrap_or_else(|| format!("{other:?}")),
    }
}

/// 텐서의 값을 짧게 적는다. 꺼낼 수 없는 텐서는 이유를 적는다.
fn preview(tensor: badak_formats::Result<badak_core::Tensor>) -> String {
    let tensor = match tensor {
        Ok(tensor) => tensor,
        Err(error) => return format!("({error})"),
    };
    let values: Vec<String> = match tensor.dtype() {
        badak_core::DType::F32 => tensor
            .to_vec::<f32>()
            .map(|v| v.iter().map(|x| x.to_string()).collect()),
        badak_core::DType::I32 => tensor
            .to_vec::<i32>()
            .map(|v| v.iter().map(|x| x.to_string()).collect()),
        badak_core::DType::U8 => tensor
            .to_vec::<u8>()
            .map(|v| v.iter().map(|x| x.to_string()).collect()),
    }
    .unwrap_or_default();
    let borrowed = if tensor.is_borrowed() {
        "빌림"
    } else {
        "복사"
    };
    format!("[{}] {borrowed}", values.join(", "))
}
