//! 20장: 우리가 만든 묶음을 llama.cpp가 만든 파일과 바이트 단위로 견준다.
//!
//! 실행: `cargo run --release -p badak_models --example ch20_files -- [모델 디렉터리]`
//!
//! 먼저 준비할 것:
//!   sh scripts/convert_to_gguf.sh [모델 디렉터리]
//!   sh scripts/quantize_gguf.sh [모델 디렉터리]

use std::collections::BTreeMap;

use badak_core::packed::{Format, Packed, Q4_K_BYTES, Q4_K_VALUES, dequantize_q4_k};
use badak_core::quant::{Granularity, fake_quantize};
use badak_formats::gguf::{GgmlType, Gguf};
use badak_models::llama::WeightFormat;
use badak_models::llama::model::LlamaModel;
use badak_models::models_dir;

/// llama.cpp가 형식에 붙인 번호(`ggml.h`의 `ggml_type`).
const Q4_0: GgmlType = GgmlType::Other(2);
const Q8_0: GgmlType = GgmlType::Other(8);
const Q4_K: GgmlType = GgmlType::Other(12);
const Q6_K: GgmlType = GgmlType::Other(14);

fn type_name(kind: GgmlType) -> String {
    match kind {
        Q4_0 => "Q4_0".to_string(),
        Q8_0 => "Q8_0".to_string(),
        Q4_K => "Q4_K".to_string(),
        Q6_K => "Q6_K".to_string(),
        other => format!("{other:?}"),
    }
}

// ANCHOR: names
/// 우리 모델의 가중치가 GGUF 파일에서 불리는 이름.
fn gguf_name(kind: &str, layer: Option<usize>) -> String {
    let part = match kind {
        "embedding" => return "token_embd.weight".to_string(),
        "q_proj" => "attn_q",
        "k_proj" => "attn_k",
        "v_proj" => "attn_v",
        "o_proj" => "attn_output",
        "gate_proj" => "ffn_gate",
        "up_proj" => "ffn_up",
        _ => "ffn_down",
    };
    format!("blk.{}.{part}.weight", layer.unwrap_or(0))
}

/// 질의와 키의 가중치는 파일에서 행의 순서가 다르다(13.3절). 헤드마다 앞
/// 절반과 뒤 절반의 행을 번갈아 놓는다. 그 순서로 옮겨 적는다.
fn in_file_order(kind: &str, values: Vec<f32>, cols: usize, head_dim: usize) -> Vec<f32> {
    if kind != "q_proj" && kind != "k_proj" {
        return values;
    }
    let half = head_dim / 2;
    let mut moved = Vec::with_capacity(values.len());
    for row in 0..values.len() / cols {
        let (head, within) = (row / head_dim, row % head_dim);
        let source = head * head_dim + within / 2 + (within % 2) * half;
        moved.extend_from_slice(&values[source * cols..(source + 1) * cols]);
    }
    moved
}
// ANCHOR_END: names

/// 제곱의 합 둘에서 SQNR(dB)을 낸다.
fn db(signal: f64, noise: f64) -> f64 {
    10.0 * (signal / noise).log10()
}

fn square_difference(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let name = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "llama32_1b".to_string());
    let directory = models_dir().join(&name);
    let model = LlamaModel::load_as(&directory, WeightFormat::F32)?;
    let head_dim = model.config.head_dim();
    // 파일의 순서로 옮겨 적은 `f32` 가중치들.
    let mut matrices = Vec::new();
    for (kind, layer, matrix) in model.matrices() {
        let (rows, cols) = (matrix.shape()[0], matrix.shape()[1]);
        let values = matrix.to_f32()?.to_vec::<f32>()?;
        let values = in_file_order(kind, values, cols, head_dim);
        matrices.push((gguf_name(kind, layer), rows, cols, values));
    }
    println!("모델 {name}: 행렬 모양의 가중치 {}개", matrices.len());

    // 1. 같은 값을 같은 방법으로 묶었으면 바이트가 같아야 한다.
    for (file, format, code) in [
        ("model_q8_0.gguf", Format::Q8_0, Q8_0),
        ("model_q4_0.gguf", Format::Q4_0, Q4_0),
    ] {
        let gguf = Gguf::open(directory.join(file))?;
        let mut kinds: BTreeMap<String, usize> = BTreeMap::new();
        for tensor in gguf.names() {
            let info = gguf.info(tensor).expect("이름이 있다");
            *kinds.entry(type_name(info.ggml_type)).or_default() += 1;
        }
        // ANCHOR: compare
        let (mut same, mut compared, mut bytes) = (0, 0, 0_usize);
        for (tensor, rows, cols, values) in &matrices {
            let info = gguf.info(tensor).ok_or("파일에 없는 텐서")?;
            if info.ggml_type != code {
                continue;
            }
            // 우리 손으로 묶는다.
            let ours = Packed::quantize(format, values, *rows, *cols)?;
            // 파일에서 같은 길이만큼 읽는다.
            let theirs = &gguf.bytes_from(tensor)?[..ours.bytes().len()];
            compared += 1;
            bytes += theirs.len();
            same += usize::from(ours.bytes() == theirs);
        }
        // ANCHOR_END: compare
        println!();
        println!("{file}");
        let kinds: Vec<String> = kinds.iter().map(|(k, n)| format!("{k} {n}개")).collect();
        println!("  저장 방식별 텐서의 수         {}", kinds.join(", "));
        println!("  견준 텐서                     {compared}개, {bytes}바이트");
        println!("  바이트가 모두 같은 텐서       {same}개");
    }

    // 2. 4비트 방법 셋의 오차. Q4_K_M 파일에서 Q4_K로 적힌 텐서들로 견준다.
    let gguf = Gguf::open(directory.join("model_q4_k_m.gguf"))?;
    let mut kinds: BTreeMap<String, usize> = BTreeMap::new();
    for tensor in gguf.names() {
        let info = gguf.info(tensor).expect("이름이 있다");
        *kinds.entry(type_name(info.ggml_type)).or_default() += 1;
    }
    let (mut signal, mut count, mut tensors) = (0.0_f64, 0_usize, 0);
    let mut noise = [0.0_f64; 4];
    for (tensor, rows, cols, values) in &matrices {
        let info = gguf.info(tensor).ok_or("파일에 없는 텐서")?;
        if info.ggml_type != Q4_K || values.len() % Q4_K_VALUES != 0 {
            continue;
        }
        tensors += 1;
        count += values.len();
        signal += values.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>();
        // 19장의 4비트: 정수 15가지, 눈금의 끝은 가장 큰 절댓값.
        let mut plain = values.clone();
        fake_quantize(&mut plain, 4, Granularity::Blocks(32));
        noise[0] += square_difference(values, &plain);
        // Q4_0과 Q8_0: 우리가 묶었다가 되돌린다.
        for (slot, format) in [(1, Format::Q4_0), (3, Format::Q8_0)] {
            let packed = Packed::quantize(format, values, *rows, *cols)?;
            noise[slot] += square_difference(values, &packed.dequantize_rows(0..*rows));
        }
        // Q4_K: llama.cpp가 묶은 것을 우리가 되돌린다.
        let bytes = &gguf.bytes_from(tensor)?[..values.len() / Q4_K_VALUES * Q4_K_BYTES];
        let mut restored = vec![0.0_f32; values.len()];
        dequantize_q4_k(bytes, &mut restored);
        noise[2] += square_difference(values, &restored);
    }
    println!();
    println!("model_q4_k_m.gguf");
    let kinds: Vec<String> = kinds.iter().map(|(k, n)| format!("{k} {n}개")).collect();
    println!("  저장 방식별 텐서의 수         {}", kinds.join(", "));
    println!("  Q4_K로 적힌 행렬              {tensors}개, 값 {count}개");
    println!();
    println!("그 행렬들을 방법마다 줄였을 때의 오차");
    println!("{:<34} {:>10} {:>12}", "방법", "값당 비트", "SQNR (dB)");
    for (label, bits, noise) in [
        ("19장의 4비트 (정수 15가지)", 4.5, noise[0]),
        ("Q4_0 (정수 16가지)", 4.5, noise[1]),
        ("Q4_K (스케일을 찾는다, 비대칭)", 4.5, noise[2]),
        ("Q8_0", 8.5, noise[3]),
    ] {
        println!("{label:<34} {bits:>10.1} {:>12.1}", db(signal, noise));
    }
    Ok(())
}
