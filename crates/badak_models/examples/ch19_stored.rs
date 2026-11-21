//! 19장: 16비트로 저장된 가중치를 펼치지 않고 그대로 꺼내 본다.
//!
//! 실행: `cargo run --release -p badak_models --example ch19_stored -- [모델 디렉터리]`

use std::time::Instant;

use badak_core::Tensor;
use badak_core::quant::Bf16;
use badak_models::llama::LlamaFiles;
use badak_models::models_dir;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let name = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "llama32_1b".to_string());
    let files = LlamaFiles::open(models_dir().join(&name))?;
    let weights = files.weights();
    let names: Vec<&str> = weights.names().collect();

    println!("모델 {name}: 텐서 {}개", names.len());
    println!(
        "{:<18} {:>8} {:>12} {:>12} {:>16}",
        "읽는 방법", "자료형", "시간 (ms)", "메모리 (GB)", "파일을 빌려 씀"
    );
    let report = |label: &str, tensors: &[Tensor], seconds: f64| {
        let bytes: usize = tensors.iter().map(Tensor::size_in_bytes).sum();
        let borrowed = tensors.iter().filter(|t| t.is_borrowed()).count();
        println!(
            "{:<18} {:>8} {:>12.1} {:>12.2} {:>16}",
            label,
            tensors[0].dtype().to_string(),
            seconds * 1e3,
            bytes as f64 / 1e9,
            format!("{borrowed} / {}", tensors.len())
        );
    };

    // ANCHOR: read
    // 지금까지의 방법: 16비트를 f32로 펼친다. 값을 모두 읽어 새로 적는다.
    let start = Instant::now();
    let widened: Vec<Tensor> = names
        .iter()
        .map(|name| weights.tensor(name))
        .collect::<Result<_, _>>()?;
    report("f32로 펼쳐 읽기", &widened, start.elapsed().as_secs_f64());

    // 새 방법: 저장된 그대로. 파일을 매핑한 메모리를 가리키기만 한다.
    let start = Instant::now();
    let stored: Vec<Tensor> = names
        .iter()
        .map(|name| weights.tensor_as_stored(name))
        .collect::<Result<_, _>>()?;
    report("저장된 그대로", &stored, start.elapsed().as_secs_f64());
    // ANCHOR_END: read

    // 두 방법으로 읽은 값이 같은지 모두 견준다.
    let mut count = 0_usize;
    let mut same = true;
    for (wide, half) in widened.iter().zip(&stored) {
        let wide = wide.as_slice::<f32>()?;
        let half = half.as_slice::<Bf16>()?;
        same &= wide.iter().zip(half).all(|(&a, b)| a == b.to_f32());
        count += wide.len();
    }
    println!();
    println!(
        "값 {count}개를 견준 결과: {}",
        if same {
            "모두 같다"
        } else {
            "다른 값이 있다"
        }
    );
    Ok(())
}
