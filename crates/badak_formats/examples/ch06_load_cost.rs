//! 6장: 파일을 통째로 읽는 것과 메모리에 매핑하는 것의 비용을 잰다.
//!
//! 실행: `cargo run --release -p badak_formats --example ch06_load_cost`
//!
//! 임시 디렉터리에 1GiB짜리 safetensors 파일을 만들어 실험하고 지운다.
//! 초시계 수준의 거친 측정이다.

use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

use badak_core::Tensor;
use badak_core::linalg::sum_pairwise;
use badak_formats::safetensors::{self, SafeTensors};

const TENSORS: usize = 8;
const ELEMENTS: usize = 32 * 1024 * 1024; // 텐서 하나에 128MiB

/// 한 번 실행한 시간을 밀리초로 돌려준다.
fn time<R>(f: impl FnOnce() -> R) -> (R, f64) {
    let start = Instant::now();
    let result = f();
    (result, start.elapsed().as_secs_f64() * 1e3)
}

/// 세 번 실행해 가장 빠른 시간을 밀리초로 돌려준다.
fn best_of_three<R>(mut f: impl FnMut() -> R) -> f64 {
    (0..3)
        .map(|_| time(|| black_box(f())).1)
        .fold(f64::INFINITY, f64::min)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::temp_dir().join(format!("badak_ch06_{}.safetensors", std::process::id()));

    // 실험용 파일을 만든다.
    let names: Vec<String> = (0..TENSORS).map(|i| format!("layer{i}.weight")).collect();
    let tensors: Vec<Tensor> = (0..TENSORS)
        .map(|i| Tensor::from_vec(vec![i as f32 + 0.5; ELEMENTS], &[ELEMENTS]))
        .collect::<Result<_, _>>()?;
    let pairs: Vec<(&str, &Tensor)> = names.iter().map(String::as_str).zip(&tensors).collect();
    safetensors::write(&path, &pairs, &[])?;
    drop(tensors);
    let size = std::fs::metadata(&path)?.len();
    println!(
        "파일: 텐서 {TENSORS}개, 각 {}MiB, 전체 {}MiB",
        ELEMENTS * 4 / (1024 * 1024),
        size / (1024 * 1024)
    );
    println!("(방금 쓴 파일이라 운영체제의 파일 캐시에 올라와 있는 상태다)");

    println!();
    println!("== 여는 데 걸리는 시간 ==");
    let (bytes, read_all) = time(|| std::fs::read(&path));
    let bytes = Arc::new(bytes?);
    let (from_read, parse_read) = time(|| SafeTensors::parse(bytes.clone()));
    let from_read = from_read?;
    let (mapped, open_mmap) = time(|| SafeTensors::open(&path));
    let mapped = mapped?;
    println!(
        "{:<44} {:>10.2} ms",
        "통째로 읽기 (std::fs::read)", read_all
    );
    println!("{:<44} {:>10.2} ms", "  읽은 뒤 헤더 해석", parse_read);
    println!("{:<44} {:>10.2} ms", "메모리에 매핑하고 헤더 해석", open_mmap);

    println!();
    println!("== 텐서 하나를 꺼내는 시간 ==");
    let (tensor, get_one) = time(|| mapped.tensor("layer3.weight"));
    let tensor = tensor?;
    println!("{:<44} {:>10.4} ms", "매핑한 파일에서 텐서 꺼내기", get_one);
    println!(
        "{:<44} {:>10}",
        "  복사 없이 빌렸는가",
        tensor.is_borrowed()
    );

    println!();
    println!("== 값을 실제로 읽는 시간 (모든 원소의 합) ==");
    let (sum, first_touch) = time(|| sum_pairwise(tensor.as_slice::<f32>().unwrap()));
    let second_touch = best_of_three(|| sum_pairwise(tensor.as_slice::<f32>().unwrap()));
    println!(
        "{:<44} {:>10.2} ms",
        "매핑한 텐서 하나, 처음 읽을 때", first_touch
    );
    println!(
        "{:<44} {:>10.2} ms",
        "매핑한 텐서 하나, 다시 읽을 때", second_touch
    );
    let copied = from_read.tensor("layer3.weight")?;
    let in_memory = best_of_three(|| sum_pairwise(copied.as_slice::<f32>().unwrap()));
    println!(
        "{:<44} {:>10.2} ms",
        "통째로 읽은 쪽의 같은 텐서, 다시 읽을 때", in_memory
    );
    assert_eq!(sum, 3.5 * ELEMENTS as f32);

    drop((tensor, copied, mapped, from_read));
    std::fs::remove_file(&path)?;
    Ok(())
}
