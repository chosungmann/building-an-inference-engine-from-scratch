//! 22장: 연산자 하나를 부를 때 메모리를 몇 번 할당하는지 센다.
//!
//! 실행: `cargo run --release -p badak_models --example ch22_per_operator`
//!
//! 토큰 하나(길이 2,048의 벡터 하나)를 처리할 때의 크기로 잰다.

use badak_bench::alloc::{Counting, Snapshot};
use badak_core::{Tensor, arena, ops};
use badak_models::llama::rope;
use badak_models::llama::{LlamaConfig, RopeScaling};

#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let width = 2048;
    let x = Tensor::from_vec(vec![0.5_f32; width], &[1, width])?;
    let y = Tensor::from_vec(vec![0.25_f32; width], &[1, width])?;
    let norm = Tensor::from_vec(vec![1.0_f32; width], &[width])?;
    let weight = Tensor::from_vec(vec![0.01_f32; 512 * width], &[512, width])?;
    let table = Tensor::from_vec(vec![0.01_f32; 64 * width], &[64, width])?;
    let ids = Tensor::from_vec(vec![3_i32], &[1])?;
    let config = LlamaConfig {
        vocab_size: 64,
        hidden_size: width,
        num_layers: 1,
        num_heads: 32,
        num_kv_heads: 8,
        intermediate_size: 8192,
        rms_norm_eps: 1e-5,
        rope_theta: 500_000.0,
        rope_scaling: None::<RopeScaling>,
        bos_token_id: None,
        eos_token_ids: Vec::new(),
    };
    let frequencies = rope::inverse_frequencies(&config);
    let heads = x.reshape(&[1, 32, 64])?;

    println!("연산자 하나를 부를 때의 할당 (입력은 길이 2,048의 벡터 하나)");
    println!(
        "{:<34} {:>8} {:>12} {:>14}",
        "연산", "할당", "바이트", "아레나 안에서"
    );
    // 한 번씩 미리 불러 처음에만 생기는 할당을 치른다. 스레드 풀의 스레드들이
    // 잠들 때 하는 할당이 끝나도록 잠깐 기다린다.
    ops::linear(&x, &weight, None)?;
    arena::scope(|| ops::linear(&x, &weight, None).map(drop))?;
    arena::scope(|| rope::apply(&heads, &frequencies, 5).map(drop))?;
    std::thread::sleep(std::time::Duration::from_millis(50));
    let (mut total, mut total_in_arena) = (0, 0);
    let mut row = |label: &str, per_block: usize, work: &dyn Fn() -> badak_core::Result<Tensor>| {
        let before = Snapshot::now();
        let result = work();
        let used = Snapshot::now().since(before);
        drop(result);
        // 같은 일을 아레나를 열어 놓고 한다.
        let in_arena = arena::scope(|| {
            let before = Snapshot::now();
            let result = work();
            let used = Snapshot::now().since(before);
            drop(result);
            used
        });
        println!(
            "{:<34} {:>8} {:>12} {:>14}",
            label, used.count, used.bytes, in_arena.count
        );
        total += used.count * per_block;
        total_in_arena += in_arena.count * per_block;
    };
    row("텐서 만들기 (from_vec)", 0, &|| {
        Tensor::from_vec(vec![0.0_f32; width], &[1, width])
    });
    row("모양 바꾸기 (reshape)", 3, &|| x.reshape(&[1, 32, 64]));
    row("더하기 (add)", 2, &|| ops::add(&x, &y));
    row("곱하기 (mul)", 1, &|| ops::mul(&x, &y));
    row("활성화 함수 (silu)", 1, &|| ops::silu(&x));
    row("정규화 (rms_norm)", 2, &|| {
        ops::rms_norm(&x, &norm, 1e-5)
    });
    row("선형 층 (linear)", 7, &|| ops::linear(&x, &weight, None));
    row("임베딩 (embedding)", 0, &|| ops::embedding(&table, &ids));
    row("회전 (rope::apply)", 2, &|| {
        rope::apply(&heads, &frequencies, 5).map_err(|_| badak_core::Error::NotContiguous)
    });
    println!();
    println!(
        "블록 하나에서 이 연산들이 하는 할당의 합 (횟수를 곱해서)   {total:>6} {total_in_arena:>14}"
    );
    println!(
        "블록 16개                                              {:>6} {:>14}",
        total * 16,
        total_in_arena * 16
    );
    Ok(())
}
