//! 15장: 토큰 하나에 드는 곱셈과 메모리를 세고, 속도의 상한을 계산한다.
//!
//! 실행: `cargo run --release -p badak_models --example ch15_budget -- [모델 디렉터리] [계산 한계] [대역폭 1] [대역폭 N] [스레드 수]`
//!
//! 뒤의 네 인자는 기계의 한계다. 주지 않으면 이 책의 기준 기계에서 잰 값을
//! 쓴다(15.2절, 15.3절). 계산 한계는 스레드 하나가 1초에 하는 곱셈(십억 번),
//! 대역폭은 스레드 1개와 N개가 1초에 읽는 양(GB)이다.
//!
//! 가중치를 읽지 않고 `config.json`만 본다.

use badak_models::llama::LlamaConfig;
use badak_models::models_dir;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let name = arguments.next().unwrap_or_else(|| "llama32_1b".to_string());
    let mut number = |default: f64| arguments.next().map_or(Ok(default), |text| text.parse());
    // 기준 기계(Apple M4 Max)에서 잰 값.
    let peak_one = number(19.7)?;
    let bandwidth_one = number(83.0)?;
    let bandwidth_many = number(275.0)?;
    let threads = number(12.0)?;

    let c = LlamaConfig::from_file(models_dir().join(&name).join("config.json"))?;
    // ANCHOR: budget
    // 블록 하나의 선형 층들이 가진 가중치의 수. 토큰 하나가 블록 하나를 지날
    // 때 가중치마다 곱셈을 한 번씩 한다(9.11절).
    let kv_width = c.num_kv_heads * c.head_dim();
    let block = 2 * c.hidden_size * c.hidden_size
        + 2 * kv_width * c.hidden_size
        + 3 * c.intermediate_size * c.hidden_size;
    let blocks = (block * c.num_layers) as f64;
    // 출력 층은 임베딩 표 전체와 곱한다(10.2절).
    let output = (c.vocab_size * c.hidden_size) as f64;

    // 프롬프트의 토큰은 블록만 지난다. 새로 만드는 토큰은 출력 층까지 지난다.
    let prompt_multiplies = blocks;
    let generation_multiplies = blocks + output;
    // 새 토큰 하나를 만들려면 이 가중치를 모두 한 번씩 읽어야 한다. f32로 4바이트씩.
    let generation_bytes = 4.0 * (blocks + output);
    // ANCHOR_END: budget

    println!("모델 {name}");
    println!(
        "{:<44} {:>10.3} 십억 번",
        "프롬프트의 토큰 하나에 드는 곱셈",
        prompt_multiplies / 1e9
    );
    println!(
        "{:<44} {:>10.3} 십억 번",
        "새 토큰 하나에 드는 곱셈",
        generation_multiplies / 1e9
    );
    println!(
        "{:<44} {:>10.3} GB",
        "새 토큰 하나를 만들 때 읽는 가중치 (f32)",
        generation_bytes / 1e9
    );
    println!(
        "{:<44} {:>10.3} 곱셈/바이트",
        "토큰 생성의 산술 강도",
        generation_multiplies / generation_bytes
    );

    // ANCHOR: bounds
    println!();
    println!(
        "{:<12} {:>14} {:>14} {:>14} {:>14}",
        "스레드", "계산의 한계", "메모리의 한계", "생성의 상한", "프롬프트 상한"
    );
    for (count, bandwidth) in [(1.0, bandwidth_one), (threads, bandwidth_many)] {
        // 계산의 한계: 스레드마다 `peak_one`씩 곱셈을 한다.
        let compute = peak_one * 1e9 * count;
        // 메모리의 한계: 읽어 온 바이트만큼만 곱셈을 할 수 있다. 4바이트에 한 번.
        let memory = bandwidth * 1e9 / 4.0;
        // 토큰 생성은 둘 가운데 작은 쪽에 묶인다.
        let generation = compute.min(memory) / generation_multiplies;
        // 프롬프트 처리는 가중치 하나를 여러 토큰에 다시 쓰므로 계산에만 묶인다.
        let prompt = compute / prompt_multiplies;
        println!(
            "{:<12} {:>11.1} 십억 {:>11.1} 십억 {:>9.1} 토큰/초 {:>9.1} 토큰/초",
            count,
            compute / 1e9,
            memory / 1e9,
            generation,
            prompt
        );
    }
    // ANCHOR_END: bounds
    Ok(())
}
