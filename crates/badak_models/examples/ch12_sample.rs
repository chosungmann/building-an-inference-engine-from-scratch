//! 12장: 샘플링으로 글을 이어 쓴다.
//!
//! 실행: `cargo run --release -p badak_models --example ch12_sample -- [모델 디렉터리] [글] [새 토큰 수] [설정…]`
//!
//! 설정은 `이름=값` 꼴로 여러 개 줄 수 있다.
//! 예: `... -- llama32_1b "Once upon a time" 40 temperature=0.8 top_k=40 top_p=0.95 seed=1`
//! 글 앞에 `chat:`을 붙이면 대화 형식으로 감싼다(11장).

use std::io::Write;

use badak_models::llama::cache::KvCache;
use badak_models::llama::generate::generate_sampled;
use badak_models::llama::model::LlamaModel;
use badak_models::models_dir;
use badak_models::sampling::{Sampler, SamplerConfig};
use badak_models::tokenizer::chat::{ChatFormat, Message};
use badak_models::tokenizer::{StreamDecoder, Tokenizer};

// ANCHOR: parse
/// `이름=값` 꼴의 인자들을 설정으로 바꾼다.
fn parse_config(options: &[String]) -> Result<SamplerConfig, Box<dyn std::error::Error>> {
    let mut config = SamplerConfig::default();
    for option in options {
        let (name, value) = option
            .split_once('=')
            .ok_or_else(|| format!("설정은 이름=값 꼴이어야 합니다: {option}"))?;
        match name {
            "temperature" => config.temperature = value.parse()?,
            "top_k" => config.top_k = value.parse()?,
            "top_p" => config.top_p = value.parse()?,
            "repetition_penalty" => config.repetition_penalty = value.parse()?,
            "temperature_last" => config.temperature_last = value.parse()?,
            "seed" => config.seed = value.parse()?,
            _ => return Err(format!("모르는 설정입니다: {name}").into()),
        }
    }
    Ok(config)
}
// ANCHOR_END: parse

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let name = arguments.first().map_or("llama32_1b", String::as_str);
    let prompt = arguments
        .get(1)
        .map_or("Once upon a time, there was a", String::as_str);
    let max_new_tokens: usize = arguments.get(2).map_or(Ok(40), |count| count.parse())?;
    let config = parse_config(arguments.get(3..).unwrap_or_default())?;

    let directory = models_dir().join(name);
    let model = LlamaModel::load(&directory)?;
    let tokenizer = Tokenizer::from_tokenizer_json(directory.join("tokenizer.json"))?;
    let text = match prompt.strip_prefix("chat:") {
        Some(question) => {
            // 틀의 뼈대만 쓴다. SmolLM2의 틀에 있는 기본 지시문은 넣지 않는다.
            // 그것까지 넣는 것이 `ChatTemplate`이다(47.3절).
            let format = ChatFormat::detect(&tokenizer);
            format.render(&[Message::new("user", question)], true)
        }
        None => prompt.to_string(),
    };
    let mut ids = tokenizer.prefix().to_vec();
    ids.extend(tokenizer.encode(&text, true));

    println!(
        "모델 {name}, 온도 {}{}, top-k {}, top-p {}, 반복 벌점 {}, 씨앗 {}",
        config.temperature,
        if config.temperature_last {
            " (맨 뒤에 적용)"
        } else {
            ""
        },
        config.top_k,
        config.top_p,
        config.repetition_penalty,
        config.seed
    );
    print!("{prompt}");
    let mut cache = KvCache::new(&model.config);
    let mut sampler = Sampler::new(config);
    let mut decoder = StreamDecoder::default();
    generate_sampled(
        &model,
        &ids,
        max_new_tokens,
        &mut cache,
        &mut sampler,
        |token| {
            print!("{}", decoder.push(&tokenizer, token));
            std::io::stdout().flush().ok();
        },
    )?;
    println!("{}", decoder.finish());
    Ok(())
}
