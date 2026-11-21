//! 41장: 모델이 정해 둔 꼴의 JSON만 쓰게 한다.
//!
//! 실행: `cargo run --release -p badak_models --example ch41_json -- \
//!        [--model 이름] [--out 파일] [--cities 개]`
//!
//! 도시마다 같은 질문을 두 번 한다. 한 번은 그냥 쓰게 하고, 한 번은
//! 문법에 묶어 쓰게 한다. `--out`을 주면 답들을 한 줄에 하나씩 적는다.
//! `reference/ch41_validate.py`가 그 파일을 Python의 `jsonschema`로 다시
//! 검사한다.

use std::sync::Arc;
use std::time::Instant;

use badak_formats::json;
use badak_models::constrain::{
    Constraint, Grammar, Matcher, Search, Vocabulary, generate_constrained,
};
use badak_models::llama::cache::KvCache;
use badak_models::llama::generate::generate;
use badak_models::llama::model::LlamaModel;
use badak_models::models_dir;
use badak_models::sampling::{Sampler, SamplerConfig};
use badak_models::tokenizer::chat::{ChatTemplate, Message};
use badak_models::tokenizer::{TokenId, Tokenizer};

type Outcome<T> = Result<T, Box<dyn std::error::Error>>;

/// 답이 따라야 하는 꼴. Python 쪽의 검사(`reference/ch41_validate.py`)와
/// 같은 파일을 읽는다.
pub const SCHEMA: &str = include_str!("../../../reference/ch41_schema.json");

const CITIES: [&str; 20] = [
    "Paris",
    "Tokyo",
    "Nairobi",
    "Lima",
    "Sydney",
    "Toronto",
    "Cairo",
    "Seoul",
    "Berlin",
    "Mumbai",
    "Lagos",
    "Buenos Aires",
    "Rome",
    "Bangkok",
    "Mexico City",
    "Oslo",
    "Istanbul",
    "Auckland",
    "Chicago",
    "Cape Town",
];
const LIMIT: usize = 96;

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn main() -> Outcome<()> {
    let (mut model, mut out, mut cities) = ("smollm2_135m".to_owned(), None, CITIES.len());
    let mut arguments = std::env::args().skip(1);
    while let Some(flag) = arguments.next() {
        let mut value = || arguments.next().ok_or(format!("{flag} 뒤에 값이 없습니다"));
        match flag.as_str() {
            "--model" => model = value()?,
            "--out" => out = Some(value()?),
            "--cities" => cities = value()?.parse()?,
            _ => return Err(format!("모르는 인자입니다: {flag}").into()),
        }
    }
    let directory = models_dir().join(&model);
    if !directory.join("model.safetensors").exists() {
        println!("건너뜀: {}에 모델이 없습니다", directory.display());
        return Ok(());
    }
    let llama = LlamaModel::load(&directory)?;
    let tokenizer = Tokenizer::from_tokenizer_json(directory.join("tokenizer.json"))?;
    let format = ChatTemplate::load(&directory, &tokenizer);
    let grammar = Arc::new(Grammar::from_schema(&json::parse(SCHEMA)?)?);
    let start = Instant::now();
    let vocabulary = Arc::new(Vocabulary::new(&tokenizer));
    println!(
        "모델 {model}. 단어 사전 {}개를 마디 {}개짜리 나무로 엮는 데 {:.0}ms\n",
        tokenizer.vocab_size(),
        vocabulary.branches(),
        start.elapsed().as_secs_f64() * 1e3
    );

    let prompt = |city: &str| -> Vec<TokenId> {
        let question = format!(
            "Describe the city of {city} as a JSON object with the fields city, country, \
             continent, is_capital, population_millions and languages. Reply with JSON only."
        );
        format.encode(&tokenizer, &[Message::new("user", &question)], true)
    };
    let greedy = || {
        Sampler::new(SamplerConfig {
            temperature: 0.0,
            ..SamplerConfig::default()
        })
    };

    // 1. 도시마다: 그냥 쓴 답과 문법에 묶어 쓴 답.
    let (mut free_valid, mut bound_valid, mut lines) = (0, 0, Vec::new());
    let (mut step_ms, mut search_ms) = (Vec::new(), Vec::new());
    for (index, city) in CITIES.iter().take(cities).enumerate() {
        let ids = prompt(city);
        let free = generate(
            &llama,
            &ids,
            LIMIT,
            &mut KvCache::new(&llama.config),
            |_| {},
        )?;
        let free = tokenizer.decode(&free);
        // 그냥 쓴 답은 너그럽게 본다. 글 전체가 JSON으로 읽히고 그 값이
        // 스키마에 맞으면 된다. 빈칸과 줄바꿈은 보지 않는다.
        let free_ok = json::parse(free.trim()).is_ok_and(|value| grammar.validates(&value));

        let mut constraint = Constraint::new(&vocabulary, &grammar, Search::Trie);
        let start = Instant::now();
        let (bound, searching) = generate_constrained(
            &llama,
            &ids,
            LIMIT,
            &mut KvCache::new(&llama.config),
            &mut greedy(),
            &mut constraint,
            |_| {},
        )?;
        let elapsed = start.elapsed().as_secs_f64();
        let text = tokenizer.decode(&bound);
        let bound_ok = Matcher::matches(&grammar, &text) && json::parse(&text).is_ok();
        step_ms.push(elapsed * 1e3 / bound.len().max(1) as f64);
        search_ms.push(searching * 1e3 / bound.len().max(1) as f64);
        free_valid += usize::from(free_ok);
        bound_valid += usize::from(bound_ok);
        if index < 3 {
            println!("{city}");
            println!("  그냥:   {:?}", free.trim());
            println!("  묶어서: {text}");
        }
        lines.push(format!(
            "{{\"city\":{},\"free\":{},\"bound\":{}}}",
            json::quote(city),
            json::quote(free.trim()),
            json::quote(&text)
        ));
    }
    println!(
        "\n도시 {cities}곳 가운데 꼴에 맞는 답: 그냥 쓰면 {free_valid}곳, 묶어 쓰면 {bound_valid}곳"
    );
    println!(
        "묶어 쓸 때 토큰 하나에 {:.2}ms, 그 가운데 올 수 있는 토큰을 찾는 데 {:.2}ms (가운데 값)",
        median(step_ms),
        median(search_ms)
    );
    if let Some(path) = out {
        std::fs::write(&path, lines.join("\n") + "\n")?;
        println!("{path} 기록");
    }

    // 2. 올 수 있는 토큰을 찾는 세 방법. 같은 글을 쓰는지와 걸음마다의 시간.
    println!(
        "\n찾는 방법       쓴 토큰  찾는 시간(ms/토큰)  올 수 있던 토큰(가운데 값, 가장 많을 때)"
    );
    let ids = prompt(CITIES[0]);
    let mut texts = Vec::new();
    for (name, search) in [
        ("모두 하나씩", Search::Scan),
        ("나무를 따라", Search::Trie),
        ("점수 높은 것부터", Search::Best),
    ] {
        let mut constraint = Constraint::new(&vocabulary, &grammar, search);
        let (tokens, searching) = generate_constrained(
            &llama,
            &ids,
            LIMIT,
            &mut KvCache::new(&llama.config),
            &mut greedy(),
            &mut constraint,
            |_| {},
        )?;
        let allowed: Vec<f64> = constraint
            .allowed
            .iter()
            .map(|&count| count as f64)
            .collect();
        println!(
            "{name:<14} {:>4}     {:>8.3}            {:>6.0}, {:>6.0}",
            tokens.len(),
            searching * 1e3 / tokens.len().max(1) as f64,
            median(allowed.clone()),
            allowed.iter().copied().fold(0.0, f64::max),
        );
        texts.push(tokenizer.decode(&tokens));
    }
    println!(
        "세 방법이 쓴 글이 같은가: {}",
        if texts.iter().all(|text| text == &texts[0]) {
            "같음"
        } else {
            "다름"
        }
    );
    Ok(())
}
