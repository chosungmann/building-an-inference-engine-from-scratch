//! 41장: 문법에 묶은 생성과 추측 디코딩을 확인한다.
//!
//! 실행: `cargo run --release -p badak_models --example ch41_verify`
//!
//! `models/smollm2_135m`만 있으면 돈다. 큰 모델이 필요한 측정은
//! `ch41_speculate`에, PEFT와의 대조는 `ch41_lora`에 있다.

use std::sync::Arc;

use badak_formats::json;
use badak_models::constrain::{
    Constraint, Grammar, Matcher, Search, Vocabulary, generate_constrained,
};
use badak_models::llama::cache::KvCache;
use badak_models::llama::generate::generate;
use badak_models::llama::model::LlamaModel;
use badak_models::llama::speculative::{DraftModel, Lookup, Proposer, generate_speculative};
use badak_models::models_dir;
use badak_models::sampling::{Sampler, SamplerConfig};
use badak_models::tokenizer::chat::{ChatTemplate, Message};
use badak_models::tokenizer::{TokenId, Tokenizer};

type Outcome<T> = Result<T, Box<dyn std::error::Error>>;

const SCHEMA: &str = include_str!("../../../reference/ch41_schema.json");

fn check(name: &str, passed: bool) -> Outcome<()> {
    println!("{} {name}", if passed { "같음  " } else { "다름  " });
    if passed {
        Ok(())
    } else {
        Err(format!("{name}: 어긋났습니다").into())
    }
}

fn main() -> Outcome<()> {
    let directory = models_dir().join("smollm2_135m");
    if !directory.join("model.safetensors").exists() {
        println!("건너뜀: {}에 모델이 없습니다", directory.display());
        return Ok(());
    }
    let model = LlamaModel::load(&directory)?;
    let tokenizer = Tokenizer::from_tokenizer_json(directory.join("tokenizer.json"))?;
    let template = ChatTemplate::load(&directory, &tokenizer);
    let ask = |question: &str| -> Vec<TokenId> {
        template.encode(&tokenizer, &[Message::new("user", question)], true)
    };

    // 1. 문법에 묶어 쓴 답은 문법에 맞고, 찾는 방법을 바꿔도 같은 글이다.
    let grammar = Arc::new(Grammar::from_schema(&json::parse(SCHEMA)?)?);
    let vocabulary = Arc::new(Vocabulary::new(&tokenizer));
    let ids = ask("Describe the city of Seoul as a JSON object. Reply with JSON only.");
    let mut texts = Vec::new();
    for (search, temperature) in [
        (Search::Scan, 0.0),
        (Search::Trie, 0.0),
        (Search::Best, 0.0),
        (Search::Trie, 1.0),
    ] {
        let mut sampler = Sampler::new(SamplerConfig {
            temperature,
            seed: 7,
            ..SamplerConfig::default()
        });
        let mut constraint = Constraint::new(&vocabulary, &grammar, search);
        let (tokens, _) = generate_constrained(
            &model,
            &ids,
            120,
            &mut KvCache::new(&model.config),
            &mut sampler,
            &mut constraint,
            |_| {},
        )?;
        texts.push(tokenizer.decode(&tokens));
    }
    println!("점수가 가장 높은 토큰으로: {}", texts[0]);
    println!("확률로 뽑아서:             {}", texts[3]);
    check(
        "세 가지로 찾아 쓴 글이 같다",
        texts[0] == texts[1] && texts[1] == texts[2],
    )?;
    check(
        "묶어 쓴 글 넷이 모두 스키마에 맞는 JSON이다",
        texts
            .iter()
            .all(|text| Matcher::matches(&grammar, text) && json::parse(text).is_ok()),
    )?;
    let free = generate(&model, &ids, 120, &mut KvCache::new(&model.config), |_| {})?;
    println!(
        "묶지 않고 쓴 글이 스키마에 맞는가: {}",
        Matcher::matches(&grammar, tokenizer.decode(&free).trim())
    );

    // 2. 추측 디코딩은 혼자 쓴 것과 같은 토큰을 쓴다. 초안이 얼마나 맞든.
    println!();
    let passage =
        "The lighthouse at the end of the harbor was built in 1862 by a crew of forty men.";
    for (name, question) in [
        (
            "자유롭게 쓰기",
            "Write two sentences about the sea.".to_owned(),
        ),
        (
            "베껴 쓰기",
            format!("Repeat this sentence twice, exactly: {passage}"),
        ),
    ] {
        let prompt = ask(&question);
        let expected = generate(
            &model,
            &prompt,
            64,
            &mut KvCache::new(&model.config),
            |_| {},
        )?;
        // 초안을 쓰는 모델이 채점하는 모델과 같으면 모든 짐작이 맞아야 한다.
        let mut same_model = DraftModel::new(&model);
        let mut lookup = Lookup::default();
        let proposers: [(&str, &mut dyn Proposer, usize); 2] = [
            ("같은 모델의 초안", &mut same_model, 4),
            ("글에서 찾은 초안", &mut lookup, 8),
        ];
        for (label, proposer, lookahead) in proposers {
            let record = generate_speculative(
                &model,
                proposer,
                &prompt,
                64,
                lookahead,
                &mut KvCache::new(&model.config),
                |_| {},
            )?;
            println!(
                "        {name}, {label}: 순전파 {}번, 맞은 짐작 {}/{}",
                record.rounds, record.accepted, record.proposed
            );
            check(
                &format!("{name}, {label}: 혼자 쓴 것과 토큰이 같다"),
                record.tokens == expected,
            )?;
            if label == "같은 모델의 초안" {
                // 끝 토큰을 짐작한 것은 맞아도 세지 않는다. 거기서 글이 끝난다.
                check(
                    "        같은 모델의 초안은 끝 토큰 말고는 모두 맞는다",
                    record.accepted + 1 >= record.proposed,
                )?;
            }
        }
    }
    println!("\n모두 맞았습니다.");
    Ok(())
}
