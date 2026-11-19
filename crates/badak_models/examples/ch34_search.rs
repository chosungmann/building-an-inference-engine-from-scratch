//! 34장: 글들을 임베딩으로 바꿔 쌓아 두고, 질의와 뜻이 가까운 글을 찾는다.
//!
//! 실행: `cargo run --release -p badak_models --example ch34_search [질의...]`
//!
//! 찾을 글은 STS 벤치마크의 모든 글이다(겹치는 글은 하나만 남긴다).

use std::collections::HashSet;
use std::time::Instant;

use badak_models::bert::{Batching, Bert, Plan};
use badak_models::models_dir;
use badak_models::search::Index;
use badak_models::tokenizer::TokenId;
use badak_models::tokenizer::wordpiece::WordPiece;

type Outcome<T> = Result<T, Box<dyn std::error::Error>>;

const LIMIT: usize = 256;

fn main() -> Outcome<()> {
    let directory = models_dir().join("minilm");
    let tokenizer = WordPiece::from_vocab(directory.join("vocab.txt"))?;
    let model = Bert::load(&directory)?.fast()?;

    // 1. 글들을 읽는다.
    let table = std::fs::read_to_string(directory.join("stsbenchmark.tsv"))?;
    let mut seen = HashSet::new();
    let texts: Vec<&str> = table
        .lines()
        .skip(1)
        .flat_map(|line| line.split('\t').skip(6).take(2))
        .filter(|text| seen.insert(*text))
        .collect();

    // 2. 대량 처리: 모든 글을 임베딩으로 바꿔 색인에 넣는다.
    let start = Instant::now();
    let encoded: Vec<Vec<TokenId>> = texts
        .iter()
        .map(|text| tokenizer.encode(text, LIMIT))
        .collect();
    let tokenize_time = start.elapsed();
    let tokens: usize = encoded.iter().map(Vec::len).sum();
    let plan = Plan {
        batch: 32,
        batching: Batching::Ragged,
        sorted: true,
        across: true,
    };
    let start = Instant::now();
    let vectors = model.embed_many(&encoded, tokenizer.padding(), plan)?;
    let embed_time = start.elapsed();
    let mut index = Index::new(model.width());
    index.extend(&vectors);
    println!(
        "글 {}개(토큰 {tokens}개)를 색인에 넣었습니다. 토큰으로 {:.0}ms, 임베딩으로 {:.2}초(초당 {:.0}글), 색인 {:.1}MB",
        index.len(),
        tokenize_time.as_secs_f64() * 1e3,
        embed_time.as_secs_f64(),
        index.len() as f64 / embed_time.as_secs_f64(),
        (vectors.len() * 4) as f64 / 1e6
    );

    // 3. 저지연 조회: 질의 하나를 임베딩으로 바꿔 가까운 글을 찾는다.
    let given: Vec<String> = std::env::args().skip(1).collect();
    let queries = if given.is_empty() {
        vec![
            "How is the weather tomorrow?".to_owned(),
            "someone cooking dinner".to_owned(),
            "stock prices went down".to_owned(),
            "a kid riding a bike".to_owned(),
        ]
    } else {
        vec![given.join(" ")]
    };
    for query in &queries {
        let start = Instant::now();
        let ids = tokenizer.encode(query, LIMIT);
        let trace = model.forward(&[&ids], Batching::Ragged, tokenizer.padding())?;
        let embed_time = start.elapsed();
        let start = Instant::now();
        let hits = index.search(trace.pooled.as_slice::<f32>()?, 3);
        let search_time = start.elapsed();
        println!(
            "\n질의: {query:?} (토큰 {}개, 임베딩 {:.2}ms, 검색 {:.2}ms)",
            ids.len(),
            embed_time.as_secs_f64() * 1e3,
            search_time.as_secs_f64() * 1e3
        );
        for hit in hits {
            println!("  {:.3}  {}", hit.score, texts[hit.index]);
        }
    }
    Ok(())
}
