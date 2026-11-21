//! 34장: 임베딩의 두 가지 부하. 많은 글의 처리량과 글 하나의 지연 시간.
//!
//! 실행:
//!   `ch34_bench bulk <배치 크기> <padded|ragged> <given|sorted> <inside|across> [fast]`
//!   `ch34_bench latency [fast]`
//!   `ch34_bench tokenize`
//!   `ch34_bench search [개수]`
//!
//! 스레드의 수는 환경 변수 `BADAK_THREADS`로 정한다. 글은 STS 벤치마크
//! 파일의 앞에서부터 `BULK`개를 쓴다.

use std::hint::black_box;
use std::time::Instant;

use badak_bench::stats::Summary;
use badak_bench::timing::{Plan as Timing, measure};
use badak_core::threads;
use badak_models::bert::{Batching, Bert, Plan, layout};
use badak_models::models_dir;
use badak_models::search::Index;
use badak_models::tokenizer::TokenId;
use badak_models::tokenizer::wordpiece::WordPiece;

type Outcome<T> = Result<T, Box<dyn std::error::Error>>;

const LIMIT: usize = 256;
/// 처리량을 잴 때 쓰는 글의 수.
const BULK: usize = 4096;

/// STS 벤치마크의 글들. 줄마다 두 글이 있다.
fn sentences(count: usize) -> Outcome<Vec<String>> {
    let table = std::fs::read_to_string(models_dir().join("minilm/stsbenchmark.tsv"))?;
    Ok(table
        .lines()
        .skip(1)
        .flat_map(|line| line.split('\t').skip(6).take(2))
        .take(count)
        .map(str::to_owned)
        .collect())
}

fn main() -> Outcome<()> {
    let directory = models_dir().join("minilm");
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let has = |word: &str| arguments.iter().any(|argument| argument == word);
    let tokenizer = WordPiece::from_vocab(directory.join("vocab.txt"))?;
    let model = Bert::load(&directory)?;
    let model = if has("fast") { model.fast()? } else { model };
    let path = if has("fast") { "빠른 길" } else { "기본" };
    let workers = threads::global().pool().threads();
    let padding = tokenizer.padding();

    match arguments.first().map(String::as_str) {
        Some("bulk") => {
            let texts = sentences(BULK)?;
            let encoded: Vec<Vec<TokenId>> = texts
                .iter()
                .map(|text| tokenizer.encode(text, LIMIT))
                .collect();
            let plan = Plan {
                batch: arguments
                    .get(1)
                    .and_then(|text| text.parse().ok())
                    .unwrap_or(32),
                batching: if has("padded") {
                    Batching::Padded
                } else {
                    Batching::Ragged
                },
                sorted: has("sorted"),
                across: has("across"),
            };
            // 배치들이 차지하는 줄의 합. 채운 자리가 얼마나 되는지 보인다.
            let mut order: Vec<usize> = (0..encoded.len()).collect();
            if plan.sorted {
                order.sort_by_key(|&index| std::cmp::Reverse(encoded[index].len()));
            }
            let real: usize = encoded.iter().map(Vec::len).sum();
            let rows: usize = order
                .chunks(plan.batch)
                .map(|batch| {
                    let lengths: Vec<usize> =
                        batch.iter().map(|&index| encoded[index].len()).collect();
                    layout(&lengths, plan.batching)
                        .iter()
                        .map(|span| span.rows)
                        .sum::<usize>()
                })
                .sum();
            // 한 번 돌려 데운 뒤 세 번 재서 가장 빠른 것을 쓴다.
            black_box(model.embed_many(&encoded[..64.min(encoded.len())], padding, plan)?);
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                let start = Instant::now();
                black_box(model.embed_many(&encoded, padding, plan)?);
                best = best.min(start.elapsed().as_secs_f64());
            }
            println!(
                "bulk {path} 스레드 {workers:>2} 배치 {:>3} {:<6} {:<6} {:<6} 줄/토큰 {:.2} 초당 {:>8.1}글 ({:.2}초)",
                plan.batch,
                if has("padded") { "padded" } else { "ragged" },
                if plan.sorted { "sorted" } else { "given" },
                if plan.across { "across" } else { "inside" },
                rows as f64 / real as f64,
                encoded.len() as f64 / best,
                best
            );
        }
        Some("latency") => {
            // 길이가 다른 글 하나씩. 긴 글을 잘라서 만든다.
            let text = std::fs::read_to_string(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../reference/ch34_sentences.txt"),
            )?;
            let long = text.lines().last().ok_or("글이 없습니다")?;
            for length in [8, 16, 32, 64, 128, 256] {
                let ids = tokenizer.encode(long, length);
                let timing = Timing {
                    warmup: 5,
                    repetitions: 101,
                };
                let samples = measure(timing, || {
                    black_box(
                        model
                            .forward(&[&ids], Batching::Ragged, padding)
                            .expect("실행")
                            .pooled,
                    );
                });
                let summary = Summary::new(&samples);
                println!(
                    "latency {path} 스레드 {workers:>2} 토큰 {:>3} 중앙값 {:>7.3}ms 최소 {:>7.3}ms",
                    ids.len(),
                    summary.median * 1e3,
                    summary.min * 1e3
                );
            }
        }
        Some("tokenize") => {
            let texts = sentences(usize::MAX)?;
            let mut best = f64::INFINITY;
            let mut tokens = 0;
            for _ in 0..5 {
                let start = Instant::now();
                tokens = texts
                    .iter()
                    .map(|text| black_box(tokenizer.encode(text, LIMIT)).len())
                    .sum();
                best = best.min(start.elapsed().as_secs_f64());
            }
            let bytes: usize = texts.iter().map(String::len).sum();
            println!(
                "tokenize 글 {}개 {}바이트 토큰 {tokens}개 {:.1}ms (글 하나에 {:.2}µs, 초당 {:.1}MB)",
                texts.len(),
                bytes,
                best * 1e3,
                best * 1e6 / texts.len() as f64,
                bytes as f64 / best / 1e6
            );
        }
        Some("search") => {
            // 임베딩을 한 번 만들어 색인에 넣고, 원하는 개수까지 되풀이해
            // 채운다. 견주는 시간은 값이 아니라 개수에 달려 있다.
            let wanted: usize = arguments
                .get(1)
                .and_then(|text| text.parse().ok())
                .unwrap_or(0);
            let texts = sentences(BULK)?;
            let encoded: Vec<Vec<TokenId>> = texts
                .iter()
                .map(|text| tokenizer.encode(text, LIMIT))
                .collect();
            let plan = Plan {
                batch: 32,
                batching: Batching::Ragged,
                sorted: true,
                across: true,
            };
            let vectors = model.embed_many(&encoded, padding, plan)?;
            let mut index = Index::new(model.width());
            while index.len() < wanted.max(BULK) {
                index.extend(&vectors);
            }
            let query = &vectors[..model.width()];
            let timing = Timing {
                warmup: 3,
                repetitions: 31,
            };
            let scores = Summary::new(&measure(timing, || {
                black_box(index.scores(query));
            }));
            let search = Summary::new(&measure(timing, || {
                black_box(index.search(query, 10));
            }));
            println!(
                "search 스레드 {workers:>2} 벡터 {:>8}개 내적 {:>8.3}ms 고르기까지 {:>8.3}ms",
                index.len(),
                scores.median * 1e3,
                search.median * 1e3
            );
        }
        _ => return Err("bulk, latency, tokenize, search 중 하나를 고르십시오".into()),
    }
    Ok(())
}
