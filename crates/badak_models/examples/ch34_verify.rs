//! 34장: all-MiniLM-L6-v2의 토크나이저와 계산을 transformers의 정답과 견준다.
//!
//! 실행: `cargo run --release -p badak_models --example ch34_verify`

use badak_formats::safetensors::SafeTensors;
use badak_models::bert::{Batching, Bert, Plan};
use badak_models::models_dir;
use badak_models::tokenizer::TokenId;
use badak_models::tokenizer::wordpiece::WordPiece;

type Outcome<T> = Result<T, Box<dyn std::error::Error>>;

/// 모델이 받는 가장 긴 글(토큰). `sentence_bert_config.json`의 값이다.
const LIMIT: usize = 256;

fn gap(ours: &[f32], theirs: &[f32]) -> f32 {
    assert_eq!(ours.len(), theirs.len());
    ours.iter()
        .zip(theirs)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

/// 순위. 같은 값은 순위의 평균을 나눠 갖는다.
fn ranks(values: &[f32]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let mut ranks = vec![0.0; values.len()];
    let mut start = 0;
    while start < order.len() {
        let mut end = start;
        while end + 1 < order.len() && values[order[end + 1]] == values[order[start]] {
            end += 1;
        }
        for &index in &order[start..=end] {
            ranks[index] = (start + end) as f64 / 2.0;
        }
        start = end + 1;
    }
    ranks
}

/// 스피어먼 상관: 순위끼리의 피어슨 상관.
fn spearman(a: &[f32], b: &[f32]) -> f64 {
    let (a, b) = (ranks(a), ranks(b));
    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
    let (mean_a, mean_b) = (mean(&a), mean(&b));
    let (mut both, mut left, mut right) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(&b) {
        both += (x - mean_a) * (y - mean_b);
        left += (x - mean_a) * (x - mean_a);
        right += (y - mean_b) * (y - mean_b);
    }
    both / (left * right).sqrt()
}

fn main() -> Outcome<()> {
    let directory = models_dir().join("minilm");
    let reference = SafeTensors::open(directory.join("reference.safetensors"))?;
    let tokenizer = WordPiece::from_vocab(directory.join("vocab.txt"))?;
    let model = Bert::load(&directory)?;
    println!(
        "all-MiniLM-L6-v2: 조각 {}가지, 폭 {}, 블록 {}개",
        tokenizer.vocab_size(),
        model.width(),
        model.depth()
    );

    // 1. 토크나이저.
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference/ch34_sentences.txt"),
    )?;
    let sentences: Vec<&str> = text.lines().collect();
    println!("\n토크나이저: 글 {}개", sentences.len());
    println!("  {:>3} {:>5} {:<6} 글", "번호", "토큰", "같은가");
    let mut encoded: Vec<Vec<TokenId>> = Vec::new();
    let mut all_same = true;
    for (index, sentence) in sentences.iter().enumerate() {
        let ids = tokenizer.encode(sentence, LIMIT);
        let expected = reference.tensor(&format!("ids.{index}"))?.to_vec::<i32>()?;
        let same = ids.iter().map(|&id| id as i32).eq(expected.iter().copied());
        all_same &= same;
        let shown: String = sentence.chars().take(44).collect();
        println!(
            "  {index:>4} {:>6} {:<8} {shown:?}",
            ids.len(),
            if same { "같음" } else { "다름" }
        );
        encoded.push(ids);
    }
    let pieces: Vec<&str> = encoded[1]
        .iter()
        .filter_map(|&id| tokenizer.token_text(id))
        .collect();
    println!("  1번의 조각: {}", pieces.join(" "));

    // 2. 모델. 앞의 네 글은 블록마다.
    println!("\n단계마다 PyTorch와의 가장 큰 차이 (앞의 네 글 중 가장 큰 값)");
    let traced = 4;
    let mut steps = vec![0.0_f32; model.depth() + 1];
    for (index, ids) in encoded.iter().take(traced).enumerate() {
        let trace = model.forward(&[ids], Batching::Ragged, tokenizer.padding())?;
        let tensor = |key: String| -> Outcome<Vec<f32>> { Ok(reference.tensor(&key)?.to_vec()?) };
        steps[0] = steps[0].max(gap(
            &trace.embedded.to_vec::<f32>()?,
            &tensor(format!("embedded.{index}"))?,
        ));
        for (layer, output) in trace.layers.iter().enumerate() {
            steps[layer + 1] = steps[layer + 1].max(gap(
                &output.to_vec::<f32>()?,
                &tensor(format!("layer{layer}.{index}"))?,
            ));
        }
    }
    println!("  {:<12} {:.2e}", "임베딩", steps[0]);
    for (layer, step) in steps.iter().enumerate().skip(1) {
        println!("  블록 {:<8} {step:.2e}", layer - 1);
    }

    // 3. 글마다의 임베딩. 묶는 방법 셋과 빠른 길.
    let expected = reference.tensor("pooled")?.to_vec::<f32>()?;
    let expected_padded = reference.tensor("pooled_padded")?.to_vec::<f32>()?;
    let plan = |batch, batching| Plan {
        batch,
        batching,
        sorted: false,
        across: false,
    };
    let one = model.embed_many(&encoded, tokenizer.padding(), plan(1, Batching::Ragged))?;
    let padded = model.embed_many(&encoded, tokenizer.padding(), plan(16, Batching::Padded))?;
    let ragged = model.embed_many(&encoded, tokenizer.padding(), plan(16, Batching::Ragged))?;
    println!("\n임베딩 {}개의 가장 큰 차이", encoded.len());
    println!(
        "  {:<34} {:.2e}",
        "하나씩 ↔ PyTorch 하나씩",
        gap(&one, &expected)
    );
    println!(
        "  {:<34} {:.2e}",
        "채운 묶음 ↔ PyTorch 채운 묶음",
        gap(&padded, &expected_padded)
    );
    println!("  {:<34} {:.2e}", "채운 묶음 ↔ 하나씩", gap(&padded, &one));
    println!(
        "  {:<34} {:.2e}",
        "이어 붙인 묶음 ↔ 하나씩",
        gap(&ragged, &one)
    );
    let fast = Bert::load(&directory)?.fast()?;
    let quick = fast.embed_many(&encoded, tokenizer.padding(), plan(16, Batching::Ragged))?;
    println!("  {:<34} {:.2e}", "빠른 길 ↔ 하나씩", gap(&quick, &one));

    let worst = gap(&one, &expected)
        .max(gap(&padded, &expected_padded))
        .max(gap(&ragged, &one));
    if !all_same || worst > 1e-4 {
        return Err("정답과 다릅니다".into());
    }

    // 4. STS 벤치마크의 시험 쌍. 글 파일을 받지 못했으면 건너뛴다.
    let Ok(table) = std::fs::read_to_string(directory.join("stsbenchmark.tsv")) else {
        println!("\nSTS 벤치마크: 글 파일이 없어 건너뜁니다");
        return Ok(());
    };
    let (mut left, mut right, mut human) = (Vec::new(), Vec::new(), Vec::new());
    for line in table.lines().skip(1) {
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() >= 8 && fields[0] == "test" {
            left.push(tokenizer.encode(fields[6], LIMIT));
            right.push(tokenizer.encode(fields[7], LIMIT));
            human.push(fields[5].parse::<f32>()?);
        }
    }
    let bulk = Plan {
        batch: 32,
        batching: Batching::Ragged,
        sorted: true,
        across: true,
    };
    let a = model.embed_many(&left, tokenizer.padding(), bulk)?;
    let b = model.embed_many(&right, tokenizer.padding(), bulk)?;
    let width = model.width();
    let cosine: Vec<f32> = a
        .chunks_exact(width)
        .zip(b.chunks_exact(width))
        .map(|(x, y)| x.iter().zip(y).map(|(p, q)| p * q).sum())
        .collect();
    let theirs = reference.tensor("stsb.cosine")?.to_vec::<f32>()?;
    let their_score = reference.tensor("stsb.spearman")?.to_vec::<f32>()?[0];
    println!("\nSTS 벤치마크: 시험 쌍 {}개", human.len());
    println!(
        "  코사인 유사도의 가장 큰 차이   {:.2e}",
        gap(&cosine, &theirs)
    );
    println!(
        "  사람의 점수와의 스피어먼 상관  {:.4} (PyTorch {their_score:.4})",
        spearman(&cosine, &human)
    );

    if gap(&cosine, &theirs) > 1e-4 {
        return Err("정답과 다릅니다".into());
    }
    Ok(())
}
