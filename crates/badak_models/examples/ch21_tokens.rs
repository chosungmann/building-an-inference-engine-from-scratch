//! 21장: 글 하나에서 토큰마다의 확률과 손실을 본다.
//!
//! 실행: `cargo run --release -p badak_models --example ch21_tokens -- [모델] [글]`
//!
//! 모델이 앞의 토큰들을 보고 다음 토큰에 준 확률을 자리마다 찍는다. 그
//! 확률들에서 평균 손실과 퍼플렉서티가 어떻게 나오는지 보인다.

use badak_models::llama::cache::KvCache;
use badak_models::llama::model::LlamaModel;
use badak_models::llama::perplexity::{Losses, token_loss};
use badak_models::models_dir;
use badak_models::tokenizer::Tokenizer;

const TEXT: &str = "The capital of France is Paris, and the capital of Italy is Rome.";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let name = arguments.next().unwrap_or_else(|| "llama32_1b".to_string());
    let text = arguments.next().unwrap_or_else(|| TEXT.to_string());
    let directory = models_dir().join(&name);
    let model = LlamaModel::load(&directory)?;
    let tokenizer = Tokenizer::from_tokenizer_json(directory.join("tokenizer.json"))?;
    let mut tokens = tokenizer.prefix().to_vec();
    tokens.extend(tokenizer.encode(&text, false));

    // ANCHOR: score
    // 글 전체를 한 번에 넣어 자리마다의 점수를 얻는다. `p`번째 줄이 `p + 1`번째
    // 토큰을 맞히는 점수다.
    let mut cache = KvCache::new(&model.config);
    let hidden = model.hidden_cached(&tokens, &mut cache)?;
    let logits = model.logits(&hidden)?;
    let logits = logits.as_slice::<f32>()?;
    let vocab = model.config.vocab_size;
    let mut losses = Losses::default();
    for position in 0..tokens.len() - 1 {
        let target = tokens[position + 1] as usize;
        let loss = token_loss(&logits[position * vocab..(position + 1) * vocab], target);
        losses.values.push(loss);
    }
    // ANCHOR_END: score

    println!("모델 {name}, 글 {text:?}");
    println!("{:>4} {:<12} {:>10} {:>8}", "자리", "토큰", "확률", "손실");
    for (index, loss) in losses.values.iter().enumerate() {
        println!(
            "{:>4} {:<12} {:>9.2}% {:>8.3}",
            index + 1,
            format!("{:?}", tokenizer.decode(&tokens[index + 1..index + 2])),
            (-loss).exp() * 100.0,
            loss
        );
    }
    println!();
    println!(
        "토큰 {}개의 평균 손실   {:.3}",
        losses.values.len(),
        losses.mean()
    );
    println!("퍼플렉서티             {:.2}", losses.perplexity());
    println!(
        "확률들의 기하 평균      {:.2}% (= 1 ÷ 퍼플렉서티)",
        100.0 / losses.perplexity()
    );
    Ok(())
}
