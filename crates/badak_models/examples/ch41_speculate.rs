//! 41장: 다음 토큰들을 미리 짐작해 큰 모델의 순전파를 아낀다.
//!
//! 실행: `cargo run --release -p badak_models --example ch41_speculate -- \
//!        [--target 이름] [--draft 이름] [--tokens 개] [--repeats 번]`
//!
//! 큰 모델(기본 `smollm2_1_7b`)이 혼자 쓴 글을 기준으로 삼고, 작은
//! 모델(`smollm2_135m`)이 짐작하게 한 것과 지금까지의 글에서 찾아 짐작하게
//! 한 것을 견준다. 쓴 토큰이 기준과 같은지와 속도를 본다.

use std::time::Instant;

use badak_models::llama::cache::KvCache;
use badak_models::llama::generate::generate;
use badak_models::llama::model::LlamaModel;
use badak_models::llama::speculative::{DraftModel, Lookup, Proposer, generate_speculative};
use badak_models::models_dir;
use badak_models::tokenizer::chat::{ChatTemplate, Message};
use badak_models::tokenizer::{TokenId, Tokenizer};

type Outcome<T> = Result<T, Box<dyn std::error::Error>>;

const PASSAGE: &str = "The lighthouse at the end of the harbor was built in 1862 by a crew of \
    forty men who carried every stone up the cliff by hand. Its lamp burned whale oil until \
    1901, when the keeper, a retired sailor named Tomas Arden, installed a kerosene burner \
    that could be seen from eleven miles away. Arden kept a daily log of the weather, the \
    ships that passed, and the birds that struck the glass at night.";

/// (이름, 질문).
fn tasks() -> Vec<(&'static str, String)> {
    vec![
        (
            "자유롭게 쓰기",
            "Write a short story about a cat who learns to sail.".to_owned(),
        ),
        (
            "묻고 답하기",
            "List the planets of the solar system in order and say one thing about each."
                .to_owned(),
        ),
        (
            "고쳐 쓰기",
            format!(
                "Copy the following passage exactly, changing only \"whale oil\" to \"lamp oil\".\n\n{PASSAGE}"
            ),
        ),
    ]
}

/// 한 가지 일을 재는 데 드는 것들.
struct Setup<'a> {
    big: &'a LlamaModel,
    prompt: &'a [TokenId],
    limit: usize,
    repeats: usize,
    /// 큰 모델이 혼자 쓴 토큰들.
    expected: &'a [TokenId],
    /// 그때 걸린 시간(초).
    alone: f64,
}

impl Setup<'_> {
    /// `make`가 만든 것으로 짐작하게 해서 잰다. 쓴 토큰이 기준과 같은지를
    /// 돌려준다.
    fn run<'p>(
        &self,
        label: &str,
        lookahead: usize,
        make: &dyn Fn() -> Box<dyn Proposer + 'p>,
    ) -> Outcome<bool> {
        let (mut times, mut last) = (Vec::new(), None);
        for _ in 0..self.repeats {
            let mut cache = KvCache::new(&self.big.config);
            self.big
                .forward_cached(&self.prompt[..self.prompt.len() - 1], &mut cache)?;
            let mut proposer = make();
            let start = Instant::now();
            let record = generate_speculative(
                self.big,
                proposer.as_mut(),
                self.prompt,
                self.limit,
                lookahead,
                &mut cache,
                |_| {},
            )?;
            times.push(start.elapsed().as_secs_f64());
            last = Some(record);
        }
        let (record, time) = (last.ok_or("측정이 없습니다")?, median(times));
        let same = record.tokens == self.expected;
        println!(
            "  {label} 미리 {lookahead:>2}개   {:>6.1} 토큰/초  {:.2}배  큰 모델의 순전파 {:>3}번  맞은 짐작 {:>3}/{:<3} ({:>3.0}%)  {}",
            record.tokens.len() as f64 / time,
            self.alone / time,
            record.rounds,
            record.accepted,
            record.proposed,
            100.0 * record.accepted as f64 / record.proposed.max(1) as f64,
            if same { "같음" } else { "다름" },
        );
        Ok(same)
    }
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn main() -> Outcome<()> {
    let (mut target, mut draft) = ("smollm2_1_7b".to_owned(), "smollm2_135m".to_owned());
    let (mut limit, mut repeats) = (128usize, 3usize);
    let mut arguments = std::env::args().skip(1);
    while let Some(flag) = arguments.next() {
        let mut value = || arguments.next().ok_or(format!("{flag} 뒤에 값이 없습니다"));
        match flag.as_str() {
            "--target" => target = value()?,
            "--draft" => draft = value()?,
            "--tokens" => limit = value()?.parse()?,
            "--repeats" => repeats = value()?.parse()?,
            _ => return Err(format!("모르는 인자입니다: {flag}").into()),
        }
    }
    let (target_directory, draft_directory) =
        (models_dir().join(&target), models_dir().join(&draft));
    for directory in [&target_directory, &draft_directory] {
        if !directory.join("model.safetensors").exists() {
            println!("건너뜀: {}에 모델이 없습니다", directory.display());
            return Ok(());
        }
    }
    let big = LlamaModel::load(&target_directory)?;
    let small = LlamaModel::load(&draft_directory)?;
    let tokenizer = Tokenizer::from_tokenizer_json(target_directory.join("tokenizer.json"))?;
    // 모델의 틀대로 묻는다. SmolLM2는 기본 지시문 한 줄이 앞에 놓인다(47.3절).
    let template = ChatTemplate::load(&target_directory, &tokenizer);
    println!(
        "큰 모델 {target}(값 {:.0}M개), 작은 모델 {draft}(값 {:.0}M개), 토큰 {limit}개씩\n",
        big.parameter_count() as f64 / 1e6,
        small.parameter_count() as f64 / 1e6
    );

    let mut all_same = true;
    for (name, question) in tasks() {
        let prompt: Vec<TokenId> =
            template.encode(&tokenizer, &[Message::new("user", &question)], true);

        // 기준: 큰 모델이 혼자 쓴다. 프롬프트를 읽은 뒤부터의 시간을 잰다.
        let mut expected = Vec::new();
        let mut alone = Vec::new();
        for _ in 0..repeats {
            let mut cache = KvCache::new(&big.config);
            big.forward_cached(&prompt[..prompt.len() - 1], &mut cache)?;
            let start = Instant::now();
            expected = generate(&big, &prompt, limit, &mut cache, |_| {})?;
            alone.push(start.elapsed().as_secs_f64());
        }
        let alone = median(alone);
        println!(
            "{name} (프롬프트 {}토큰, 쓴 토큰 {}개)",
            prompt.len(),
            expected.len()
        );
        println!(
            "  혼자                          {:>6.1} 토큰/초",
            expected.len() as f64 / alone
        );

        let setup = Setup {
            big: &big,
            prompt: &prompt,
            limit,
            repeats,
            expected: &expected,
            alone,
        };
        for lookahead in [2, 4, 8] {
            all_same &= setup.run("작은 모델로", lookahead, &|| {
                Box::new(DraftModel::new(&small))
            })?;
        }
        for lookahead in [4, 8, 16] {
            all_same &= setup.run("글에서 찾아", lookahead, &|| {
                Box::new(Lookup::default())
            })?;
        }
        println!();
    }
    println!(
        "쓴 토큰이 모두 혼자 쓴 것과 같은가: {}",
        if all_same { "같음" } else { "다름" }
    );
    if !all_same {
        return Err("추측 디코딩의 결과가 기준과 다릅니다".into());
    }
    Ok(())
}
