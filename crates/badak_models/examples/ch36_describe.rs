//! 36장: 사진 파일을 보고 글로 답한다.
//!
//! 실행: `cargo run --release -p badak_models --example ch36_describe -- \
//!        [사진.jpg] [질문]`
//!
//! 사진을 주지 않으면 `models/images/`의 사진들을 차례로 본다.

use std::io::Write;
use std::time::Instant;

use badak_image::Image;
use badak_image::lanczos::squash;
use badak_models::models_dir;
use badak_models::smolvlm::SmolVlm;
use badak_models::tokenizer::{StreamDecoder, Tokenizer};

type Outcome<T> = Result<T, Box<dyn std::error::Error>>;

fn main() -> Outcome<()> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let directory = models_dir().join("smolvlm");
    let model = SmolVlm::load(&directory)?.fast()?;
    let tokenizer = Tokenizer::from_tokenizer_json(directory.join("tokenizer.json"))?;
    let stop = tokenizer
        .token_id("<end_of_utterance>")
        .ok_or("끝 토큰이 없습니다")?;
    let question = arguments
        .get(1)
        .map_or("Describe this image.", String::as_str);
    let paths: Vec<std::path::PathBuf> = match arguments.first() {
        Some(path) => vec![path.into()],
        None => ["cup", "golden_retriever", "guitar", "pelican", "tabby_cat"]
            .iter()
            .map(|name| models_dir().join(format!("images/{name}.jpg")))
            .collect(),
    };
    // ANCHOR: prompt
    // 글의 틀. 사진 자리 64개를 표시 토큰이 감싸고, 그 뒤에 질문이 온다.
    let prompt = format!(
        "<|im_start|>User:<fake_token_around_image><global-img>{}<fake_token_around_image>{question}<end_of_utterance>\nAssistant:",
        "<image>".repeat(model.image_tokens())
    );
    let ids = tokenizer.encode(&prompt, true);
    // ANCHOR_END: prompt

    for path in paths {
        let image = Image::open(&path)?;
        let start = Instant::now();
        let pixels = squash(&image, 2048, model.vision.image_size());
        let preprocess_time = start.elapsed();
        let start = Instant::now();
        let seen = model.see(&pixels)?;
        let see_time = start.elapsed();

        println!(
            "\n{} ({} × {})",
            path.file_name()
                .map_or_else(String::new, |name| name.to_string_lossy().into_owned()),
            image.width,
            image.height
        );
        print!("  ");
        let start = Instant::now();
        let mut first = None;
        let mut decoder = StreamDecoder::default();
        let (written, _) = model.generate(&ids, &seen, stop, 64, |token| {
            first.get_or_insert_with(|| start.elapsed());
            print!("{}", decoder.push(&tokenizer, token));
            std::io::stdout().flush().ok();
        })?;
        println!("{}", decoder.finish());
        let total = start.elapsed();
        let first = first.unwrap_or(total);
        println!(
            "  전처리 {:.0}ms, 사진 보기 {:.0}ms, 글 읽기(토큰 {}개) {:.0}ms, 쓰기 {}토큰 {:.0}ms(토큰마다 {:.1}ms)",
            preprocess_time.as_secs_f64() * 1e3,
            see_time.as_secs_f64() * 1e3,
            ids.len(),
            first.as_secs_f64() * 1e3,
            written.len(),
            (total - first).as_secs_f64() * 1e3,
            (total - first).as_secs_f64() * 1e3 / written.len().max(2).saturating_sub(1) as f64
        );
    }
    Ok(())
}
