//! 8장: 파일을 토큰 번호로 바꿔 llama.cpp의 `llama-tokenize`와 같은 형식으로
//! 출력한다. 두 도구의 결과를 비교할 때 쓴다.
//!
//! 실행: `cargo run --release -p badak_models --example ch08_dump -- 단어사전.gguf 글.txt`

use badak_formats::gguf::Gguf;
use badak_models::tokenizer::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let (Some(vocab), Some(file)) = (arguments.next(), arguments.next()) else {
        return Err("단어 사전 파일과 글 파일의 경로를 주십시오".into());
    };
    let tokenizer = Tokenizer::from_gguf(&Gguf::open(vocab)?)?;
    let ids = tokenizer.encode(&std::fs::read_to_string(file)?, false);
    let ids: Vec<String> = ids.iter().map(|id| id.to_string()).collect();
    println!("[{}]", ids.join(", "));
    Ok(())
}
