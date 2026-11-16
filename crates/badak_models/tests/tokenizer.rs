//! llama.cpp가 자기 토크나이저를 시험할 때 쓰는 자료로 우리 토크나이저를
//! 검증한다.
//!
//! 파일이 없으면 건너뛴다. `sh scripts/download_ch08.sh`로 받는다.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use badak_formats::gguf::Gguf;
use badak_models::models_dir;
use badak_models::tokenizer::chat::{ChatFormat, Message};
use badak_models::tokenizer::{StreamDecoder, TokenId, Tokenizer};

/// 시험 입력들을 가르는 줄.
const SEPARATOR: &str = "\n__ggml_vocab_test__\n";

fn vocab_file(name: &str) -> Option<PathBuf> {
    let path = models_dir().join("llama_cpp_vocab").join(name);
    if !path.exists() {
        eprintln!("건너뜀: {}가 없습니다", path.display());
        return None;
    }
    Some(path)
}

/// 시험 자료를 (입력, 기대하는 토큰 번호들)의 목록으로 읽는다.
fn test_vectors(gguf: &Path) -> Vec<(String, Vec<TokenId>)> {
    let inputs = std::fs::read_to_string(gguf.with_extension("gguf.inp")).unwrap();
    let outputs = std::fs::read_to_string(gguf.with_extension("gguf.out")).unwrap();
    inputs
        .split(SEPARATOR)
        .zip(outputs.lines())
        .map(|(input, line)| {
            let ids = line
                .split_whitespace()
                .map(|id| id.parse().unwrap())
                .collect();
            (input.to_string(), ids)
        })
        .collect()
}

// ANCHOR: check
fn check(name: &str) {
    let Some(path) = vocab_file(name) else { return };
    let tokenizer = Tokenizer::from_gguf(&Gguf::open(&path).unwrap()).unwrap();
    let vectors = test_vectors(&path);
    assert!(vectors.len() > 40, "시험 자료가 너무 적다");

    let mut cache = HashMap::new();
    for (input, expected) in &vectors {
        let ids = tokenizer.encode(input, false);
        assert_eq!(&ids, expected, "입력 {input:?}");
        // 캐시를 쓰는 길도 같은 답을 내야 한다.
        assert_eq!(&tokenizer.encode_cached(input, false, &mut cache), expected);
        // 되돌리면 원래 글이 나와야 한다.
        assert_eq!(&tokenizer.decode(&ids), input);
        // 한 토큰씩 되돌려도 같아야 한다.
        let mut stream = StreamDecoder::default();
        let mut text: String = ids.iter().map(|&id| stream.push(&tokenizer, id)).collect();
        text.push_str(&stream.finish());
        assert_eq!(&text, input);
    }
}
// ANCHOR_END: check

#[test]
fn llama3_tokenizer_matches_llama_cpp() {
    check("ggml-vocab-llama-bpe.gguf");
}

#[test]
fn gpt2_tokenizer_matches_llama_cpp() {
    check("ggml-vocab-gpt-2.gguf");
}

#[test]
fn special_tokens_are_parsed_only_on_request() {
    let Some(path) = vocab_file("ggml-vocab-llama-bpe.gguf") else {
        return;
    };
    let tokenizer = Tokenizer::from_gguf(&Gguf::open(&path).unwrap()).unwrap();
    assert_eq!(tokenizer.vocab_size(), 128_256);
    assert_eq!(tokenizer.bos(), Some(128_000));

    let prompt = ChatFormat::Llama3.render(&[Message::new("user", "Hi")], true);
    let ids = tokenizer.encode(&prompt, true);
    let header_start = tokenizer.token_id("<|start_header_id|>").unwrap();
    let header_end = tokenizer.token_id("<|end_header_id|>").unwrap();
    let end_of_turn = tokenizer.token_id("<|eot_id|>").unwrap();
    // <|start_header_id|> user <|end_header_id|> \n\n Hi <|eot_id|> ...
    assert_eq!(ids[0], header_start);
    assert_eq!(ids[2], header_end);
    assert_eq!(ids[5], end_of_turn);
    assert_eq!(ids.iter().filter(|&&id| id >= 128_000).count(), 5);
    assert_eq!(tokenizer.decode(&ids), prompt);

    // 사용자의 글로 다룰 때는 같은 표기가 여러 개의 보통 토큰으로 쪼개진다.
    let plain = tokenizer.encode("<|eot_id|>", false);
    assert!(plain.len() > 1 && !plain.contains(&end_of_turn));
}

/// 모델 디렉터리의 `tokenizer.json`으로 만든 토크나이저를, 같은 파일을
/// Hugging Face의 tokenizers 라이브러리로 읽어 만든 시험 자료와 대조한다.
// ANCHOR: check_json
fn check_tokenizer_json(model: &str) {
    let directory = models_dir().join(model);
    let vectors = directory.join("tokenizer_vectors.json");
    if !vectors.exists() || !directory.join("tokenizer.json").exists() {
        eprintln!(
            "건너뜀: {}에 토크나이저 파일이나 시험 자료가 없습니다",
            directory.display()
        );
        return;
    }
    let tokenizer = Tokenizer::from_tokenizer_json(directory.join("tokenizer.json")).unwrap();
    let vectors = badak_formats::json::parse(&std::fs::read_to_string(vectors).unwrap()).unwrap();
    let vectors = vectors.as_array().unwrap();
    assert!(vectors.len() > 50);

    for vector in vectors {
        let text = vector.get("text").unwrap().as_str().unwrap();
        let expected: Vec<TokenId> = vector
            .get("ids")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_usize().unwrap() as TokenId)
            .collect();
        // 기준 라이브러리는 글 속의 특수 토큰 표기를 특수 토큰으로 바꾼다.
        let ids = tokenizer.encode(text, true);
        assert_eq!(ids, expected, "입력 {text:?}");
        // 단어 사전에 없는 바이트가 버려진 글은 되돌려도 원래 글이 되지 않는다.
        if !text.contains('\u{4}') {
            assert_eq!(tokenizer.decode(&ids), text);
        }
    }
}
// ANCHOR_END: check_json

#[test]
fn smollm2_tokenizer_json_matches_the_reference_library() {
    check_tokenizer_json("smollm2_135m");
}

#[test]
fn llama32_tokenizer_json_matches_the_reference_library() {
    check_tokenizer_json("llama32_1b");
}

#[test]
fn llama32_tokenizer_json_agrees_with_the_gguf_vocabulary() {
    // 같은 토크나이저를 두 형식에서 읽었으니 결과가 같아야 한다.
    let (Some(gguf), json) = (
        vocab_file("ggml-vocab-llama-bpe.gguf"),
        models_dir().join("llama32_1b/tokenizer.json"),
    ) else {
        return;
    };
    if !json.exists() {
        eprintln!("건너뜀: {}가 없습니다", json.display());
        return;
    }
    let from_gguf = Tokenizer::from_gguf(&Gguf::open(&gguf).unwrap()).unwrap();
    let from_json = Tokenizer::from_tokenizer_json(json).unwrap();
    assert_eq!(from_json.vocab_size(), from_gguf.vocab_size());
    assert_eq!(from_json.merge_count(), from_gguf.merge_count());
    for (input, expected) in test_vectors(&gguf) {
        assert_eq!(from_json.encode(&input, false), expected, "입력 {input:?}");
    }
}
