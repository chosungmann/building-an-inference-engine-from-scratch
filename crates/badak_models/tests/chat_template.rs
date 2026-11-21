//! 모델의 대화 틀을 transformers가 낸 글과 견준다 (47.3절).
//!
//! 정답은 `bench/ch47/chat_template.txt`에 있다. transformers의
//! `apply_chat_template`이 모델과 함께 온 틀(Jinja)을 실행해 낸 글과 그
//! 토큰이다(`reference/ch47_chat_template.py`). 모델의 파일이 없으면
//! 건너뛴다. `sh scripts/download_smollm2.sh`로 받는다.

use std::path::Path;

use badak_formats::json;
use badak_models::models_dir;
use badak_models::tokenizer::chat::{ChatTemplate, Message};
use badak_models::tokenizer::{TokenId, Tokenizer};

#[test]
fn rendered_chats_match_transformers() {
    let answers = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/ch47/chat_template.txt");
    let answers = std::fs::read_to_string(answers).unwrap();
    let mut checked = 0;
    for line in answers.lines() {
        let answer = json::parse(line).unwrap();
        let text =
            |value: &json::Value, key: &str| value.get(key).unwrap().as_str().unwrap().to_owned();
        let directory = models_dir().join(text(&answer, "model"));
        if !directory.join("tokenizer_config.json").exists() {
            eprintln!("건너뜀: {}에 틀이 없습니다", directory.display());
            continue;
        }
        let tokenizer = Tokenizer::from_tokenizer_json(directory.join("tokenizer.json")).unwrap();
        let template = ChatTemplate::load(&directory, &tokenizer);
        assert_eq!(
            template.default_system.as_deref(),
            Some("You are a helpful AI assistant named SmolLM, trained by Hugging Face")
        );
        let messages: Vec<Message> = answer
            .get("messages")
            .and_then(json::Value::as_array)
            .unwrap()
            .iter()
            .map(|message| Message::new(&text(message, "role"), &text(message, "content")))
            .collect();
        let expected: Vec<TokenId> = answer
            .get("ids")
            .and_then(json::Value::as_array)
            .unwrap()
            .iter()
            .map(|id| id.as_usize().unwrap() as TokenId)
            .collect();
        // 글이 같고, 그 글을 통째로 토큰으로 바꾼 것이 같다.
        let rendered = template.render(&messages, true);
        assert_eq!(rendered, text(&answer, "text"));
        assert_eq!(tokenizer.encode(&rendered, true), expected);
        // 뼈대와 내용을 따로 토큰으로 바꿔도(39장) 같다.
        assert_eq!(template.encode(&tokenizer, &messages, true), expected);
        checked += 1;
    }
    eprintln!("대조한 대화 {checked}개");
}
