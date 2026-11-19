//! 음성 대화 파이프라인을 PyTorch의 정답과 견준다 (38장).
//!
//! 모델 셋과 디코더의 가중치, 정답이 모두 있어야 돈다. 하나라도 없으면
//! 건너뛴다. 받는 방법은 `scripts/download_ch38.sh`의 머리말에 있다.

use badak_audio::Wav;
use badak_formats::safetensors::SafeTensors;
use badak_models::llama::model::LlamaModel;
use badak_models::models_dir;
use badak_models::talk::{Assistant, Settings, Speak};
use badak_models::tokenizer::chat::ChatFormat;
use badak_models::tokenizer::{TokenId, Tokenizer};
use badak_models::tts::{OuteTts, Voice};
use badak_models::whisper::{Vocabulary, Whisper};

#[test]
fn the_chain_writes_the_same_tokens_as_the_reference() {
    let (whisper, outetts, chat, talk) = (
        models_dir().join("whisper"),
        models_dir().join("outetts"),
        models_dir().join("smollm2_135m"),
        models_dir().join("talk"),
    );
    let needed = [
        whisper.join("model.safetensors"),
        outetts.join("model.safetensors"),
        outetts.join("wavtokenizer.safetensors"),
        chat.join("model.safetensors"),
        talk.join("reference.safetensors"),
    ];
    if let Some(missing) = needed.iter().find(|path| !path.exists()) {
        eprintln!("건너뜀: {}이 없습니다", missing.display());
        return;
    }
    let reference = SafeTensors::open(talk.join("reference.safetensors")).unwrap();
    let ids = |key: &str| -> Vec<TokenId> {
        reference
            .tensor(key)
            .unwrap()
            .to_vec::<i32>()
            .unwrap()
            .iter()
            .map(|&id| id as TokenId)
            .collect()
    };
    let assistant = Assistant {
        whisper: Whisper::load(&whisper).unwrap(),
        vocabulary: Vocabulary::load(&whisper).unwrap(),
        chat: LlamaModel::load(&chat).unwrap(),
        chat_tokenizer: Tokenizer::from_tokenizer_json(chat.join("tokenizer.json")).unwrap(),
        format: ChatFormat::ChatMl,
        system: "You are a friendly voice assistant. Answer in two short sentences.".to_owned(),
        tts: OuteTts::load(&outetts).unwrap(),
        tts_tokenizer: Tokenizer::from_tokenizer_json(outetts.join("tokenizer.json")).unwrap(),
        voice: Voice::load(outetts.join("wavtokenizer.safetensors")).unwrap(),
    };
    let wav = Wav::open(whisper.join("jfk.wav")).unwrap();
    // 시험이 오래 걸리지 않게 답은 16토큰, 소리는 문장마다 40토큰까지만 쓴다.
    let settings = Settings {
        speak: Speak::BySentence,
        answer_tokens: 16,
        speech_tokens: 40,
        voice_seed: None,
    };
    let mut pieces = 0;
    let turn = assistant
        .respond(&wav.samples, &settings, |samples| pieces += samples.len())
        .unwrap();
    assert!(turn.heard.starts_with("And so my fellow Americans"));
    assert_eq!(turn.answer_ids, ids("answer")[..16]);
    assert_eq!(
        turn.sentences[0],
        "I'm glad you're enjoying the conversation."
    );
    assert_eq!(turn.speech_ids[0], ids("speech.0")[..40]);
    // 내보낸 조각을 이으면 돌려준 소리다.
    assert_eq!(pieces, turn.audio.len());
    assert!(!turn.chunks.is_empty());
    // 시각이 순서대로 놓인다.
    assert!(turn.encoded_at <= turn.heard_at);
    assert!(turn.heard_at <= turn.first_token_at);
    assert!(turn.first_token_at <= turn.first_sentence_at);
    assert!(turn.first_sentence_at <= turn.first_audio_at);
    assert!(turn.first_audio_at <= turn.done_at);
    // 문장마다 말하면 답을 다 쓰기 전에 첫 소리가 나온다.
    assert!(turn.first_audio_at < turn.answered_at);

    let whole = assistant
        .respond(
            &wav.samples,
            &Settings {
                speak: Speak::AfterAnswer,
                ..settings
            },
            |_| {},
        )
        .unwrap();
    assert!(whole.first_audio_at >= whole.answered_at);
    assert_eq!(whole.audio, turn.audio);
}
