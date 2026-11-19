//! 38장: 파이프라인의 단계마다 PyTorch로 만든 정답과 견준다.
//!
//! 실행: `cargo run --release -p badak_models --example ch38_verify`
//!
//! 모델 셋(Whisper-tiny, SmolLM2-135M-Instruct, OuteTTS-0.1-350M)이 차례로
//! 쓴 토큰이 transformers로 같은 사슬을 돌린 것과 같은지 본다.

use badak_audio::Wav;
use badak_formats::safetensors::SafeTensors;
use badak_models::llama::model::LlamaModel;
use badak_models::models_dir;
use badak_models::talk::{Assistant, Settings, Speak};
use badak_models::tokenizer::chat::ChatFormat;
use badak_models::tokenizer::{TokenId, Tokenizer};
use badak_models::tts::{OuteTts, Voice};
use badak_models::whisper::{Vocabulary, Whisper};

type Outcome<T> = Result<T, Box<dyn std::error::Error>>;

fn main() -> Outcome<()> {
    let reference = SafeTensors::open(models_dir().join("talk/reference.safetensors"))?;
    let ids = |key: &str| -> Outcome<Vec<TokenId>> {
        Ok(reference
            .tensor(key)?
            .to_vec::<i32>()?
            .iter()
            .map(|&id| id as TokenId)
            .collect())
    };
    let (whisper, outetts, chat) = (
        models_dir().join("whisper"),
        models_dir().join("outetts"),
        models_dir().join("smollm2_135m"),
    );
    // 디코더의 가중치는 변환해야 생긴다. 없으면 오디오 토큰까지만 견준다.
    let weights = outetts.join("wavtokenizer.safetensors");
    if !weights.exists() {
        println!("오디오 디코더의 가중치가 없어 건너뜁니다 (reference/ch37_convert.py)");
        return Ok(());
    }
    let assistant = Assistant {
        whisper: Whisper::load(&whisper)?,
        vocabulary: Vocabulary::load(&whisper)?,
        chat: LlamaModel::load(&chat)?,
        chat_tokenizer: Tokenizer::from_tokenizer_json(chat.join("tokenizer.json"))?,
        format: ChatFormat::ChatMl,
        system: "You are a friendly voice assistant. Answer in two short sentences.".to_owned(),
        tts: OuteTts::load(&outetts)?,
        tts_tokenizer: Tokenizer::from_tokenizer_json(outetts.join("tokenizer.json"))?,
        voice: Voice::load(weights)?,
    };
    let wav = Wav::open(whisper.join("jfk.wav"))?;
    let mut failed = false;
    let mut first: Option<Vec<f32>> = None;
    for speak in [Speak::BySentence, Speak::AfterAnswer] {
        let settings = Settings {
            speak,
            answer_tokens: 64,
            speech_tokens: 1500,
            voice_seed: None,
        };
        let turn = assistant.respond(&wav.samples, &settings, |_| {})?;
        println!(
            "\n{}",
            match speak {
                Speak::BySentence => "문장마다 소리로",
                Speak::AfterAnswer => "답을 다 쓴 뒤에 소리로",
            }
        );
        println!("  들은 말: {:?}", turn.heard);
        let same = turn.answer_ids == ids("answer")?;
        failed |= !same;
        println!(
            "  답의 토큰 {}개: {}",
            turn.answer_ids.len(),
            if same {
                "정답과 같음"
            } else {
                "정답과 다름"
            }
        );
        let expected = ids("sentences")?[0] as usize;
        println!("  문장 {}개 (정답 {expected}개)", turn.sentences.len());
        failed |= turn.sentences.len() != expected;
        for (index, written) in turn.speech_ids.iter().enumerate() {
            let same = *written == ids(&format!("speech.{index}"))?;
            failed |= !same;
            println!(
                "  문장 {}의 소리 토큰 {}개: {}",
                index + 1,
                written.len(),
                if same {
                    "정답과 같음"
                } else {
                    "정답과 다름"
                }
            );
        }
        // 소리로 바꾸는 때가 달라도 만들어지는 소리는 같아야 한다.
        match &first {
            None => first = Some(turn.audio),
            Some(audio) => {
                let same = *audio == turn.audio;
                failed |= !same;
                println!(
                    "  소리 {:.2}초: 문장마다 바꾼 것과 {}",
                    turn.audio.len() as f64 / 24_000.0,
                    if same {
                        "표본까지 같음"
                    } else {
                        "다름"
                    }
                );
            }
        }
    }
    if failed {
        return Err("정답과 다릅니다".into());
    }
    Ok(())
}
