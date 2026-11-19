//! 38장: 말을 듣고 말로 답한다.
//!
//! 실행: `cargo run --release -p badak_models --example ch38_talk -- \
//!        [--input 소리.wav] [--whole] [--seed 번호] [--chat 모델 디렉터리의 이름] [--out 답.wav]`
//!
//! 사용자의 말은 WAV 파일로 받는다(16kHz가 아니면 바꾼다). `--whole`을
//! 주면 답을 다 쓴 뒤에 소리로 바꾸고, 주지 않으면 문장마다 바꾼다.

use badak_audio::{Wav, resample};
use badak_models::llama::model::LlamaModel;
use badak_models::models_dir;
use badak_models::talk::{Assistant, Settings, Speak, Turn};
use badak_models::tokenizer::Tokenizer;
use badak_models::tokenizer::chat::ChatFormat;
use badak_models::tts::{OuteTts, SAMPLE_RATE, Voice};
use badak_models::whisper::{Vocabulary, Whisper};

type Outcome<T> = Result<T, Box<dyn std::error::Error>>;

/// 모델 셋을 읽는다. `chat`은 답을 쓰는 모델의 디렉터리 이름이다.
fn load(chat: &str) -> Outcome<Assistant> {
    let (whisper, outetts, chat_directory) = (
        models_dir().join("whisper"),
        models_dir().join("outetts"),
        models_dir().join(chat),
    );
    Ok(Assistant {
        whisper: Whisper::load(&whisper)?.fast()?,
        vocabulary: Vocabulary::load(&whisper)?,
        chat: LlamaModel::load(&chat_directory)?,
        chat_tokenizer: Tokenizer::from_tokenizer_json(chat_directory.join("tokenizer.json"))?,
        format: if chat.contains("llama") {
            ChatFormat::Llama3
        } else {
            ChatFormat::ChatMl
        },
        system: "You are a friendly voice assistant. Answer in two short sentences.".to_owned(),
        tts: OuteTts::load(&outetts)?,
        tts_tokenizer: Tokenizer::from_tokenizer_json(outetts.join("tokenizer.json"))?,
        voice: Voice::load(outetts.join("wavtokenizer.safetensors"))?.fast(),
    })
}

/// 한 번 주고받은 기록을 적는다.
fn report(turn: &Turn) {
    println!("들은 말: {:?}", turn.heard);
    println!("쓴 답:   {:?}", turn.answer.trim());
    println!("구간                         끝난 때     걸린 시간");
    let stages = [
        ("듣기: 멜과 인코더", 0.0, turn.encoded_at),
        ("듣기: 글 쓰기", turn.encoded_at, turn.heard_at),
        ("답: 글 읽고 첫 토큰", turn.heard_at, turn.first_token_at),
        (
            "답: 첫 문장이 끝날 때까지",
            turn.first_token_at,
            turn.first_sentence_at,
        ),
        (
            "말: 첫 소리까지",
            // 답을 다 쓴 뒤에 말하는 설정에서는 이 사이에 답의 나머지도 쓴다.
            turn.first_sentence_at,
            turn.first_audio_at,
        ),
    ];
    for (name, from, until) in stages {
        println!(
            "  {name:<22} {:>8.0}ms {:>10.0}ms",
            until * 1e3,
            (until - from) * 1e3
        );
    }
    let seconds = turn.audio.len() as f64 / f64::from(SAMPLE_RATE);
    println!(
        "응답 지연(말이 끝난 때부터 첫 소리까지) {:.0}ms. 답 {}토큰, 소리 토큰 {}개, 소리 {seconds:.2}초, 모두 {:.2}초",
        turn.first_audio_at * 1e3,
        turn.answer_tokens,
        turn.speech_tokens,
        turn.done_at
    );
    // 첫 소리가 나오자마자 틀기 시작하면 어디서 얼마나 끊기는가.
    let mut playing = turn.first_audio_at;
    let (mut stalls, mut waited) = (0, 0.0);
    for chunk in &turn.chunks {
        if chunk.ready > playing + 1e-3 {
            stalls += 1;
            waited += chunk.ready - playing;
            playing = chunk.ready;
        }
        playing += chunk.seconds;
    }
    println!(
        "소리 조각 {}개. 바로 틀면 {stalls}번 끊기고 끊긴 시간의 합은 {:.0}ms",
        turn.chunks.len(),
        waited * 1e3
    );
    // 벤치마크 스크립트가 읽는 한 줄.
    println!(
        "talk 스레드 {:>2} 인코더 {:.0} 들음 {:.0} 첫토큰 {:.0} 첫문장 {:.0} 답끝 {:.0} 첫소리 {:.0} 끝 {:.0} 소리 {:.2} 끊김 {stalls} {:.0}",
        badak_core::threads::global().pool().threads(),
        turn.encoded_at * 1e3,
        turn.heard_at * 1e3,
        turn.first_token_at * 1e3,
        turn.first_sentence_at * 1e3,
        turn.answered_at * 1e3,
        turn.first_audio_at * 1e3,
        turn.done_at * 1e3,
        seconds,
        waited * 1e3
    );
    for (index, sentence) in turn.sentences.iter().enumerate() {
        let chunks: Vec<_> = turn
            .chunks
            .iter()
            .filter(|chunk| chunk.sentence == index)
            .collect();
        let (first, last) = (
            chunks.first().map_or(0.0, |chunk| chunk.ready),
            chunks.last().map_or(0.0, |chunk| chunk.ready),
        );
        let seconds: f64 = chunks.iter().map(|chunk| chunk.seconds).sum();
        println!(
            "  문장 {}: 소리 {seconds:.2}초, 조각이 나온 때 {:.0}~{:.0}ms  {sentence:?}",
            index + 1,
            first * 1e3,
            last * 1e3
        );
    }
}

fn main() -> Outcome<()> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let option = |name: &str| {
        arguments
            .iter()
            .position(|argument| argument == name)
            .and_then(|at| arguments.get(at + 1))
    };
    let has = |name: &str| arguments.iter().any(|argument| argument == name);
    let input = option("--input").map_or_else(
        || models_dir().join("whisper/jfk.wav"),
        std::path::PathBuf::from,
    );
    let out = option("--out").map_or_else(
        || models_dir().join("outetts/answer.wav"),
        std::path::PathBuf::from,
    );
    let chat = option("--chat").map_or("smollm2_135m", String::as_str);
    let settings = Settings {
        speak: if has("--whole") {
            Speak::AfterAnswer
        } else {
            Speak::BySentence
        },
        answer_tokens: 64,
        speech_tokens: 1500,
        voice_seed: option("--seed").map(|text| text.parse()).transpose()?,
    };

    let start = std::time::Instant::now();
    let assistant = load(chat)?;
    println!(
        "모델 셋을 읽는 데 {:.0}ms. 답을 쓰는 모델 {chat}, {}",
        start.elapsed().as_secs_f64() * 1e3,
        match settings.speak {
            Speak::BySentence => "문장마다 소리로",
            Speak::AfterAnswer => "답을 다 쓴 뒤에 소리로",
        }
    );
    let wav = Wav::open(&input)?;
    let samples = resample(&wav.samples, wav.sample_rate, 16_000);
    println!("사용자의 말: {} ({:.1}초)", input.display(), wav.seconds());

    let turn = assistant.respond(&samples, &settings, |_| {})?;
    report(&turn);
    Wav {
        sample_rate: SAMPLE_RATE,
        samples: turn.audio.clone(),
    }
    .save(&out)?;
    println!("{} 기록", out.display());

    if has("--listen") {
        // 답의 소리를 다시 받아써 본다(37.8절).
        let heard = resample(&turn.audio, SAMPLE_RATE, 16_000);
        let mel = badak_audio::LogMel::WHISPER.spectrogram(&heard);
        let encoded = assistant.whisper.encode(&mel.data)?;
        let prompt = assistant.vocabulary.prompt("en")?;
        let transcript =
            assistant
                .whisper
                .beam(&encoded, &assistant.vocabulary, &prompt, 5, 200)?;
        println!(
            "답의 소리를 Whisper가 받아쓴 글: {:?}",
            assistant.vocabulary.decode(&transcript.tokens)
        );
    }
    Ok(())
}
