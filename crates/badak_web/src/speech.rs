//! 받아쓰기의 문 (50장).
//!
//! 33장의 Whisper를 브라우저에서 부른다. 소리 파일을 읽고, 말이 있는 곳을
//! 찾아 창으로 나누고, 창마다 받아쓰는 일까지 엔진 안에서 한다. JavaScript는
//! 파일의 길을 주고 글을 받는다.

use badak_audio::vad::{self, Settings};
use badak_audio::{LogMel, Wav};
use badak_models::whisper::{Vocabulary, Whisper};

use crate::{fail, text};

/// Whisper 모델과 그 낱말 표.
pub struct Speech {
    model: Whisper,
    vocabulary: Vocabulary,
}

/// 모델 폴더(`config.json`, `model.safetensors`, `vocab.json` 등)를 연다.
/// 실패하면 빈 포인터다.
///
/// # Safety
///
/// `directory`부터 `length`바이트의 글을 읽을 수 있어야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn badak_web_whisper_open(
    directory: *const u8,
    length: usize,
) -> *mut Speech {
    // SAFETY: 부르는 쪽의 약속.
    let Some(directory) = (unsafe { text(directory, length) }) else {
        return std::ptr::null_mut();
    };
    let open = || -> Result<Speech, Box<dyn std::error::Error>> {
        Ok(Speech {
            model: Whisper::load(directory)?.fast()?,
            vocabulary: Vocabulary::load(directory)?,
        })
    };
    match open() {
        Ok(speech) => Box::into_raw(Box::new(speech)),
        Err(error) => fail::<()>(error).map_or(std::ptr::null_mut(), |()| std::ptr::null_mut()),
    }
}

// ANCHOR: transcribe
/// WAV 파일 하나를 받아쓴다. 글을 `out`에 채우고 전체 길이를 돌려준다.
/// 실패하면 -1이다.
///
/// `beams`가 1이면 스텝마다 가장 높은 토큰을 고르고, 그보다 크면 그만큼의
/// 후보를 들고 간다(33장).
///
/// # Safety
///
/// `speech`는 [`badak_web_whisper_open`]이 준 포인터여야 하고, `path`부터
/// `length`바이트의 글을 읽을 수 있어야 하며, `out`부터 `capacity`바이트를 쓸
/// 수 있어야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn badak_web_whisper_transcribe(
    speech: *const Speech,
    path: *const u8,
    length: usize,
    beams: usize,
    out: *mut u8,
    capacity: usize,
) -> i32 {
    // SAFETY: 부르는 쪽의 약속.
    let (speech, Some(path)) = (unsafe { &*speech }, unsafe { text(path, length) }) else {
        return -1;
    };
    let transcribe = || -> Result<String, Box<dyn std::error::Error>> {
        let wav = Wav::open(path)?;
        if wav.sample_rate != LogMel::WHISPER.sample_rate {
            return Err(format!(
                "표본 속도가 {}Hz입니다. 16,000Hz여야 합니다",
                wav.sample_rate
            )
            .into());
        }
        // 말이 있는 곳을 찾아 30초 이하의 창으로 나눈다.
        let segments = vad::segments(&wav.samples, wav.sample_rate, &Settings::default());
        let windows = vad::windows(&segments, LogMel::WHISPER.samples);
        let prompt = speech.vocabulary.prompt("en")?;
        let mut written = String::new();
        for window in windows {
            let mel = LogMel::WHISPER.spectrogram(&wav.samples[window]);
            let encoded = speech.model.encode(&mel.data)?;
            let transcript = if beams > 1 {
                speech
                    .model
                    .beam(&encoded, &speech.vocabulary, &prompt, beams, 200)?
            } else {
                speech
                    .model
                    .greedy(&encoded, &speech.vocabulary, &prompt, 200)?
            };
            written.push_str(&speech.vocabulary.decode(&transcript.tokens));
        }
        Ok(written)
    };
    match transcribe() {
        Ok(written) => {
            let bytes = written.as_bytes();
            // SAFETY: 부르는 쪽의 약속.
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), out, bytes.len().min(capacity))
            };
            bytes.len() as i32
        }
        Err(error) => fail::<()>(error).map_or(-1, |()| -1),
    }
}
// ANCHOR_END: transcribe

/// 모델을 지운다.
///
/// # Safety
///
/// `speech`는 [`badak_web_whisper_open`]이 준 뒤 아직 지우지 않은 포인터여야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn badak_web_whisper_close(speech: *mut Speech) {
    if !speech.is_null() {
        // SAFETY: 부르는 쪽의 약속.
        drop(unsafe { Box::from_raw(speech) });
    }
}
