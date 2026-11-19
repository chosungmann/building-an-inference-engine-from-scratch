//! 말을 듣고 말로 답하는 파이프라인 (38장).
//!
//! 모델 셋을 잇는다. 33장의 Whisper가 소리를 글로 바꾸고, 2부의 언어
//! 모델이 답을 쓰고, 37장의 음성 합성이 그 답을 소리로 바꾼다. 새로 만드는
//! 계산은 없다. 이 파일에 있는 것은 **잇는 방법**과 **시간을 재는 자리**다.

use std::time::Instant;

use badak_audio::LogMel;
use badak_formats::Result;

use crate::llama::cache::KvCache;
use crate::llama::generate::generate;
use crate::llama::model::LlamaModel;
use crate::sampling::{Sampler, SamplerConfig};
use crate::tokenizer::chat::{ChatFormat, Message};
use crate::tokenizer::{StreamDecoder, TokenId, Tokenizer};
use crate::tts::{OuteTts, SAMPLE_RATE, SAMPLES_PER_CODE, Voice, prompt};
use crate::whisper::{Vocabulary, Whisper};

// ANCHOR: sentences
/// 흘러 들어오는 글에서 문장이 끝날 때마다 그 문장을 내준다.
///
/// 언어 모델은 토큰을 하나씩 쓴다. 답을 다 쓸 때까지 기다리지 않고, 문장
/// 하나가 끝나면 그 문장부터 소리로 바꾸려는 것이다. 마침표, 물음표,
/// 느낌표 **뒤에 빈칸이 온 것을 본 다음에** 문장이 끝났다고 본다. `3.14`나
/// 글의 맨 끝에서 잘못 자르지 않으려는 것이다. 그래서 문장은 다음 문장의
/// 첫 토큰이 나온 뒤에야 나온다.
#[derive(Debug, Default)]
pub struct Sentences {
    pending: String,
}

impl Sentences {
    /// 글 조각을 더하고, 그 사이에 끝난 문장들을 돌려준다.
    pub fn push(&mut self, piece: &str) -> Vec<String> {
        self.pending.push_str(piece);
        let mut done = Vec::new();
        loop {
            let mut end = None;
            let mut previous = ' ';
            for (offset, character) in self.pending.char_indices() {
                if character.is_whitespace() && matches!(previous, '.' | '!' | '?') {
                    end = Some(offset);
                    break;
                }
                previous = character;
            }
            let Some(end) = end else {
                break;
            };
            let rest = self.pending.split_off(end);
            let sentence = std::mem::replace(&mut self.pending, rest);
            if sentence.chars().any(char::is_alphabetic) {
                done.push(sentence.trim().to_owned());
            }
        }
        done
    }

    /// 글이 끝났다. 남은 것이 있으면 마지막 문장으로 돌려준다.
    pub fn finish(self) -> Option<String> {
        let rest = self.pending.trim();
        rest.chars()
            .any(char::is_alphabetic)
            .then(|| rest.to_owned())
    }
}
// ANCHOR_END: sentences

/// 답을 언제부터 소리로 바꾸는가.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Speak {
    /// 언어 모델이 답을 다 쓴 뒤에.
    AfterAnswer,
    /// 문장 하나가 끝날 때마다. 언어 모델은 그동안 멈춘다.
    BySentence,
}

/// 파이프라인의 설정.
#[derive(Debug, Clone, Copy)]
pub struct Settings {
    /// 답을 언제부터 소리로 바꾸는가.
    pub speak: Speak,
    /// 언어 모델이 쓸 토큰의 한도.
    pub answer_tokens: usize,
    /// 음성 합성이 문장 하나에 쓸 토큰의 한도.
    pub speech_tokens: usize,
    /// 음성 합성의 난수 번호. `None`이면 점수가 가장 높은 토큰을 고른다.
    pub voice_seed: Option<u64>,
}

// ANCHOR: turn
/// 소리 조각 하나가 만들어진 기록.
#[derive(Debug, Clone, PartialEq)]
pub struct Chunk {
    /// 몇 번째 문장인가.
    pub sentence: usize,
    /// 듣기 시작한 때부터 이 조각이 나올 때까지(초).
    pub ready: f64,
    /// 조각의 길이(초).
    pub seconds: f64,
}

/// 한 번 주고받은 기록. 시각은 모두 "사용자의 말이 끝난 때"부터 잰
/// 초다.
#[derive(Debug, Clone, Default)]
pub struct Turn {
    /// 받아쓴 글.
    pub heard: String,
    /// 언어 모델이 쓴 답.
    pub answer: String,
    /// 답을 문장으로 나눈 것.
    pub sentences: Vec<String>,
    /// 답의 소리(24kHz).
    pub audio: Vec<f32>,
    /// 받아쓰기가 끝난 때.
    pub heard_at: f64,
    /// 그 가운데 멜 스펙트로그램과 인코더가 끝난 때.
    pub encoded_at: f64,
    /// 언어 모델의 첫 토큰이 나온 때.
    pub first_token_at: f64,
    /// 첫 문장이 끝난 때.
    pub first_sentence_at: f64,
    /// 언어 모델이 답을 다 쓴 때.
    pub answered_at: f64,
    /// 첫 소리가 나온 때. 이 숫자가 "응답 지연"이다.
    pub first_audio_at: f64,
    /// 모두 끝난 때.
    pub done_at: f64,
    /// 언어 모델이 쓴 토큰의 수.
    pub answer_tokens: usize,
    /// 음성 합성이 쓴 토큰의 수.
    pub speech_tokens: usize,
    /// 소리 조각들.
    pub chunks: Vec<Chunk>,
    /// 언어 모델이 쓴 토큰들. 정답과 견줄 때 쓴다.
    pub answer_ids: Vec<TokenId>,
    /// 음성 합성이 문장마다 쓴 토큰들.
    pub speech_ids: Vec<Vec<TokenId>>,
}
// ANCHOR_END: turn

/// 모델 셋을 쥐고 있는 것.
pub struct Assistant {
    /// 받아쓰는 모델.
    pub whisper: Whisper,
    /// 받아쓰는 모델의 단어 사전.
    pub vocabulary: Vocabulary,
    /// 답을 쓰는 모델.
    pub chat: LlamaModel,
    /// 답을 쓰는 모델의 토크나이저.
    pub chat_tokenizer: Tokenizer,
    /// 답을 쓰는 모델의 대화 형식.
    pub format: ChatFormat,
    /// 답을 쓰는 모델에 주는 지시.
    pub system: String,
    /// 오디오 토큰을 쓰는 모델.
    pub tts: OuteTts,
    /// 오디오 토큰을 쓰는 모델의 토크나이저.
    pub tts_tokenizer: Tokenizer,
    /// 오디오 토큰을 소리로 바꾸는 것.
    pub voice: Voice,
}

/// 조각을 풀 때 함께 넣는 앞의 코드 수(37.9절).
const BEFORE: usize = 75;

impl Assistant {
    // ANCHOR: speak
    /// 문장 하나를 소리로 바꿔 `turn.audio`에 붙인다. 낱말이 끝날 때마다
    /// 조각을 풀어 `on_audio`로 내보낸다(37.9절).
    fn speak(
        &self,
        index: usize,
        sentence: &str,
        settings: &Settings,
        start: Instant,
        turn: &mut Turn,
        on_audio: &mut dyn FnMut(&[f32]),
    ) -> Result<()> {
        let tokenizer = &self.tts_tokenizer;
        let token = |name: &str| tokenizer.token_id(name).unwrap_or(TokenId::MAX);
        let (stop, word_end, first_code) = (
            token("<|audio_end|>"),
            token("<|code_end|>"),
            token("<|0|>"),
        );
        let mut sampler = Sampler::new(match settings.voice_seed {
            Some(seed) => SamplerConfig {
                temperature: 0.4,
                top_k: 40,
                top_p: 0.9,
                repetition_penalty: 1.1,
                seed: seed + index as u64,
                ..SamplerConfig::default()
            },
            None => SamplerConfig {
                temperature: 0.0,
                repetition_penalty: 1.1,
                ..SamplerConfig::default()
            },
        });
        let ids = tokenizer.encode(&prompt(sentence), true);
        let (mut codes, mut spoken): (Vec<u32>, usize) = (Vec::new(), 0);
        let mut failure = None;
        let (written, _) =
            self.tts
                .generate(&ids, stop, settings.speech_tokens, &mut sampler, |token| {
                    if (first_code..first_code.saturating_add(4096)).contains(&token) {
                        codes.push(token - first_code);
                    }
                    if token == word_end && codes.len() > spoken {
                        match self
                            .voice
                            .decode_part(&codes, spoken, codes.len(), BEFORE, 0)
                        {
                            Ok(samples) => {
                                let ready = start.elapsed().as_secs_f64();
                                if turn.chunks.is_empty() {
                                    turn.first_audio_at = ready;
                                }
                                turn.chunks.push(Chunk {
                                    sentence: index,
                                    ready,
                                    seconds: ((codes.len() - spoken) * SAMPLES_PER_CODE) as f64
                                        / f64::from(SAMPLE_RATE),
                                });
                                on_audio(&samples);
                                turn.audio.extend_from_slice(&samples);
                                spoken = codes.len();
                            }
                            Err(error) => failure = Some(error),
                        }
                    }
                })?;
        turn.speech_tokens += written.len();
        turn.speech_ids.push(written);
        failure.map_or(Ok(()), Err)
    }
    // ANCHOR_END: speak

    // ANCHOR: respond
    /// 사용자의 말(16kHz)을 듣고 말로 답한다. 소리 조각이 만들어질 때마다
    /// `on_audio`가 불린다.
    pub fn respond(
        &self,
        samples: &[f32],
        settings: &Settings,
        mut on_audio: impl FnMut(&[f32]),
    ) -> Result<Turn> {
        let start = Instant::now();
        let mut turn = Turn::default();

        // 1. 듣는다. 소리를 멜 스펙트로그램으로, 인코더로, 글로(33장).
        let mel = LogMel::WHISPER.spectrogram(samples);
        let encoded = self.whisper.encode(&mel.data)?;
        turn.encoded_at = start.elapsed().as_secs_f64();
        let transcript = self.whisper.greedy(
            &encoded,
            &self.vocabulary,
            &self.vocabulary.prompt("en")?,
            200,
        )?;
        turn.heard = self.vocabulary.decode(&transcript.tokens).trim().to_owned();
        turn.heard_at = start.elapsed().as_secs_f64();

        // 2. 답을 쓴다(2부). 문장이 끝날 때마다 3으로 넘긴다.
        let messages = [
            Message::new("system", &self.system),
            Message::new("user", &turn.heard),
        ];
        // 모델에 따라 글의 맨 앞에 시작 토큰이 붙는다(8.6절).
        let mut ids = self.chat_tokenizer.prefix().to_vec();
        ids.extend(
            self.chat_tokenizer
                .encode(&self.format.render(&messages, true), true),
        );
        let mut cache = KvCache::new(&self.chat.config);
        let mut decoder = StreamDecoder::default();
        let mut sentences = Sentences::default();
        let mut failure = None;
        generate(
            &self.chat,
            &ids,
            settings.answer_tokens,
            &mut cache,
            |token| {
                let now = start.elapsed().as_secs_f64();
                if turn.answer_tokens == 0 {
                    turn.first_token_at = now;
                }
                turn.answer_tokens += 1;
                turn.answer_ids.push(token);
                let piece = decoder.push(&self.chat_tokenizer, token);
                turn.answer.push_str(&piece);
                for sentence in sentences.push(&piece) {
                    if turn.sentences.is_empty() {
                        turn.first_sentence_at = now;
                    }
                    // 3. 소리로 바꾼다(37장). 문장마다라면 여기서 바로.
                    if settings.speak == Speak::BySentence && failure.is_none() {
                        let index = turn.sentences.len();
                        failure = self
                            .speak(index, &sentence, settings, start, &mut turn, &mut on_audio)
                            .err();
                    }
                    turn.sentences.push(sentence);
                }
            },
        )?;
        if let Some(error) = failure {
            return Err(error);
        }
        turn.answer.push_str(&decoder.finish());
        turn.answered_at = start.elapsed().as_secs_f64();
        let spoken = match settings.speak {
            Speak::BySentence => turn.sentences.len(),
            Speak::AfterAnswer => 0,
        };
        turn.sentences.extend(sentences.finish());
        if turn.first_sentence_at == 0.0 {
            turn.first_sentence_at = turn.answered_at;
        }
        // 아직 소리로 바꾸지 않은 문장들.
        for index in spoken..turn.sentences.len() {
            let sentence = turn.sentences[index].clone();
            self.speak(index, &sentence, settings, start, &mut turn, &mut on_audio)?;
        }
        turn.done_at = start.elapsed().as_secs_f64();
        Ok(turn)
    }
    // ANCHOR_END: respond
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sentences_come_out_when_the_next_one_starts() {
        let mut sentences = Sentences::default();
        assert!(sentences.push("Hello").is_empty());
        // 마침표만으로는 끝난 줄 모른다. 뒤에 빈칸이 와야 한다.
        assert!(sentences.push(" there.").is_empty());
        assert_eq!(sentences.push(" How"), ["Hello there."]);
        assert!(sentences.push(" are you? ").len() == 1);
        assert_eq!(sentences.finish(), None);
    }

    #[test]
    fn numbers_and_the_last_sentence_are_handled() {
        let mut sentences = Sentences::default();
        assert!(sentences.push("Pi is 3.14 or so").is_empty());
        assert_eq!(sentences.push("! Yes. No"), ["Pi is 3.14 or so!", "Yes."]);
        assert_eq!(sentences.finish(), Some("No".to_owned()));
        // 글자가 없는 조각은 문장이 아니다.
        let mut sentences = Sentences::default();
        assert!(sentences.push("... ").is_empty());
        assert_eq!(sentences.finish(), None);
    }
}
