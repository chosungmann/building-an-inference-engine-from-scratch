//! 언어 모델. 문 안쪽의 보통 Rust다 (47장).

use std::fmt;
use std::path::Path;

use badak_models::llama::WeightFormat;
use badak_models::llama::cache::KvCache;
use badak_models::llama::model::LlamaModel;
use badak_models::tokenizer::chat::{ChatTemplate, Message};
use badak_models::tokenizer::{StreamDecoder, Tokenizer};

use crate::classifier::{Error, Status};

impl From<badak_core::Error> for Error {
    fn from(error: badak_core::Error) -> Self {
        Self {
            status: Status::Model,
            message: error.to_string(),
        }
    }
}

/// 모델 하나와 토크나이저. 한 번에 한 스레드만 써야 한다.
pub struct Llm {
    model: LlamaModel,
    tokenizer: Tokenizer,
    template: ChatTemplate,
}

impl fmt::Debug for Llm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Llm")
            .field("template", &self.template)
            .field("layers", &self.model.config.num_layers)
            .finish()
    }
}

impl Llm {
    /// 안에 든 모델.
    pub fn model(&self) -> &LlamaModel {
        &self.model
    }

    /// 안에 든 모델을 고친다. 선형 층을 GPU에 맡길 때 쓴다 (49장).
    pub fn model_mut(&mut self) -> &mut LlamaModel {
        &mut self.model
    }

    /// 모델 폴더(`config.json`, `model.safetensors`, `tokenizer.json`)를 연다.
    ///
    /// `weights`는 가중치를 메모리에 두는 방식이다. 0은 파일에 적힌 그대로
    /// (파일을 메모리에 비춘다), 1은 `Q8_0`, 2는 `Q4_0`으로 읽으면서 줄인다.
    pub fn open(directory: impl AsRef<Path>, weights: i32) -> Result<Self, Error> {
        let directory = directory.as_ref();
        let format = match weights {
            0 => WeightFormat::Stored,
            1 => WeightFormat::Q8_0,
            2 => WeightFormat::Q4_0,
            other => return Err(Error::argument(format!("모르는 가중치 방식 {other}"))),
        };
        let model = LlamaModel::load_as(directory, format)?;
        let tokenizer = Tokenizer::from_tokenizer_json(directory.join("tokenizer.json"))?;
        // 대화의 틀: 뼈대는 토크나이저의 특수 토큰으로, 기본 지시문은 모델과
        // 함께 온 설정 파일에서 알아낸다(47.3절).
        let chat = ChatTemplate::load(directory, &tokenizer);
        Ok(Self {
            model,
            tokenizer,
            template: chat,
        })
    }

    // ANCHOR: chat
    /// 질문 하나에 답한다. 글이 완성되는 대로 `on_piece`에 넘기고, 그것이
    /// 거짓을 돌려주면 그만 쓴다. 쓴 토큰의 수를 돌려준다.
    ///
    /// 토큰 하나가 글자 하나가 아니다. 한글 한 글자가 토큰 둘에 걸칠 수
    /// 있다. 글자가 완성될 때까지 모았다가 넘기므로, `on_piece`가 받는
    /// 조각은 늘 온전한 UTF-8이다.
    ///
    /// `system`은 대화의 맨 앞에 놓는 지시문이다. 주지 않으면 모델의 틀에
    /// 적힌 기본 지시문을 쓴다. 모델은 그 줄이 있는 대화로 학습되었고, 빼면
    /// 답이 달라진다(47장).
    pub fn chat(
        &self,
        system: Option<&str>,
        question: &str,
        max_tokens: usize,
        mut on_piece: impl FnMut(&str) -> bool,
    ) -> Result<usize, Error> {
        let mut messages = Vec::new();
        if let Some(system) = system {
            messages.push(Message::new("system", system));
        }
        messages.push(Message::new("user", question));
        let prompt = self.template.encode(&self.tokenizer, &messages, true);
        let mut cache = KvCache::new(&self.model.config);
        cache.reserve(prompt.len() + max_tokens);
        let mut decoder = StreamDecoder::default();
        let mut input = prompt;
        let mut written = 0;
        while written < max_tokens {
            let logits = self.model.forward_cached(&input, &mut cache)?;
            let scores = logits.as_slice::<f32>()?;
            // 점수가 가장 높은 토큰. 같은 점수면 번호가 작은 쪽이다(11장).
            let mut next = 0;
            for (index, &score) in scores.iter().enumerate() {
                if score > scores[next] {
                    next = index;
                }
            }
            let next = next as u32;
            if self.model.config.eos_token_ids.contains(&next) || self.tokenizer.is_special(next) {
                break;
            }
            written += 1;
            let piece = decoder.push(&self.tokenizer, next);
            if !piece.is_empty() && !on_piece(&piece) {
                return Ok(written);
            }
            input = vec![next];
        }
        let rest = decoder.finish();
        if !rest.is_empty() {
            on_piece(&rest);
        }
        Ok(written)
    }
    // ANCHOR_END: chat
}
