//! 서버가 쥐고 있는 모델과, 요청 사이에 남겨 두는 KV 캐시 (39장).

use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use badak_formats::{Error, Result};
use badak_models::constrain::{Constraint, Grammar, Search, Vocabulary};
use badak_models::llama::cache::KvCache;
use badak_models::llama::model::LlamaModel;
use badak_models::sampling::{Sampler, SamplerConfig};
use badak_models::tokenizer::chat::{ChatTemplate, Message};
use badak_models::tokenizer::{StreamDecoder, TokenId, Tokenizer};

fn invalid(message: String) -> Error {
    Error::Parse {
        format: "서버",
        offset: 0,
        message,
    }
}

/// 요청 하나의 설정.
#[derive(Debug, Clone, PartialEq)]
pub struct Options {
    /// 쓸 토큰의 한도.
    pub max_tokens: usize,
    /// 온도. 0이면 점수가 가장 높은 토큰을 고른다.
    pub temperature: f32,
    /// 누적 확률의 한도.
    pub top_p: f32,
    /// 난수의 시드.
    pub seed: u64,
    /// 앞 요청의 KV 캐시에서 같은 앞부분을 다시 쓸지.
    pub reuse: bool,
    /// 답이 따라야 하는 문법 (41장). 있으면 그 꼴의 JSON만 쓴다.
    pub grammar: Option<Arc<Grammar>>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 1.0,
            top_p: 1.0,
            seed: 0,
            reuse: true,
            grammar: None,
        }
    }
}

/// 글이 끝난 까닭.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Finish {
    /// 모델이 끝 토큰을 썼다.
    Stop,
    /// 토큰의 한도에 닿았다.
    Length,
    /// 받는 쪽이 연결을 끊었다.
    Cancelled,
}

/// 요청 하나의 결과.
#[derive(Debug, Clone, PartialEq)]
pub struct Completion {
    /// 쓴 글.
    pub text: String,
    /// 읽은 글의 토큰 수.
    pub prompt_tokens: usize,
    /// 그 가운데 앞 요청의 캐시에서 다시 쓴 토큰 수.
    pub cached_tokens: usize,
    /// 쓴 토큰 수.
    pub completion_tokens: usize,
    /// 글이 끝난 까닭.
    pub finish: Finish,
    /// 대화를 토큰으로 바꾸는 데 걸린 시간(초).
    pub tokenize: f64,
    /// 앞 요청이 끝나기를 기다린 시간(초).
    pub queued: f64,
    /// 차례가 온 때부터 첫 토큰까지(초).
    pub first_token: f64,
    /// 차례가 온 때부터 끝까지(초).
    pub total: f64,
}

// ANCHOR: session
/// 요청 사이에 남겨 두는 것. 마지막 요청이 KV 캐시에 넣은 토큰들과 그
/// 캐시다.
struct Session {
    tokens: Vec<TokenId>,
    cache: KvCache,
}

/// 모델 하나를 쥐고 요청을 차례로 처리한다.
pub struct Engine {
    /// 모델의 이름. `/v1/models`가 알려 준다.
    pub name: String,
    /// 읽는 글과 쓰는 글을 더한 길이의 한도(토큰). KV 캐시의 메모리가
    /// 이것에 비례하므로, 요청이 정하게 두지 않고 서버가 정한다.
    pub context: usize,
    // 40장: 모델과 토크나이저를 스케줄러의 스레드와 함께 쥔다.
    model: Arc<LlamaModel>,
    tokenizer: Arc<Tokenizer>,
    /// 모델의 대화 틀. 시스템의 말이 없는 요청에 넣을 기본 지시문까지 안다(47.3절).
    template: ChatTemplate,
    /// 토큰마다의 바이트들 (41장). 문법에 묶는 요청이 처음 올 때 만든다.
    vocabulary: OnceLock<Arc<Vocabulary>>,
    /// 한 번에 한 요청만 모델을 쓴다. 나머지는 여기서 기다린다.
    session: Mutex<Session>,
}
// ANCHOR_END: session

/// 두 토큰열이 앞에서부터 몇 개 같은가.
pub fn common_prefix(a: &[TokenId], b: &[TokenId]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// 파일의 맨 앞에 적는 표시.
const MAGIC: &[u8; 8] = b"badakSS1";

impl Engine {
    /// 모델 디렉터리(`config.json`, `model.safetensors`, `tokenizer.json`)를
    /// 읽는다. 대화의 형식은 디렉터리의 이름으로 정한다.
    pub fn load(directory: impl AsRef<Path>) -> Result<Self> {
        let directory = directory.as_ref();
        let name = directory.file_name().map_or_else(
            || "model".to_owned(),
            |name| name.to_string_lossy().into_owned(),
        );
        let model = Arc::new(LlamaModel::load(directory)?);
        let tokenizer = Tokenizer::from_tokenizer_json(directory.join("tokenizer.json"))?;
        let session = Session {
            tokens: Vec::new(),
            cache: KvCache::new(&model.config),
        };
        Ok(Self {
            template: ChatTemplate::load(directory, &tokenizer),
            tokenizer: Arc::new(tokenizer),
            session: Mutex::new(session),
            vocabulary: OnceLock::new(),
            context: 4096,
            model,
            name,
        })
    }

    // ANCHOR: warm
    /// 모델을 한 번 돌려 둔다. 걸린 시간(초)을 돌려준다.
    ///
    /// 가중치는 파일을 메모리에 비춘 것이어서(20장) 처음 닿을 때에야
    /// 실제로 올라온다. 그 값을 첫 요청이 치르지 않게, 요청을 받기 전에
    /// 토큰 몇 개를 넣어 모든 층을 한 번 지나가 둔다. 스레드들도 이때
    /// 깨어난다.
    pub fn warm(&self) -> Result<f64> {
        let start = Instant::now();
        let ids = self.encode(&[Message::new("user", "Hello")]);
        let mut cache = KvCache::new(&self.model.config);
        self.model.forward_cached(&ids, &mut cache)?;
        Ok(start.elapsed().as_secs_f64())
    }
    // ANCHOR_END: warm

    /// 모델과 토크나이저. 스케줄러(40장)가 제 스레드에서 쓴다.
    pub fn parts(&self) -> (Arc<LlamaModel>, Arc<Tokenizer>) {
        (Arc::clone(&self.model), Arc::clone(&self.tokenizer))
    }

    /// 문법에 묶인 글 하나의 상태를 만든다 (41장). 점수가 가장 높은
    /// 토큰을 고를 때는 그 토큰부터 보고, 확률로 뽑을 때는 올 수 있는
    /// 토큰을 모두 찾는다.
    pub fn constraint(&self, options: &Options) -> Option<Constraint> {
        let grammar = options.grammar.as_ref()?;
        let vocabulary = self
            .vocabulary
            .get_or_init(|| Arc::new(Vocabulary::new(&self.tokenizer)));
        let search = if options.temperature <= 0.0 {
            Search::Best
        } else {
            Search::Trie
        };
        Some(Constraint::new(vocabulary, grammar, search))
    }

    /// 대화를 토큰으로 바꾼다. 틀의 뼈대와 말의 내용을 따로 토큰화한다.
    pub fn encode(&self, messages: &[Message]) -> Vec<TokenId> {
        self.template.encode(&self.tokenizer, messages, true)
    }

    // ANCHOR: complete
    /// 대화에 답을 쓴다. 글 조각이 나올 때마다 `on_text`를 부르고, 그것이
    /// 거짓을 돌려주면(받는 쪽이 연결을 끊었으면) 그만 쓴다.
    pub fn complete(
        &self,
        messages: &[Message],
        options: &Options,
        on_text: &mut dyn FnMut(&str) -> bool,
    ) -> Result<Completion> {
        let arrived = Instant::now();
        let ids = self.encode(messages);
        let tokenize = arrived.elapsed().as_secs_f64();
        if ids.len() + options.max_tokens > self.context {
            return Err(Error::Unsupported(format!(
                "읽을 글({}토큰)과 쓸 글({}토큰)을 더하면 한도({}토큰)를 넘습니다",
                ids.len(),
                options.max_tokens,
                self.context
            )));
        }

        // 1. 차례를 기다린다. 모델은 하나이고 한 번에 한 요청만 쓴다.
        let waiting = Instant::now();
        let mut session = self
            .session
            .lock()
            .map_err(|_| invalid("앞 요청이 공황에 빠졌습니다".into()))?;
        let queued = waiting.elapsed().as_secs_f64();
        let start = Instant::now();
        let session = &mut *session;

        // 2. 앞 요청과 같은 앞부분은 KV 캐시에 그대로 둔다(접두사 캐시).
        //    마지막 토큰 하나는 늘 다시 계산한다. 그래야 다음 토큰의 점수가
        //    나온다.
        let common = if options.reuse {
            common_prefix(&session.tokens, &ids).min(ids.len().saturating_sub(1))
        } else {
            0
        };
        session.cache.truncate(common);
        session.tokens.truncate(common);

        // 3. 나머지를 읽고 답을 쓴다(11장, 12장).
        let mut sampler = Sampler::new(SamplerConfig {
            temperature: options.temperature,
            top_p: options.top_p,
            seed: options.seed,
            ..SamplerConfig::default()
        });
        let mut history = ids.clone();
        let mut input = ids[common..].to_vec();
        let mut decoder = StreamDecoder::default();
        let (mut text, mut written) = (String::new(), 0);
        let (mut first_token, mut finish) = (None, Finish::Length);
        session.cache.reserve(ids.len() + options.max_tokens);
        let mut constraint = self.constraint(options);
        let end = &self.model.config.eos_token_ids;
        while written < options.max_tokens {
            let logits = self.model.forward_cached(&input, &mut session.cache)?;
            session.tokens.extend_from_slice(&input);
            // 41장: 문법에 묶였으면 지금 올 수 없는 토큰의 점수를 지운다.
            let mut logits = logits.as_slice::<f32>()?.to_vec();
            if let Some(constraint) = &mut constraint {
                constraint.mask(&mut logits, end);
            }
            let next = sampler.sample(&logits, &history);
            first_token.get_or_insert_with(|| start.elapsed().as_secs_f64());
            if end.contains(&next) {
                finish = Finish::Stop;
                break;
            }
            if let Some(constraint) = &mut constraint {
                constraint.accept(next);
            }
            history.push(next);
            written += 1;
            let piece = decoder.push(&self.tokenizer, next);
            text.push_str(&piece);
            if !piece.is_empty() && !on_text(&piece) {
                finish = Finish::Cancelled;
                break;
            }
            // 문법이 끝났으면 더 쓸 것이 없다.
            if constraint.as_ref().is_some_and(Constraint::finished) {
                finish = Finish::Stop;
                break;
            }
            input = vec![next];
        }
        let rest = decoder.finish();
        if !rest.is_empty() {
            text.push_str(&rest);
            on_text(&rest);
        }
        let total = start.elapsed().as_secs_f64();
        Ok(Completion {
            text,
            prompt_tokens: ids.len(),
            cached_tokens: common,
            completion_tokens: written,
            finish,
            tokenize,
            queued,
            first_token: first_token.unwrap_or(total),
            total,
        })
    }
    // ANCHOR_END: complete

    /// 캐시에 든 토큰의 수와 캐시가 차지하는 메모리(바이트).
    pub fn cached(&self) -> (usize, usize) {
        self.session.lock().map_or((0, 0), |session| {
            (session.tokens.len(), session.cache.size_in_bytes())
        })
    }

    // ANCHOR: persist
    /// 지금의 캐시를 파일에 쓴다. 쓴 바이트의 수를 돌려준다.
    ///
    /// 파일에는 토큰들과 KV 캐시가 함께 들어간다. 토큰들이 있어야 되살린
    /// 뒤에 "새 요청과 어디까지 같은가"를 견줄 수 있다.
    pub fn save(&self, path: &Path) -> Result<usize> {
        let session = self
            .session
            .lock()
            .map_err(|_| invalid("앞 요청이 공황에 빠졌습니다".into()))?;
        let mut bytes = MAGIC.to_vec();
        bytes.extend_from_slice(&(session.tokens.len() as u32).to_le_bytes());
        for &token in &session.tokens {
            bytes.extend_from_slice(&token.to_le_bytes());
        }
        bytes.extend_from_slice(&session.cache.to_bytes());
        std::fs::write(path, &bytes)?;
        Ok(bytes.len())
    }

    /// 파일에서 캐시를 되살린다. 되살린 토큰의 수를 돌려준다.
    pub fn restore(&self, path: &Path) -> Result<usize> {
        let bytes = std::fs::read(path)?;
        let broken = || invalid("캐시 파일이 이 모델의 것이 아니거나 깨졌습니다".into());
        let (magic, rest) = bytes.split_at_checked(MAGIC.len()).ok_or_else(broken)?;
        let (count, rest) = rest.split_at_checked(4).ok_or_else(broken)?;
        let count = u32::from_le_bytes(count.try_into().map_err(|_| broken())?) as usize;
        let (tokens, cache) = rest
            .split_at_checked(count.checked_mul(4).ok_or_else(broken)?)
            .ok_or_else(broken)?;
        let cache = KvCache::from_bytes(&self.model.config, cache).ok_or_else(broken)?;
        if magic != MAGIC || cache.len() != count {
            return Err(broken());
        }
        let tokens = tokens
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| TokenId::from_le_bytes(*chunk))
            .collect();
        let mut session = self
            .session
            .lock()
            .map_err(|_| invalid("앞 요청이 공황에 빠졌습니다".into()))?;
        *session = Session { tokens, cache };
        Ok(count)
    }
    // ANCHOR_END: persist
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_the_shared_prefix() {
        assert_eq!(common_prefix(&[1, 2, 3, 4], &[1, 2, 9, 4]), 2);
        assert_eq!(common_prefix(&[1, 2], &[1, 2, 3]), 2);
        assert_eq!(common_prefix(&[], &[1]), 0);
        assert_eq!(common_prefix(&[5], &[6]), 0);
    }
}
