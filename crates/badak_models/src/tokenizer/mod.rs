//! 토크나이저: 글을 토큰 번호로, 토큰 번호를 글로 (8장).
//!
//! 바이트 단위 BPE를 구현한다. GPT-2에서 시작해 Llama 3를 비롯한 많은
//! 언어 모델이 쓰는 방식이다.
//!
//! BERT 계열이 쓰는 WordPiece는 [`wordpiece`]에 있다(34장).

pub mod chat;
pub mod clip;
pub mod pretokenize;
pub mod unicode;
mod unicode_tables;
pub mod wordpiece;

use std::collections::HashMap;

use badak_formats::gguf::{Gguf, MetadataValue};
use badak_formats::json::{self, Value};
use badak_formats::{Error, Result};

pub use pretokenize::PreTokenizer;
use unicode::{byte_to_char, char_to_byte};

/// 토큰 번호.
pub type TokenId = u32;

/// GGUF의 `tokenizer.ggml.token_type`에서 쓰는 종류 번호.
const TOKEN_TYPE_CONTROL: u64 = 3;
const TOKEN_TYPE_USER_DEFINED: u64 = 4;

// ANCHOR: tokenizer
/// 바이트 단위 BPE 토크나이저.
pub struct Tokenizer {
    /// 번호에서 토큰의 표기로. 표기는 바이트를 글자로 옮겨 적은 문자열이다.
    tokens: Vec<String>,
    /// 표기에서 번호로.
    ids: HashMap<String, TokenId>,
    /// 병합 규칙. 이웃한 두 토큰의 번호에서 (우선순위, 합친 토큰의 번호)로.
    /// 우선순위는 작을수록 먼저 적용한다.
    merges: HashMap<(TokenId, TokenId), (u32, TokenId)>,
    /// 바이트 하나짜리 토큰의 번호. 병합의 출발점이다. 단어 사전에 그
    /// 바이트의 토큰이 없으면 `None`이다.
    byte_ids: [Option<TokenId>; 256],
    /// 글 속에서 통째로 찾아야 하는 특수 토큰. 긴 것부터 놓는다.
    specials: Vec<(String, TokenId)>,
    pre_tokenizer: PreTokenizer,
    /// 조각이 통째로 단어 사전에 있으면 병합을 건너뛰고 그 토큰을 쓴다.
    ignore_merges: bool,
    bos: Option<TokenId>,
    eos: Option<TokenId>,
    /// 글을 모델에 넣을 때 맨 앞에 붙이는 토큰들. 없을 수도 있다.
    prefix: Vec<TokenId>,
}
// ANCHOR_END: tokenizer

fn invalid(message: impl Into<String>) -> Error {
    Error::Parse {
        format: "토크나이저",
        offset: 0,
        message: message.into(),
    }
}

// ANCHOR: template_prefix
/// `post_processor`의 틀에서, 글 앞에 붙이는 특수 토큰들의 표기를 찾는다.
///
/// 틀은 `TemplateProcessing`이라는 항목의 `single`에 적혀 있다. 예를 들어
/// `[특수 토큰 <|begin_of_text|>, 글 A]`이면 글 앞에 그 토큰 하나를 붙이라는
/// 뜻이다. 틀이 없으면 아무것도 붙이지 않는다.
fn template_prefix(post_processor: Option<&Value>) -> Vec<&str> {
    let Some(processor) = post_processor else {
        return Vec::new();
    };
    let kind = processor.get("type").and_then(Value::as_str);
    match kind {
        // 여러 처리를 차례로 하는 경우: 그 안에서 틀을 찾는다.
        Some("Sequence") => processor
            .get("processors")
            .and_then(Value::as_array)
            .unwrap_or_default()
            .iter()
            .flat_map(|inner| template_prefix(Some(inner)))
            .collect(),
        Some("TemplateProcessing") => processor
            .get("single")
            .and_then(Value::as_array)
            .unwrap_or_default()
            .iter()
            // 글이 들어갈 자리가 나오기 전까지의 특수 토큰만 모은다.
            .map_while(|piece| piece.get("SpecialToken")?.get("id")?.as_str())
            .collect(),
        _ => Vec::new(),
    }
}
// ANCHOR_END: template_prefix

impl Tokenizer {
    // ANCHOR: from_gguf
    /// GGUF 파일의 메타데이터에서 토크나이저를 만든다.
    pub fn from_gguf(file: &Gguf) -> Result<Self> {
        let get = |key: &str| {
            file.metadata()
                .get(key)
                .ok_or_else(|| invalid(format!("메타데이터 {key}가 없습니다")))
        };
        let strings = |key: &str| -> Result<Vec<&str>> {
            get(key)?
                .as_array()
                .and_then(|items| items.iter().map(MetadataValue::as_str).collect())
                .ok_or_else(|| invalid(format!("{key}가 문자열의 배열이 아닙니다")))
        };

        let model = get("tokenizer.ggml.model")?.as_str().unwrap_or_default();
        if model != "gpt2" {
            return Err(Error::Unsupported(format!("토크나이저 종류 {model:?}")));
        }
        let pre = get("tokenizer.ggml.pre")?.as_str().unwrap_or_default();
        let pre_tokenizer = PreTokenizer::from_name(pre)
            .ok_or_else(|| Error::Unsupported(format!("사전 분할 규칙 {pre:?}")))?;

        let tokens: Vec<String> = strings("tokenizer.ggml.tokens")?
            .into_iter()
            .map(String::from)
            .collect();
        // 병합 규칙: "왼쪽 오른쪽" 꼴의 문자열이 우선순위 순서로 적혀 있다.
        let merges = strings("tokenizer.ggml.merges")?
            .into_iter()
            .map(|rule| {
                rule.split_once(' ')
                    .ok_or_else(|| invalid(format!("병합 규칙 {rule:?}에 공백이 없습니다")))
            })
            .collect::<Result<Vec<_>>>()?;

        // 특수 토큰: 종류가 제어 또는 사용자 정의인 토큰.
        let specials: Vec<TokenId> = match file.metadata().get("tokenizer.ggml.token_type") {
            Some(types) => types
                .as_array()
                .unwrap_or_default()
                .iter()
                .enumerate()
                .filter(|(_, kind)| {
                    matches!(
                        kind.as_u64(),
                        Some(TOKEN_TYPE_CONTROL | TOKEN_TYPE_USER_DEFINED)
                    )
                })
                .map(|(id, _)| id as TokenId)
                .collect(),
            None => Vec::new(),
        };

        let token_id = |key: &str| {
            file.metadata()
                .get(key)
                .and_then(MetadataValue::as_u64)
                .map(|id| id as TokenId)
        };
        let mut tokenizer = Self::build(
            tokens,
            &merges,
            &specials,
            pre_tokenizer,
            // Llama 3의 토크나이저 설정이 이 옵션을 켜 두었다. GGUF에는 적혀
            // 있지 않아 사전 분할 규칙의 이름으로 판단한다.
            pre_tokenizer == PreTokenizer::Llama3,
        )?;
        tokenizer.bos = token_id("tokenizer.ggml.bos_token_id");
        tokenizer.eos = token_id("tokenizer.ggml.eos_token_id");
        Ok(tokenizer)
    }
    // ANCHOR_END: from_gguf

    // ANCHOR: build
    /// 단어 사전과 병합 규칙에서 토크나이저를 조립한다.
    ///
    /// `tokens`는 번호 순서대로 적힌 표기, `merges`는 우선순위 순서대로 적힌
    /// (왼쪽, 오른쪽) 쌍, `specials`는 특수 토큰의 번호다.
    fn build(
        tokens: Vec<String>,
        merges: &[(&str, &str)],
        specials: &[TokenId],
        pre_tokenizer: PreTokenizer,
        ignore_merges: bool,
    ) -> Result<Self> {
        let ids: HashMap<String, TokenId> = tokens
            .iter()
            .enumerate()
            .map(|(id, token)| (token.clone(), id as TokenId))
            .collect();
        let id_of = |token: &str| {
            ids.get(token)
                .copied()
                .ok_or_else(|| invalid(format!("토큰 {token:?}이 단어 사전에 없습니다")))
        };

        // 표기를 번호로 바꿔 둔다. 병합할 때 문자열을 비교하지 않게 된다.
        let mut merge_table = HashMap::with_capacity(merges.len());
        for (rank, &(left, right)) in merges.iter().enumerate() {
            let merged = id_of(&format!("{left}{right}"))?;
            merge_table.insert((id_of(left)?, id_of(right)?), (rank as u32, merged));
        }

        // 바이트 하나짜리 토큰들. 올바른 글에 나올 수 없는 바이트나 드문
        // 제어 문자의 토큰을 빼 둔 단어 사전도 있어서, 없는 것은 비워 둔다.
        let mut byte_ids = [None; 256];
        for byte in 0..=255_u8 {
            byte_ids[usize::from(byte)] = ids.get(&byte_to_char(byte).to_string()).copied();
        }

        let mut specials: Vec<(String, TokenId)> = specials
            .iter()
            .filter_map(|&id| Some((tokens.get(id as usize)?.clone(), id)))
            .collect();
        specials.sort_by_key(|(text, _)| std::cmp::Reverse(text.len()));

        Ok(Self {
            tokens,
            ids,
            merges: merge_table,
            byte_ids,
            specials,
            pre_tokenizer,
            ignore_merges,
            bos: None,
            eos: None,
            prefix: Vec::new(),
        })
    }
    // ANCHOR_END: build

    // ANCHOR: from_tokenizer_json
    /// Hugging Face의 `tokenizer.json`에서 토크나이저를 만든다.
    ///
    /// 모델을 만든 쪽이 배포하는 원본 설정이다. GGUF와 달리 사전 분할 규칙과
    /// `ignore_merges` 설정이 파일에 그대로 적혀 있다.
    pub fn from_tokenizer_json(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let root = json::parse(&std::fs::read_to_string(path)?)?;
        let model = root
            .get("model")
            .ok_or_else(|| invalid("model이 없습니다"))?;
        let kind = model
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if kind != "BPE" {
            return Err(Error::Unsupported(format!("토크나이저 종류 {kind:?}")));
        }

        // 단어 사전: 표기에서 번호로 가는 객체. 번호 순서의 목록으로 뒤집는다.
        let vocab = model
            .get("vocab")
            .and_then(Value::as_object)
            .ok_or_else(|| invalid("model.vocab이 없습니다"))?;
        let added = root
            .get("added_tokens")
            .and_then(Value::as_array)
            .unwrap_or_default();
        let mut entries: Vec<(usize, &str)> = Vec::with_capacity(vocab.len() + added.len());
        for (token, id) in vocab {
            let id = id
                .as_usize()
                .ok_or_else(|| invalid("토큰 번호가 정수가 아닙니다"))?;
            entries.push((id, token));
        }
        // 덧붙인 토큰: 특수 토큰이 여기에 있다. 단어 사전에 없는 번호일 수도 있다.
        let mut specials = Vec::new();
        for token in added {
            let id = token.get("id").and_then(Value::as_usize);
            let content = token.get("content").and_then(Value::as_str);
            let (Some(id), Some(content)) = (id, content) else {
                return Err(invalid("added_tokens의 항목이 올바르지 않습니다"));
            };
            entries.push((id, content));
            specials.push(id as TokenId);
        }
        let size = entries.iter().map(|&(id, _)| id + 1).max().unwrap_or(0);
        if size > entries.len() {
            return Err(invalid("단어 사전의 번호에 빈 곳이 있습니다"));
        }
        let mut tokens = vec![String::new(); size];
        for (id, token) in entries {
            tokens[id] = token.to_string();
        }

        // 병합 규칙: "왼쪽 오른쪽" 문자열이거나 [왼쪽, 오른쪽] 배열이다.
        let merges = model
            .get("merges")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid("model.merges가 없습니다"))?
            .iter()
            .map(|rule| match rule {
                Value::String(text) => text.split_once(' '),
                Value::Array(pair) => match pair.as_slice() {
                    [left, right] => left.as_str().zip(right.as_str()),
                    _ => None,
                },
                _ => None,
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| invalid("병합 규칙의 모양이 올바르지 않습니다"))?;

        let pre_tokenizer = PreTokenizer::from_tokenizer_json(root.get("pre_tokenizer"))?;
        let ignore_merges = model.get("ignore_merges") == Some(&Value::Bool(true));
        let mut tokenizer = Self::build(tokens, &merges, &specials, pre_tokenizer, ignore_merges)?;
        tokenizer.prefix = template_prefix(root.get("post_processor"))
            .iter()
            .map(|token| {
                tokenizer
                    .token_id(token)
                    .ok_or_else(|| invalid(format!("토큰 {token:?}이 단어 사전에 없습니다")))
            })
            .collect::<Result<_>>()?;
        Ok(tokenizer)
    }
    // ANCHOR_END: from_tokenizer_json

    /// 글을 모델에 넣을 때 맨 앞에 붙이는 토큰들.
    ///
    /// `tokenizer.json`의 `post_processor`에 적힌 틀에서 읽은 것이다. Llama
    /// 3.2는 글의 시작을 알리는 토큰 하나를 붙이고, SmolLM2는 아무것도 붙이지
    /// 않는다.
    pub fn prefix(&self) -> &[TokenId] {
        &self.prefix
    }

    /// 문장의 시작과 끝을 알리는 토큰을 정한다. 모델의 설정 파일에서 읽은
    /// 값을 넣을 때 쓴다.
    pub fn with_bos_eos(mut self, bos: Option<TokenId>, eos: Option<TokenId>) -> Self {
        (self.bos, self.eos) = (bos, eos);
        self
    }

    /// 단어 사전의 크기.
    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    /// 병합 규칙의 수.
    pub fn merge_count(&self) -> usize {
        self.merges.len()
    }

    /// 문장의 시작을 알리는 토큰.
    pub fn bos(&self) -> Option<TokenId> {
        self.bos
    }

    /// 문장의 끝을 알리는 토큰.
    pub fn eos(&self) -> Option<TokenId> {
        self.eos
    }

    /// 이 토크나이저의 사전 분할 규칙.
    pub fn pre_tokenizer(&self) -> PreTokenizer {
        self.pre_tokenizer
    }

    /// 특수 토큰인가. 글의 내용이 아니라 구조를 나타내는 토큰이다 (41장).
    pub fn is_special(&self, id: TokenId) -> bool {
        self.specials.iter().any(|(_, special)| *special == id)
    }

    /// 표기로 토큰 번호를 찾는다. 특수 토큰의 번호를 얻을 때 쓴다.
    pub fn token_id(&self, text: &str) -> Option<TokenId> {
        self.ids.get(text).copied()
    }

    /// 토큰의 표기. 바이트를 글자로 옮겨 적은 문자열이다.
    pub fn token_text(&self, id: TokenId) -> Option<&str> {
        self.tokens.get(id as usize).map(String::as_str)
    }

    // ANCHOR: bpe
    /// 조각 하나에 BPE를 적용한다.
    ///
    /// 조각을 바이트 단위 토큰으로 펼친 뒤, 이웃한 쌍 가운데 우선순위가
    /// 가장 높은 것을 찾아 합치기를 더 합칠 쌍이 없을 때까지 되풀이한다.
    ///
    /// 모델에 따라서는 조각이 통째로 단어 사전에 있으면 병합을 거치지 않고
    /// 그 토큰을 바로 쓴다.
    pub fn bpe(&self, piece: &str) -> Vec<TokenId> {
        if self.ignore_merges {
            let whole: String = piece.bytes().map(byte_to_char).collect();
            if let Some(&id) = self.ids.get(&whole) {
                return vec![id];
            }
        }
        let mut symbols = self.byte_tokens(piece);
        while let Some(next) = self.merge_once(&symbols) {
            symbols = next;
        }
        symbols
    }

    /// 조각을 바이트 하나짜리 토큰들로 펼친다. 병합의 출발점이다.
    ///
    /// 단어 사전에 토큰이 없는 바이트는 건너뛴다. 기준 구현과 같은 동작이다.
    fn byte_tokens(&self, piece: &str) -> Vec<TokenId> {
        piece
            .bytes()
            .filter_map(|byte| self.byte_ids[usize::from(byte)])
            .collect()
    }

    /// 병합을 한 번 한다. 합칠 쌍이 없으면 `None`.
    fn merge_once(&self, symbols: &[TokenId]) -> Option<Vec<TokenId>> {
        // 우선순위가 가장 높은(숫자가 가장 작은) 쌍을 찾는다.
        let &(rank, merged) = symbols
            .windows(2)
            .filter_map(|pair| self.merges.get(&(pair[0], pair[1])))
            .min_by_key(|&&(rank, _)| rank)?;

        // 그 쌍이 나오는 자리를 모두 합친다.
        let mut next = Vec::with_capacity(symbols.len());
        let mut index = 0;
        while index < symbols.len() {
            let pair = symbols.get(index + 1).map(|&right| (symbols[index], right));
            if pair.and_then(|pair| self.merges.get(&pair)) == Some(&(rank, merged)) {
                next.push(merged);
                index += 2;
            } else {
                next.push(symbols[index]);
                index += 1;
            }
        }
        Some(next)
    }
    // ANCHOR_END: bpe

    /// 병합이 진행되는 모습을 단계별로 돌려준다. 설명과 디버깅에 쓴다.
    ///
    /// 첫 원소는 바이트 단위로 펼친 상태이고, 마지막 원소가 최종 결과다.
    pub fn bpe_steps(&self, piece: &str) -> Vec<Vec<TokenId>> {
        let mut steps = vec![self.byte_tokens(piece)];
        while let Some(next) = self.merge_once(steps.last().expect("하나 이상 있다")) {
            steps.push(next);
        }
        steps
    }

    // ANCHOR: encode
    /// 글을 토큰 번호로 바꾼다.
    ///
    /// `parse_special`이 참이면 글 속의 특수 토큰 표기(`<|eot_id|>` 등)를
    /// 토큰 하나로 바꾼다. 사용자가 입력한 글을 다룰 때는 거짓으로 두어,
    /// 그런 글자가 보통의 글로 처리되게 한다.
    pub fn encode(&self, text: &str, parse_special: bool) -> Vec<TokenId> {
        let mut ids = Vec::new();
        self.encode_with(text, parse_special, &mut ids, &mut |piece| self.bpe(piece));
        ids
    }

    fn encode_with(
        &self,
        text: &str,
        parse_special: bool,
        ids: &mut Vec<TokenId>,
        bpe: &mut dyn FnMut(&str) -> Vec<TokenId>,
    ) {
        let mut rest = text;
        while !rest.is_empty() {
            // 다음 특수 토큰의 위치를 찾는다. 없으면 남은 글 전체가 보통의 글이다.
            let special = parse_special.then(|| self.find_special(rest)).flatten();
            let (plain, tail) = match special {
                Some((start, _, _)) => rest.split_at(start),
                None => (rest, ""),
            };
            for piece in self.pre_tokenizer.split(plain) {
                ids.extend(bpe(piece));
            }
            rest = match special {
                Some((_, len, id)) => {
                    ids.push(id);
                    &tail[len..]
                }
                None => "",
            };
        }
    }

    /// 글에서 가장 먼저 나오는 특수 토큰의 (위치, 길이, 번호)를 찾는다.
    fn find_special(&self, text: &str) -> Option<(usize, usize, TokenId)> {
        text.char_indices().find_map(|(start, _)| {
            self.specials
                .iter()
                .find(|(special, _)| text[start..].starts_with(special.as_str()))
                .map(|(special, id)| (start, special.len(), *id))
        })
    }
    // ANCHOR_END: encode

    // ANCHOR: encode_cached
    /// [`encode`](Self::encode)와 같은 결과를 내되, 같은 조각을 다시 만나면
    /// 앞서 계산한 결과를 `cache`에서 꺼내 쓴다.
    pub fn encode_cached(
        &self,
        text: &str,
        parse_special: bool,
        cache: &mut HashMap<String, Vec<TokenId>>,
    ) -> Vec<TokenId> {
        let mut ids = Vec::new();
        self.encode_with(text, parse_special, &mut ids, &mut |piece| {
            if let Some(known) = cache.get(piece) {
                return known.clone();
            }
            let computed = self.bpe(piece);
            cache.insert(piece.to_string(), computed.clone());
            computed
        });
        ids
    }
    // ANCHOR_END: encode_cached

    // ANCHOR: decode
    /// 토큰 하나가 나타내는 바이트들.
    ///
    /// 보통의 토큰은 표기의 글자마다 바이트 하나로 되돌린다. 특수 토큰은
    /// 그런 표기가 아니므로 표기 그대로의 UTF-8 바이트를 돌려준다.
    pub fn token_bytes(&self, id: TokenId) -> Vec<u8> {
        let Some(text) = self.token_text(id) else {
            return Vec::new();
        };
        text.chars()
            .map(char_to_byte)
            .collect::<Option<Vec<u8>>>()
            .unwrap_or_else(|| text.as_bytes().to_vec())
    }

    /// 토큰 번호들을 글로 되돌린다.
    ///
    /// 토큰의 경계가 글자의 경계와 같다는 보장이 없으므로, 모든 토큰의
    /// 바이트를 이어 붙인 뒤에 한꺼번에 글로 해석한다. 올바른 UTF-8이
    /// 아닌 부분은 대체 문자(U+FFFD)가 된다.
    pub fn decode(&self, ids: &[TokenId]) -> String {
        let bytes: Vec<u8> = ids.iter().flat_map(|&id| self.token_bytes(id)).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
    // ANCHOR_END: decode
}

// ANCHOR: stream
/// 토큰을 하나씩 받아 글로 되돌리는 도구.
///
/// 글자 하나가 여러 토큰에 걸쳐 있을 수 있다. 글자가 완성될 때까지
/// 바이트를 모아 두었다가, 완성된 만큼만 내놓는다.
#[derive(Debug, Default)]
pub struct StreamDecoder {
    pending: Vec<u8>,
}

impl StreamDecoder {
    /// 토큰 하나를 받아, 이번에 새로 완성된 글을 돌려준다.
    pub fn push(&mut self, tokenizer: &Tokenizer, id: TokenId) -> String {
        self.pending.extend(tokenizer.token_bytes(id));
        let complete = match std::str::from_utf8(&self.pending) {
            Ok(_) => self.pending.len(),
            // 끝이 잘린 글자라면 그 앞까지만 내놓고 나머지는 기다린다.
            Err(error) if error.error_len().is_none() => error.valid_up_to(),
            // 어떻게 이어져도 글자가 될 수 없는 바이트라면 더 기다리지 않는다.
            Err(_) => self.pending.len(),
        };
        let ready: Vec<u8> = self.pending.drain(..complete).collect();
        String::from_utf8_lossy(&ready).into_owned()
    }

    /// 더 올 토큰이 없을 때, 남아 있는 바이트를 글로 내놓는다.
    pub fn finish(&mut self) -> String {
        String::from_utf8_lossy(&std::mem::take(&mut self.pending)).into_owned()
    }
}
// ANCHOR_END: stream
