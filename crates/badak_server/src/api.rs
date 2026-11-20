//! OpenAI의 대화 API와 같은 꼴의 요청과 응답 (39장).
//!
//! 요청은 JSON이고(`model`, `messages`, `max_tokens`, `temperature`,
//! `stream`, …), 응답도 JSON이다. `stream`이 참이면 응답이 조각조각 온다.
//! 조각 하나는 `data: {JSON}` 한 줄과 빈 줄이고(서버가 보내는 사건,
//! server-sent events), `data: [DONE]`이 끝을 알린다.
//!
//! 이 꼴을 따르면 그 API를 쓰도록 짠 프로그램들이 주소만 바꿔 이 서버를
//! 쓸 수 있다. 이 파일은 글자를 맞추는 일만 한다. 계산은 하지 않는다.

use std::sync::Arc;

use badak_formats::json::{self, Value};
use badak_models::constrain::Grammar;
use badak_models::tokenizer::chat::Message;

use crate::engine::{Completion, Finish, Options};

/// 해석한 요청.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatRequest {
    /// 대화.
    pub messages: Vec<Message>,
    /// 설정.
    pub options: Options,
    /// 조각조각 보낼지.
    pub stream: bool,
    /// 조각으로 보낼 때, 끝에 토큰의 수를 적은 조각을 하나 더 보낼지
    /// (`stream_options.include_usage`).
    pub include_usage: bool,
}

// ANCHOR: parse
/// 요청의 몸통을 해석한다. 꼴이 맞지 않으면 무엇이 틀렸는지를 글로
/// 돌려준다. 그 글은 400 응답에 담겨 보낸 쪽으로 간다.
pub fn parse_chat(body: &[u8]) -> Result<ChatRequest, String> {
    let text = std::str::from_utf8(body).map_err(|_| "몸통이 UTF-8이 아닙니다".to_owned())?;
    let root = json::parse(text).map_err(|error| format!("몸통이 JSON이 아닙니다: {error}"))?;
    let listed = root
        .get("messages")
        .and_then(Value::as_array)
        .ok_or("messages가 없거나 배열이 아닙니다")?;
    if listed.is_empty() {
        return Err("messages가 비었습니다".into());
    }
    let mut messages = Vec::with_capacity(listed.len());
    for item in listed {
        // 역할은 틀의 뼈대에 그대로 들어간다. 정해진 것만 받는다.
        let role = match item.get("role").and_then(Value::as_str) {
            Some(role @ ("system" | "user" | "assistant")) => role,
            Some("developer") => "system",
            _ => return Err("role은 system, user, assistant 가운데 하나여야 합니다".into()),
        };
        // 내용은 글이거나, 글 조각들의 배열이다.
        let content = match item.get("content") {
            Some(Value::String(text)) => text.clone(),
            Some(Value::Array(parts)) => {
                let mut joined = String::new();
                for part in parts {
                    match (part.get("type").and_then(Value::as_str), part.get("text")) {
                        (Some("text"), Some(Value::String(text))) => joined.push_str(text),
                        _ => return Err("글이 아닌 내용은 받지 않습니다".into()),
                    }
                }
                joined
            }
            _ => return Err("content가 없거나 글이 아닙니다".into()),
        };
        messages.push(Message::new(role, &content));
    }
    let number = |name: &str, low: f64, high: f64| -> Result<Option<f64>, String> {
        match root.get(name) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::Number(value)) if (low..=high).contains(value) => Ok(Some(*value)),
            _ => Err(format!("{name}: {low}에서 {high} 사이의 숫자여야 합니다")),
        }
    };
    let defaults = Options::default();
    let limit = match number("max_completion_tokens", 1.0, 65_536.0)? {
        Some(limit) => Some(limit),
        None => number("max_tokens", 1.0, 65_536.0)?,
    };
    let stream = match root.get("stream") {
        None | Some(Value::Null) => false,
        Some(Value::Bool(stream)) => *stream,
        _ => return Err("stream은 참이나 거짓이어야 합니다".into()),
    };
    // 41장: 답의 꼴을 JSON 스키마로 정할 수 있다.
    let grammar = match root.get("response_format") {
        None | Some(Value::Null) => None,
        Some(format) => match format.get("type").and_then(Value::as_str) {
            Some("text") => None,
            Some("json_schema") => {
                let schema = format
                    .get("json_schema")
                    .and_then(|named| named.get("schema"))
                    .ok_or("response_format.json_schema.schema가 없습니다")?;
                let grammar = Grammar::from_schema(schema)
                    .map_err(|error| format!("받지 못하는 스키마입니다: {error}"))?;
                Some(Arc::new(grammar))
            }
            _ => return Err("response_format의 type은 text나 json_schema여야 합니다".into()),
        },
    };
    Ok(ChatRequest {
        messages,
        options: Options {
            max_tokens: limit.map_or(defaults.max_tokens, |limit| limit as usize),
            temperature: number("temperature", 0.0, 2.0)?
                .map_or(defaults.temperature, |value| value as f32),
            top_p: number("top_p", 0.0, 1.0)?.map_or(defaults.top_p, |value| value as f32),
            seed: number("seed", 0.0, 9_007_199_254_740_992.0)?
                .map_or(defaults.seed, |value| value as u64),
            reuse: !matches!(root.get("cache_prompt"), Some(Value::Bool(false))),
            grammar,
        },
        stream,
        include_usage: matches!(
            root.get("stream_options")
                .and_then(|options| options.get("include_usage")),
            Some(Value::Bool(true))
        ),
    })
}
// ANCHOR_END: parse

/// 오류를 알리는 응답의 몸통.
pub fn error(message: &str, kind: &str) -> String {
    format!(
        "{{\"error\":{{\"message\":{},\"type\":{}}}}}",
        json::quote(message),
        json::quote(kind)
    )
}

fn finish_reason(finish: Finish) -> &'static str {
    match finish {
        Finish::Stop | Finish::Cancelled => "stop",
        Finish::Length => "length",
    }
}

fn usage(completion: &Completion) -> String {
    format!(
        "{{\"prompt_tokens\":{},\"completion_tokens\":{},\"total_tokens\":{},\"prompt_tokens_details\":{{\"cached_tokens\":{}}}}}",
        completion.prompt_tokens,
        completion.completion_tokens,
        completion.prompt_tokens + completion.completion_tokens,
        completion.cached_tokens
    )
}

/// 이 서버가 덧붙이는 시간 기록(밀리초). OpenAI의 꼴에는 없는 것이다.
fn timings(completion: &Completion) -> String {
    format!(
        "{{\"tokenize_ms\":{:.3},\"queued_ms\":{:.3},\"first_token_ms\":{:.3},\"total_ms\":{:.3}}}",
        completion.tokenize * 1e3,
        completion.queued * 1e3,
        completion.first_token * 1e3,
        completion.total * 1e3
    )
}

// ANCHOR: response
/// 한 번에 보내는 응답의 몸통.
pub fn chat_response(id: &str, created: u64, model: &str, completion: &Completion) -> String {
    format!(
        "{{\"id\":{},\"object\":\"chat.completion\",\"created\":{created},\"model\":{},\"choices\":[{{\"index\":0,\"message\":{{\"role\":\"assistant\",\"content\":{}}},\"finish_reason\":\"{}\"}}],\"usage\":{},\"timings\":{}}}",
        json::quote(id),
        json::quote(model),
        json::quote(&completion.text),
        finish_reason(completion.finish),
        usage(completion),
        timings(completion)
    )
}

/// 조각으로 보낼 때의 조각 하나. `delta`는 `{"role":"assistant"}`나
/// `{"content":"…"}`나 `{}`다.
pub fn chat_chunk(
    id: &str,
    created: u64,
    model: &str,
    delta: &str,
    finish: Option<&Completion>,
) -> String {
    // 마지막 조각에는 끝난 까닭과 시간 기록이 붙는다.
    let (reason, tail) = match finish {
        Some(completion) => (
            format!("\"{}\"", finish_reason(completion.finish)),
            format!(",\"timings\":{}", timings(completion)),
        ),
        None => ("null".to_owned(), String::new()),
    };
    format!(
        "data: {{\"id\":{},\"object\":\"chat.completion.chunk\",\"created\":{created},\"model\":{},\"choices\":[{{\"index\":0,\"delta\":{delta},\"finish_reason\":{reason}}}]{tail}}}\n\n",
        json::quote(id),
        json::quote(model)
    )
}

/// 토큰의 수를 적은 조각. 요청이 `stream_options.include_usage`를 켰을
/// 때만, `[DONE]` 바로 앞에 보낸다. `choices`는 비어 있다.
pub fn usage_chunk(id: &str, created: u64, model: &str, completion: &Completion) -> String {
    format!(
        "data: {{\"id\":{},\"object\":\"chat.completion.chunk\",\"created\":{created},\"model\":{},\"choices\":[],\"usage\":{}}}\n\n",
        json::quote(id),
        json::quote(model),
        usage(completion)
    )
}

/// 글 조각 하나를 담은 `delta`.
pub fn content_delta(text: &str) -> String {
    format!("{{\"content\":{}}}", json::quote(text))
}
// ANCHOR_END: response

/// `/v1/models`의 응답.
pub fn models(model: &str, created: u64) -> String {
    format!(
        "{{\"object\":\"list\",\"data\":[{{\"id\":{},\"object\":\"model\",\"created\":{created},\"owned_by\":\"badak\"}}]}}",
        json::quote(model)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_typical_request() {
        let request = parse_chat(
            br#"{"model":"x","messages":[{"role":"system","content":"Be brief."},
                {"role":"user","content":[{"type":"text","text":"Hi "},{"type":"text","text":"there"}]}],
                "max_tokens":32,"temperature":0,"stream":true,"seed":7}"#,
        )
        .unwrap();
        assert_eq!(request.messages.len(), 2);
        assert_eq!(request.messages[1], Message::new("user", "Hi there"));
        assert!(request.stream && !request.include_usage);
        assert_eq!(request.options.max_tokens, 32);
        assert_eq!(request.options.temperature, 0.0);
        assert_eq!(request.options.seed, 7);
        assert!(request.options.reuse);
    }

    #[test]
    fn uses_defaults_and_the_newer_limit_name() {
        let request = parse_chat(
            br#"{"messages":[{"role":"developer","content":"x"}],"max_completion_tokens":5,"max_tokens":9,"cache_prompt":false,"stream":true,"stream_options":{"include_usage":true}}"#,
        )
        .unwrap();
        assert_eq!(request.messages[0].role, "system");
        assert_eq!(request.options.max_tokens, 5);
        assert_eq!(request.options.temperature, 1.0);
        assert!(request.stream && request.include_usage && !request.options.reuse);
    }

    #[test]
    fn reads_a_json_schema_for_the_answer() {
        let body = br#"{"messages":[{"role":"user","content":"x"}],"response_format":{"type":"json_schema",
            "json_schema":{"name":"n","schema":{"type":"object","properties":{"a":{"type":"integer"}}}}}}"#;
        assert!(parse_chat(body).unwrap().options.grammar.is_some());
        let plain =
            br#"{"messages":[{"role":"user","content":"x"}],"response_format":{"type":"text"}}"#;
        assert!(parse_chat(plain).unwrap().options.grammar.is_none());
        for (format, word) in [
            (r#"{"type":"json_object"}"#, "type"),
            (r#"{"type":"json_schema"}"#, "schema"),
            (
                r#"{"type":"json_schema","json_schema":{"schema":{"type":"tuple"}}}"#,
                "tuple",
            ),
        ] {
            let body = format!(
                r#"{{"messages":[{{"role":"user","content":"x"}}],"response_format":{format}}}"#
            );
            let message = parse_chat(body.as_bytes()).unwrap_err();
            assert!(message.contains(word), "{message}");
        }
    }

    #[test]
    fn says_what_is_wrong() {
        for (body, word) in [
            (&b"not json"[..], "JSON"),
            (br#"{"messages":[]}"#, "비었"),
            (br#"{"model":"x"}"#, "messages"),
            (br#"{"messages":[{"role":"tool","content":"x"}]}"#, "role"),
            (br#"{"messages":[{"role":"user","content":7}]}"#, "content"),
            (
                br#"{"messages":[{"role":"user","content":[{"type":"image_url"}]}]}"#,
                "글이 아닌",
            ),
            (
                br#"{"messages":[{"role":"user","content":"x"}],"temperature":9}"#,
                "temperature",
            ),
            (
                br#"{"messages":[{"role":"user","content":"x"}],"max_tokens":0}"#,
                "max_tokens",
            ),
            (
                br#"{"messages":[{"role":"user","content":"x"}],"stream":"yes"}"#,
                "stream",
            ),
            (&[0xff, 0xfe][..], "UTF-8"),
        ] {
            let message = parse_chat(body).unwrap_err();
            assert!(message.contains(word), "{message}");
        }
    }

    #[test]
    fn responses_are_valid_json_with_escaped_text() {
        let completion = Completion {
            text: "He said \"hi\"\n".into(),
            prompt_tokens: 10,
            cached_tokens: 4,
            completion_tokens: 3,
            finish: Finish::Length,
            tokenize: 0.0,
            queued: 0.0,
            first_token: 0.0125,
            total: 0.05,
        };
        let body = json::parse(&chat_response("chatcmpl-1", 5, "m", &completion)).unwrap();
        let choice = &body.get("choices").unwrap().as_array().unwrap()[0];
        assert_eq!(
            choice
                .get("message")
                .unwrap()
                .get("content")
                .unwrap()
                .as_str(),
            Some("He said \"hi\"\n")
        );
        assert_eq!(
            choice.get("finish_reason").unwrap().as_str(),
            Some("length")
        );
        let usage = body.get("usage").unwrap();
        assert_eq!(usage.get("total_tokens").unwrap().as_usize(), Some(13));
        assert_eq!(
            usage
                .get("prompt_tokens_details")
                .unwrap()
                .get("cached_tokens")
                .unwrap()
                .as_usize(),
            Some(4)
        );

        // 조각은 `data: `로 시작하고 빈 줄로 끝난다. 그 사이가 JSON이다.
        let chunk = chat_chunk("chatcmpl-1", 5, "m", &content_delta("a\"b"), None);
        assert!(chunk.starts_with("data: ") && chunk.ends_with("\n\n"));
        let body = json::parse(chunk["data: ".len()..].trim()).unwrap();
        let choice = &body.get("choices").unwrap().as_array().unwrap()[0];
        assert_eq!(
            choice
                .get("delta")
                .unwrap()
                .get("content")
                .unwrap()
                .as_str(),
            Some("a\"b")
        );
        assert_eq!(choice.get("finish_reason"), Some(&Value::Null));
        let last = chat_chunk("chatcmpl-1", 5, "m", "{}", Some(&completion));
        let body = json::parse(last["data: ".len()..].trim()).unwrap();
        assert!(body.get("timings").is_some() && body.get("usage").is_none());
        let counts = usage_chunk("chatcmpl-1", 5, "m", &completion);
        let body = json::parse(counts["data: ".len()..].trim()).unwrap();
        assert_eq!(body.get("choices").unwrap().as_array().unwrap().len(), 0);
        assert_eq!(
            body.get("usage")
                .unwrap()
                .get("completion_tokens")
                .unwrap()
                .as_usize(),
            Some(3)
        );
        assert!(json::parse(&error("bad \"x\"", "invalid_request_error")).is_ok());
        assert!(json::parse(&models("m", 1)).is_ok());
    }
}
