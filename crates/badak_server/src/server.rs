//! 연결을 받고, 요청을 읽고, 길을 찾아 답한다 (39장).

use std::io::{BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use badak_formats::json::{self, Value};

use crate::api;
use badak_formats::Result as EngineResult;
use badak_models::tokenizer::chat::Message;

use crate::engine::{Completion, Engine, Options};
use crate::http::{self, Chunked, Request};
use crate::scheduler::{Scheduler, Settings};

/// 한 번에 열어 두는 연결의 한도. 넘으면 503으로 답하고 닫는다.
pub const MAX_CONNECTIONS: usize = 64;
/// 연결이 이만큼 조용하면 닫는다. 요청을 조금씩 흘려 보내며 자리를
/// 차지하는 쪽을 막는다.
pub const IDLE: Duration = Duration::from_secs(30);

/// 서버 하나.
pub struct Server {
    /// 모델과 KV 캐시.
    pub engine: Arc<Engine>,
    /// 요청을 묶어 처리하는 스케줄러(40장). 없으면 39장처럼 한 번에 한
    /// 요청을 처리한다.
    pub scheduler: Option<Scheduler>,
    /// 캐시 파일을 두는 디렉터리. 없으면 저장과 복원을 받지 않는다.
    pub cache_dir: Option<PathBuf>,
    served: AtomicU64,
    connections: AtomicUsize,
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs())
}

fn send(writer: &mut impl Write, status: u16, body: &str) -> std::io::Result<()> {
    http::respond(writer, status, "application/json", body.as_bytes())
}

fn refuse(writer: &mut impl Write, status: u16, message: &str) -> std::io::Result<()> {
    let kind = match status {
        404 => "not_found_error",
        500.. => "server_error",
        _ => "invalid_request_error",
    };
    send(writer, status, &api::error(message, kind))
}

// ANCHOR: cache_name
/// 캐시 파일의 이름으로 받을 만한가. 영문자, 숫자, `_`, `-`만 받는다.
///
/// 이름은 밖에서 온 글이고 파일의 경로가 된다. `../../etc/passwd` 같은
/// 이름이 디렉터리 밖을 가리키지 못하게, 경로를 이루는 글자(`/`, `.`,
/// `\`)를 아예 받지 않는다.
pub fn valid_cache_name(name: &str) -> bool {
    (1..=64).contains(&name.len())
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
}
// ANCHOR_END: cache_name

impl Server {
    /// 서버를 만든다.
    pub fn new(engine: Engine, cache_dir: Option<PathBuf>) -> Self {
        Self {
            engine: Arc::new(engine),
            scheduler: None,
            cache_dir,
            served: AtomicU64::new(0),
            connections: AtomicUsize::new(0),
        }
    }

    /// 요청을 묶어 처리하게 한다 (40장).
    pub fn with_scheduler(mut self, settings: Settings) -> Self {
        self.scheduler = Some(Scheduler::start(Arc::clone(&self.engine), settings));
        self
    }

    /// 대화에 답을 쓴다. 스케줄러가 있으면 그쪽으로 넘긴다.
    fn complete(
        &self,
        messages: &[Message],
        options: &Options,
        on_text: &mut dyn FnMut(&str) -> bool,
    ) -> EngineResult<Completion> {
        match &self.scheduler {
            Some(scheduler) => scheduler.complete(messages, options, on_text),
            None => self.engine.complete(messages, options, on_text),
        }
    }

    // ANCHOR: route
    /// 요청 하나에 답한다.
    pub fn handle(&self, request: &Request, writer: &mut impl Write) -> std::io::Result<()> {
        match (request.method.as_str(), request.path.as_str()) {
            ("GET", "/health") => send(writer, 200, "{\"status\":\"ok\"}"),
            ("GET", "/v1/models") => send(writer, 200, &api::models(&self.engine.name, now())),
            ("POST", "/v1/chat/completions") => self.chat(request, writer),
            ("GET", "/cache") if self.scheduler.is_some() => self.pool(writer),
            ("GET", "/cache") => {
                let (tokens, bytes) = self.engine.cached();
                send(
                    writer,
                    200,
                    &format!("{{\"tokens\":{tokens},\"bytes\":{bytes}}}"),
                )
            }
            ("POST", "/cache/save") => self.persist(request, writer, true),
            ("POST", "/cache/restore") => self.persist(request, writer, false),
            (
                _,
                "/health"
                | "/v1/models"
                | "/v1/chat/completions"
                | "/cache"
                | "/cache/save"
                | "/cache/restore",
            ) => refuse(writer, 405, "이 길은 그 방법을 받지 않습니다"),
            _ => refuse(writer, 404, "그런 길이 없습니다"),
        }
    }
    // ANCHOR_END: route

    // ANCHOR: chat
    fn chat(&self, request: &Request, writer: &mut impl Write) -> std::io::Result<()> {
        let chat = match api::parse_chat(&request.body) {
            Ok(chat) => chat,
            Err(message) => return refuse(writer, 400, &message),
        };
        let id = format!("chatcmpl-{}", self.served.fetch_add(1, Ordering::Relaxed));
        let (created, model) = (now(), self.engine.name.as_str());

        if !chat.stream {
            // 다 쓴 뒤에 한 번에 보낸다.
            return match self.complete(&chat.messages, &chat.options, &mut |_| true) {
                Ok(completion) => send(
                    writer,
                    200,
                    &api::chat_response(&id, created, model, &completion),
                ),
                Err(badak_formats::Error::Unsupported(message)) => refuse(writer, 400, &message),
                Err(error) => refuse(writer, 500, &error.to_string()),
            };
        }

        // 쓰는 대로 보낸다. 머리는 첫 조각이 나올 때 쓴다. 그 전에 일이
        // 틀어지면(글이 한도를 넘으면) 아직 보통의 오류 응답을 보낼 수 있다.
        let open = |writer| -> std::io::Result<Chunked<_>> {
            let mut stream = Chunked::start(writer, "text/event-stream")?;
            let role = api::chat_chunk(&id, created, model, "{\"role\":\"assistant\"}", None);
            stream.write(role.as_bytes())?;
            Ok(stream)
        };
        let mut writer = Some(writer);
        let mut stream = None;
        let mut failure = None;
        let completion = self.complete(&chat.messages, &chat.options, &mut |text| {
            // 보내지 못했으면(받는 쪽이 연결을 끊었으면) 거짓을 돌려준다.
            // 엔진은 거기서 쓰기를 그만둔다. 듣는 이 없는 답에 계산을 쓰지
            // 않는다.
            let sent = (|| {
                if let Some(writer) = writer.take() {
                    stream = Some(open(writer)?);
                }
                let chunk = api::chat_chunk(&id, created, model, &api::content_delta(text), None);
                stream
                    .as_mut()
                    .map_or(Ok(()), |stream| stream.write(chunk.as_bytes()))
            })();
            sent.map_err(|error| failure = Some(error)).is_ok()
        });
        if let Some(error) = failure {
            return Err(error);
        }
        let completion = match (completion, &mut writer) {
            (Ok(completion), _) => completion,
            (Err(badak_formats::Error::Unsupported(message)), Some(writer)) => {
                return refuse(writer, 400, &message);
            }
            (Err(error), Some(writer)) => return refuse(writer, 500, &error.to_string()),
            // 머리를 이미 보냈으면 상태 번호를 바꿀 수 없다. 연결을 끊어
            // 알린다. 끝을 알리는 조각이 없으므로 받는 쪽은 응답이 잘렸음을
            // 안다.
            (Err(error), None) => return Err(std::io::Error::other(error.to_string())),
        };
        let mut stream = match (stream, writer) {
            (Some(stream), _) => stream,
            (None, Some(writer)) => open(writer)?,
            (None, None) => unreachable!("머리를 썼으면 stream이 있다"),
        };
        let last = api::chat_chunk(&id, created, model, "{}", Some(&completion));
        stream.write(last.as_bytes())?;
        if chat.include_usage {
            stream.write(api::usage_chunk(&id, created, model, &completion).as_bytes())?;
        }
        stream.write(b"data: [DONE]\n\n")?;
        stream.finish()
    }
    // ANCHOR_END: chat

    /// 스케줄러의 페이지 풀이 어떤지 알려 준다.
    fn pool(&self, writer: &mut impl Write) -> std::io::Result<()> {
        let Some(scheduler) = &self.scheduler else {
            return refuse(writer, 404, "스케줄러가 없습니다");
        };
        let (pool, stats) = (&scheduler.pool, &scheduler.stats);
        send(
            writer,
            200,
            &format!(
                "{{\"pages\":{},\"peak_pages\":{},\"page_bytes\":{},\"running\":{},\"idle\":{},\"steps\":{},\"rows\":{}}}",
                pool.used(),
                pool.peak(),
                pool.page_bytes(),
                stats.running.load(Ordering::Acquire),
                stats.idle.load(Ordering::Acquire),
                stats.steps.load(Ordering::Relaxed),
                stats.rows.load(Ordering::Relaxed),
            ),
        )
    }

    fn persist(
        &self,
        request: &Request,
        writer: &mut impl Write,
        save: bool,
    ) -> std::io::Result<()> {
        let Some(directory) = &self.cache_dir else {
            return refuse(
                writer,
                404,
                "이 서버는 캐시 파일을 두는 곳을 정하지 않았습니다",
            );
        };
        let name = std::str::from_utf8(&request.body)
            .ok()
            .and_then(|text| json::parse(text).ok())
            .and_then(|root| root.get("name").and_then(Value::as_str).map(str::to_owned));
        let Some(name) = name.filter(|name| valid_cache_name(name)) else {
            return refuse(
                writer,
                400,
                "name은 영문자, 숫자, _, -로 된 64자 이하의 글이어야 합니다",
            );
        };
        let path = directory.join(format!("{name}.kv"));
        let start = std::time::Instant::now();
        let done = if save {
            self.engine
                .save(&path)
                .map(|bytes| (self.engine.cached().0, bytes))
        } else {
            self.engine.restore(&path).map(|tokens| {
                (
                    tokens,
                    std::fs::metadata(&path).map_or(0, |file| file.len() as usize),
                )
            })
        };
        match done {
            Ok((tokens, bytes)) => send(
                writer,
                200,
                &format!(
                    "{{\"name\":{},\"tokens\":{tokens},\"bytes\":{bytes},\"ms\":{:.3}}}",
                    json::quote(&name),
                    start.elapsed().as_secs_f64() * 1e3
                ),
            ),
            Err(badak_formats::Error::Io(error))
                if error.kind() == std::io::ErrorKind::NotFound =>
            {
                refuse(writer, 404, "그런 이름의 캐시 파일이 없습니다")
            }
            Err(error) => refuse(writer, 400, &error.to_string()),
        }
    }

    // ANCHOR: connection
    /// 연결 하나를 맡는다. 요청을 읽고 답하기를 연결이 닫힐 때까지 되풀이한다.
    fn connection(&self, stream: TcpStream) -> std::io::Result<()> {
        stream.set_read_timeout(Some(IDLE))?;
        stream.set_write_timeout(Some(IDLE))?;
        // 조각을 쓰는 대로 내보낸다. 작은 조각을 모았다가 보내는 네이글
        // 알고리즘을 끈다. `BADAK_NAGLE`이 있으면 켜 둔다(재 보려고 둔 것).
        stream.set_nodelay(std::env::var_os("BADAK_NAGLE").is_none())?;
        let mut writer = stream.try_clone()?;
        let mut reader = BufReader::new(stream);
        loop {
            let request = match http::read_request(&mut reader, &mut writer) {
                Ok(Some(request)) => request,
                Ok(None) => return Ok(()),
                Err(http::Error::Io(error)) => return Err(error),
                // 꼴이 틀린 요청 뒤에는 어디서부터가 다음 요청인지 알 수
                // 없다. 답하고 닫는다.
                Err(http::Error::Malformed(what)) => return refuse(&mut writer, 400, what),
                Err(http::Error::TooLarge) => {
                    return refuse(&mut writer, 413, "요청이 너무 깁니다");
                }
            };
            self.handle(&request, &mut writer)?;
            if request.wants_close() {
                return Ok(());
            }
        }
    }

    /// 연결을 받는 대로 스레드 하나에 맡긴다. 돌아오지 않는다.
    pub fn serve(self: &Arc<Self>, listener: &TcpListener) -> std::io::Result<()> {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            if self.connections.fetch_add(1, Ordering::AcqRel) >= MAX_CONNECTIONS {
                self.connections.fetch_sub(1, Ordering::AcqRel);
                let _ = refuse(&mut stream, 503, "연결이 너무 많습니다");
                continue;
            }
            let server = Arc::clone(self);
            std::thread::spawn(move || {
                // 연결에서 생긴 오류(끊김, 시간 초과)는 그 연결의 일이다.
                // 서버는 계속 돈다.
                let _ = server.connection(stream);
                server.connections.fetch_sub(1, Ordering::AcqRel);
            });
        }
        Ok(())
    }
    // ANCHOR_END: connection
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_names_cannot_leave_the_directory() {
        for name in ["turn_1", "a", "A-b_9"] {
            assert!(valid_cache_name(name), "{name}");
        }
        for name in [
            "",
            "../x",
            "a/b",
            "a.kv",
            "a\\b",
            "이름",
            "a b",
            &"x".repeat(65),
        ] {
            assert!(!valid_cache_name(name), "{name:?}");
        }
    }
}
