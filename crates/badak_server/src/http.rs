//! HTTP/1.1을 필요한 만큼만 (39장).
//!
//! 요청 하나는 글로 된 머리와 바이트로 된 몸통이다. 머리의 첫 줄이 무엇을
//! 하라는 것인지(`POST /v1/chat/completions HTTP/1.1`)를, 그 뒤의 줄들이
//! 딸린 정보(`이름: 값`)를 적고, 빈 줄 하나가 머리의 끝을 알린다. 몸통의
//! 길이는 머리의 `Content-Length`가 알려 준다.
//!
//! 응답도 같은 모양이다. 길이를 미리 모르는 응답(토큰을 쓰는 대로
//! 내보내는 응답)은 **조각 전송**으로 보낸다. 조각마다 길이를 먼저 적고,
//! 길이가 0인 조각이 끝을 알린다.

use std::io::{BufRead, Read, Write};

/// 머리 전체가 이보다 길면 받지 않는다.
pub const MAX_HEAD: usize = 16 * 1024;
/// 몸통이 이보다 길면 받지 않는다.
pub const MAX_BODY: usize = 4 * 1024 * 1024;

/// 요청을 읽다가 생긴 일.
#[derive(Debug)]
pub enum Error {
    /// 연결에서 읽거나 쓰지 못했다.
    Io(std::io::Error),
    /// 요청이 HTTP의 꼴이 아니다. 400으로 답한다.
    Malformed(&'static str),
    /// 머리나 몸통이 한도보다 길다. 413으로 답한다.
    TooLarge,
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "연결에서 읽거나 쓰지 못했습니다: {error}"),
            Self::Malformed(what) => write!(f, "요청이 HTTP의 꼴이 아닙니다: {what}"),
            Self::TooLarge => write!(f, "요청이 너무 깁니다"),
        }
    }
}

impl std::error::Error for Error {}

// ANCHOR: request
/// 요청 하나.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Request {
    /// `GET`, `POST` 같은 것.
    pub method: String,
    /// 경로. 물음표 뒤(질의)는 떼어 버린다.
    pub path: String,
    /// 머리의 줄들. 이름은 소문자로 바꿔 둔다(HTTP는 이름의 대소문자를
    /// 가리지 않는다).
    pub headers: Vec<(String, String)>,
    /// 몸통.
    pub body: Vec<u8>,
}

impl Request {
    /// 머리에서 이름으로 값을 찾는다.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
    }

    /// 이 요청 뒤에 연결을 닫아 달라고 했는가.
    pub fn wants_close(&self) -> bool {
        self.header("connection")
            .is_some_and(|value| value.eq_ignore_ascii_case("close"))
    }
}
// ANCHOR_END: request

/// 줄 하나를 읽는다. 끝의 `\r\n`은 뗀다. 연결이 닫혔으면 `None`.
fn read_line(reader: &mut impl BufRead, budget: &mut usize) -> Result<Option<String>, Error> {
    let mut line = Vec::new();
    // 한도보다 한 바이트 더 읽어 보면 넘쳤는지 알 수 있다.
    let read = reader
        .by_ref()
        .take(*budget as u64 + 1)
        .read_until(b'\n', &mut line)?;
    if read == 0 {
        return Ok(None);
    }
    if read > *budget {
        return Err(Error::TooLarge);
    }
    *budget -= read;
    if line.pop() != Some(b'\n') {
        return Err(Error::Malformed("줄이 끝나기 전에 연결이 끊겼습니다"));
    }
    if line.last() == Some(&b'\r') {
        line.pop();
    }
    String::from_utf8(line)
        .map(Some)
        .map_err(|_| Error::Malformed("머리가 UTF-8이 아닙니다"))
}

// ANCHOR: read
/// 연결에서 요청 하나를 읽는다. 연결이 조용히 닫혔으면 `None`이다.
///
/// 길이를 믿지 않는다. 머리는 `MAX_HEAD`, 몸통은 `MAX_BODY`까지만 읽고,
/// 넘으면 읽기를 그만둔다. 누구나 보낼 수 있는 글을 받는 곳이므로, 보내는
/// 쪽이 적은 길이만큼 메모리를 잡아서는 안 된다.
///
/// `interim`은 같은 연결의 쓰는 쪽이다. 보내는 쪽이 `Expect: 100-continue`로
/// "몸통을 보내도 되는가"를 물었을 때 거기에 "보내라"고 답한다.
pub fn read_request(
    reader: &mut impl BufRead,
    interim: &mut impl Write,
) -> Result<Option<Request>, Error> {
    let mut budget = MAX_HEAD;
    // 1. 첫 줄: 방법, 경로, 버전.
    let Some(first) = read_line(reader, &mut budget)? else {
        return Ok(None);
    };
    let mut parts = first.split(' ');
    let (Some(method), Some(target), Some(version), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return Err(Error::Malformed("첫 줄이 세 낱말이 아닙니다"));
    };
    if !version.starts_with("HTTP/1.") {
        return Err(Error::Malformed("HTTP/1.x가 아닙니다"));
    }
    let path = target.split('?').next().unwrap_or(target);

    // 2. 머리의 줄들. 빈 줄에서 끝난다.
    let mut headers = Vec::new();
    loop {
        let line = read_line(reader, &mut budget)?
            .ok_or(Error::Malformed("머리가 끝나기 전에 연결이 끊겼습니다"))?;
        if line.is_empty() {
            break;
        }
        let (name, value) = line
            .split_once(':')
            .ok_or(Error::Malformed("머리의 줄에 쌍점이 없습니다"))?;
        headers.push((name.trim().to_ascii_lowercase(), value.trim().to_owned()));
    }

    // 3. 몸통. 길이는 머리가 알려 준다.
    let length = |name: &str| headers.iter().find(|(key, _)| key == name);
    if length("transfer-encoding").is_some() {
        return Err(Error::Malformed("조각으로 보낸 요청은 받지 않습니다"));
    }
    let length = match length("content-length") {
        Some((_, value)) => value
            .parse::<usize>()
            .map_err(|_| Error::Malformed("Content-Length가 숫자가 아닙니다"))?,
        None => 0,
    };
    if length > MAX_BODY {
        return Err(Error::TooLarge);
    }
    // 몸통이 길면(curl은 1,024바이트부터) 보내는 쪽이 머리만 보내고 허락을
    // 기다리기도 한다. 답하지 않으면 1초쯤 기다리다가 보낸다.
    let expects = headers
        .iter()
        .any(|(key, value)| key == "expect" && value.eq_ignore_ascii_case("100-continue"));
    if expects && length > 0 {
        interim.write_all(b"HTTP/1.1 100 Continue\r\n\r\n")?;
        interim.flush()?;
    }
    let mut body = vec![0; length];
    reader
        .read_exact(&mut body)
        .map_err(|_| Error::Malformed("몸통이 적힌 길이보다 짧습니다"))?;
    Ok(Some(Request {
        method: method.to_owned(),
        path: path.to_owned(),
        headers,
        body,
    }))
}
// ANCHOR_END: read

/// 상태 번호에 붙는 말.
fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        408 => "Request Timeout",
        413 => "Content Too Large",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "Unknown",
    }
}

// ANCHOR: respond
/// 길이를 아는 응답을 한 번에 쓴다.
pub fn respond(
    writer: &mut impl Write,
    status: u16,
    content_type: &str,
    body: &[u8],
) -> std::io::Result<()> {
    write!(
        writer,
        "HTTP/1.1 {status} {}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n\r\n",
        reason(status),
        body.len()
    )?;
    writer.write_all(body)?;
    writer.flush()
}
// ANCHOR_END: respond

// ANCHOR: chunked
/// 길이를 모르는 응답을 조각으로 쓴다.
///
/// 조각 하나는 `길이(16진수)\r\n내용\r\n`이다. 조각을 쓸 때마다 바로
/// 내보낸다(`flush`). 모아 두었다가 보내면 흘려보내는 뜻이 없다.
pub struct Chunked<W: Write> {
    writer: W,
}

impl<W: Write> Chunked<W> {
    /// 머리를 쓰고 조각을 받을 준비를 한다.
    pub fn start(mut writer: W, content_type: &str) -> std::io::Result<Self> {
        write!(
            writer,
            "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\n\r\n"
        )?;
        writer.flush()?;
        Ok(Self { writer })
    }

    /// 조각 하나를 쓰고 바로 내보낸다. 빈 조각은 쓰지 않는다(길이 0은 끝을
    /// 뜻한다).
    pub fn write(&mut self, data: &[u8]) -> std::io::Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        write!(self.writer, "{:x}\r\n", data.len())?;
        self.writer.write_all(data)?;
        self.writer.write_all(b"\r\n")?;
        self.writer.flush()
    }

    /// 끝을 알리는 길이 0의 조각을 쓴다.
    pub fn finish(mut self) -> std::io::Result<()> {
        self.writer.write_all(b"0\r\n\r\n")?;
        self.writer.flush()
    }
}
// ANCHOR_END: chunked

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> Result<Option<Request>, Error> {
        read_request(
            &mut std::io::BufReader::new(text.as_bytes()),
            &mut std::io::sink(),
        )
    }

    #[test]
    fn reads_a_request_with_a_body() {
        let request = parse(
            "POST /v1/chat/completions?x=1 HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nCONTENT-LENGTH: 7\r\n\r\n{\"a\":1}extra",
        )
        .unwrap()
        .unwrap();
        assert_eq!(request.method, "POST");
        assert_eq!(request.path, "/v1/chat/completions");
        assert_eq!(request.header("content-type"), Some("application/json"));
        assert_eq!(request.body, b"{\"a\":1}");
        assert!(!request.wants_close());
    }

    #[test]
    fn reads_two_requests_from_one_connection() {
        let text =
            "GET /health HTTP/1.1\r\n\r\nGET /v1/models HTTP/1.1\r\nConnection: close\r\n\r\n";
        let mut reader = std::io::BufReader::new(text.as_bytes());
        let first = read_request(&mut reader, &mut std::io::sink())
            .unwrap()
            .unwrap();
        let second = read_request(&mut reader, &mut std::io::sink())
            .unwrap()
            .unwrap();
        assert_eq!(
            (first.path.as_str(), second.path.as_str()),
            ("/health", "/v1/models")
        );
        assert!(second.wants_close());
        // 연결이 닫히면 오류가 아니라 "더 없음"이다.
        assert!(
            read_request(&mut reader, &mut std::io::sink())
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn rejects_what_is_not_http() {
        for text in [
            "hello\r\n\r\n",
            "GET / HTTP/2\r\n\r\n",
            "GET / HTTP/1.1\r\nno colon\r\n\r\n",
            "POST / HTTP/1.1\r\nContent-Length: many\r\n\r\n",
            "POST / HTTP/1.1\r\nContent-Length: 10\r\n\r\nshort",
            "POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n",
            "GET / HTTP/1.1\r\nHost: x",
        ] {
            assert!(matches!(parse(text), Err(Error::Malformed(_))), "{text:?}");
        }
    }

    #[test]
    fn refuses_oversized_heads_and_bodies_without_reading_them() {
        let long = format!("GET / HTTP/1.1\r\nX: {}\r\n\r\n", "a".repeat(MAX_HEAD));
        assert!(matches!(parse(&long), Err(Error::TooLarge)));
        // 몸통은 적힌 길이만 보고 거절한다. 그만큼의 메모리를 잡지 않는다.
        let huge = "POST / HTTP/1.1\r\nContent-Length: 99999999999\r\n\r\n";
        assert!(matches!(parse(huge), Err(Error::TooLarge)));
    }

    #[test]
    fn answers_a_sender_that_waits_for_permission() {
        let text = "POST / HTTP/1.1\r\nExpect: 100-continue\r\nContent-Length: 2\r\n\r\nhi";
        let mut interim = Vec::new();
        let request = read_request(&mut std::io::BufReader::new(text.as_bytes()), &mut interim)
            .unwrap()
            .unwrap();
        assert_eq!(request.body, b"hi");
        assert_eq!(interim, b"HTTP/1.1 100 Continue\r\n\r\n");
    }

    #[test]
    fn writes_plain_and_chunked_responses() {
        let mut out = Vec::new();
        respond(&mut out, 404, "application/json", b"{}").unwrap();
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}"
        );
        let mut out = Vec::new();
        let mut chunked = Chunked::start(&mut out, "text/event-stream").unwrap();
        chunked.write(b"hello").unwrap();
        chunked.write(b"").unwrap();
        chunked.write(&[b'x'; 26]).unwrap();
        chunked.finish().unwrap();
        let text = String::from_utf8(out).unwrap();
        let body = text.split("\r\n\r\n").nth(1).unwrap();
        assert!(text.contains("Transfer-Encoding: chunked"));
        assert!(body.starts_with("5\r\nhello\r\n1a\r\nxxxx"));
        assert!(text.ends_with("\r\n0\r\n\r\n"));
    }
}
