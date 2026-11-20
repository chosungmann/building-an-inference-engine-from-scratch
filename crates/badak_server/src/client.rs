//! 검증과 측정에 쓰는 작은 HTTP 클라이언트 (39장).
//!
//! 서버를 만들었으면 서버를 부르는 쪽도 있어야 한다. 이 클라이언트는
//! 응답의 조각이 **언제** 왔는지를 적는다. 첫 토큰까지의 시간은 서버가
//! 스스로 잰 것이 아니라 받는 쪽에서 본 것이어야 한다.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::time::Instant;

/// 응답 하나.
#[derive(Debug, Clone)]
pub struct Response {
    /// 상태 번호.
    pub status: u16,
    /// 머리의 줄들. 이름은 소문자다.
    pub headers: Vec<(String, String)>,
    /// 몸통. 조각으로 왔으면 이어 붙인 것이다.
    pub body: Vec<u8>,
    /// 몸통의 조각 하나하나가 끝나는 자리(바이트)와, 요청을 보낸 때부터 그
    /// 조각이 온 때까지(초).
    pub arrivals: Vec<(usize, f64)>,
    /// 요청을 보낸 때부터 응답이 끝난 때까지(초).
    pub total: f64,
}

impl Response {
    /// 몸통을 글로.
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }

    /// 서버가 보낸 사건들(`data: …` 줄의 내용)과 그것이 다 온 때(초).
    pub fn events(&self) -> Vec<(&str, f64)> {
        let text = std::str::from_utf8(&self.body).unwrap_or_default();
        let mut events = Vec::new();
        let mut at = 0;
        for event in text.split_inclusive("\n\n") {
            at += event.len();
            if let Some(data) = event.trim().strip_prefix("data: ") {
                // 이 사건의 마지막 바이트가 든 조각이 온 때.
                let arrived = self
                    .arrivals
                    .iter()
                    .find(|(end, _)| *end >= at)
                    .map_or(self.total, |(_, time)| *time);
                events.push((data, arrived));
            }
        }
        events
    }
}

/// 서버와의 연결 하나. 요청을 여러 번 보낼 수 있다.
pub struct Client {
    reader: BufReader<TcpStream>,
    writer: TcpStream,
    host: String,
}

fn broken(what: &str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, what.to_owned())
}

fn line(reader: &mut impl BufRead) -> std::io::Result<String> {
    let mut text = String::new();
    if reader.read_line(&mut text)? == 0 {
        return Err(broken("응답이 끝나기 전에 연결이 끊겼습니다"));
    }
    Ok(text.trim_end().to_owned())
}

impl Client {
    /// `주소:포트`에 잇는다.
    pub fn connect(address: &str) -> std::io::Result<Self> {
        let writer = TcpStream::connect(address)?;
        writer.set_nodelay(std::env::var_os("BADAK_NAGLE").is_none())?;
        Ok(Self {
            reader: BufReader::new(writer.try_clone()?),
            writer,
            host: address.to_owned(),
        })
    }

    /// 요청 하나를 보내고 응답을 끝까지 읽는다.
    pub fn send(&mut self, method: &str, path: &str, body: &str) -> std::io::Result<Response> {
        let start = Instant::now();
        write!(
            self.writer,
            "{method} {path} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            self.host,
            body.len()
        )?;
        self.writer.flush()?;

        let first = line(&mut self.reader)?;
        let status = first
            .split(' ')
            .nth(1)
            .and_then(|status| status.parse().ok())
            .ok_or_else(|| broken("상태 줄을 읽지 못했습니다"))?;
        let mut headers = Vec::new();
        loop {
            let text = line(&mut self.reader)?;
            let Some((name, value)) = text.split_once(':') else {
                break;
            };
            headers.push((name.trim().to_ascii_lowercase(), value.trim().to_owned()));
        }
        let find = |name: &str| {
            headers
                .iter()
                .find(|(key, _)| key == name)
                .map(|(_, value)| value.as_str())
        };
        let (mut body, mut arrivals) = (Vec::new(), Vec::new());
        if find("transfer-encoding").is_some_and(|value| value.eq_ignore_ascii_case("chunked")) {
            // 조각: 길이(16진수) 한 줄, 내용, 빈 줄. 길이 0이 끝이다.
            loop {
                let size = line(&mut self.reader)?;
                let size = usize::from_str_radix(size.split(';').next().unwrap_or(""), 16)
                    .map_err(|_| broken("조각의 길이를 읽지 못했습니다"))?;
                let at = body.len();
                body.resize(at + size, 0);
                self.reader.read_exact(&mut body[at..])?;
                line(&mut self.reader)?;
                if size == 0 {
                    break;
                }
                arrivals.push((body.len(), start.elapsed().as_secs_f64()));
            }
        } else {
            let length = find("content-length")
                .and_then(|value| value.parse().ok())
                .ok_or_else(|| broken("응답의 길이를 알 수 없습니다"))?;
            body.resize(length, 0);
            self.reader.read_exact(&mut body)?;
            arrivals.push((body.len(), start.elapsed().as_secs_f64()));
        }
        Ok(Response {
            status,
            headers,
            body,
            arrivals,
            total: start.elapsed().as_secs_f64(),
        })
    }
}

/// 연결을 새로 맺어 요청 하나를 보낸다.
pub fn request(address: &str, method: &str, path: &str, body: &str) -> std::io::Result<Response> {
    Client::connect(address)?.send(method, path, body)
}
