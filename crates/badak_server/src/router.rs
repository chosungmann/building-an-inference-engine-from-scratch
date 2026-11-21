//! 서버 여러 대 앞에 서서 요청을 나눠 주는 라우터 (42장).
//!
//! 서버가 한 대일 때는 "어느 서버로 보낼까"를 물을 일이 없었다. 여러
//! 대가 되면 그 선택이 성능을 정한다. 39장의 접두사 캐시는 **그 캐시를
//! 가진 서버로 요청이 가야** 듣기 때문이다.

use std::collections::VecDeque;
use std::io::{BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use badak_formats::json::{self, Value};

use crate::http::{self, Request};

/// 서버마다 기억해 두는 글의 수.
const REMEMBERED: usize = 64;

// ANCHOR: policy
/// 요청을 어느 서버로 보낼지 정하는 방법.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    /// 차례대로 돌아가며 보낸다.
    RoundRobin,
    /// 처리하고 있는 요청이 가장 적은 서버로 보낸다.
    LeastBusy,
    /// 글의 앞 `bytes`바이트로 서버를 정한다. 같은 앞부분은 늘 같은 서버로
    /// 간다. 서버가 무엇을 쥐고 있는지는 보지 않는다.
    PrefixHash {
        /// 볼 바이트의 수.
        bytes: usize,
    },
    /// 서버마다 최근에 보낸 글들을 기억해 두고, 새 글과 **앞에서부터 가장
    /// 길게 겹치는** 글을 받았던 서버로 보낸다. 겹치는 것이 `min`바이트에
    /// 못 미치면 가장 한가한 서버로 보낸다.
    LongestPrefix {
        /// 캐시가 있다고 볼 겹침의 임계값(바이트).
        min: usize,
    },
}
// ANCHOR_END: policy

/// 뒤에 있는 서버 하나.
struct Backend {
    address: String,
    /// 지금 처리하고 있는 요청의 수.
    busy: AtomicUsize,
    /// 지금까지 보낸 요청의 수.
    served: AtomicUsize,
    /// 최근에 보낸 글들.
    seen: Mutex<VecDeque<String>>,
}

/// 라우터.
pub struct Router {
    backends: Vec<Backend>,
    policy: Policy,
    next: AtomicUsize,
}

/// 두 글이 앞에서부터 몇 바이트 같은가.
fn common(a: &str, b: &str) -> usize {
    a.bytes().zip(b.bytes()).take_while(|(x, y)| x == y).count()
}

/// 요청의 몸통에서 대화의 글을 뽑아 한 줄로 잇는다. 서버가 읽게 될 글과
/// 앞부분이 같은 순서로 놓인다.
pub fn conversation(body: &[u8]) -> String {
    let root = std::str::from_utf8(body)
        .ok()
        .and_then(|text| json::parse(text).ok());
    let mut text = String::new();
    let messages = root.as_ref().and_then(|root| root.get("messages"));
    for message in messages.and_then(Value::as_array).unwrap_or_default() {
        for name in ["role", "content"] {
            if let Some(part) = message.get(name).and_then(Value::as_str) {
                text.push_str(part);
                text.push('\n');
            }
        }
    }
    text
}

impl Router {
    /// 서버들의 주소와 방법으로 라우터를 만든다.
    pub fn new(addresses: &[String], policy: Policy) -> Self {
        Self {
            backends: addresses
                .iter()
                .map(|address| Backend {
                    address: address.clone(),
                    busy: AtomicUsize::new(0),
                    served: AtomicUsize::new(0),
                    seen: Mutex::new(VecDeque::new()),
                })
                .collect(),
            policy,
            next: AtomicUsize::new(0),
        }
    }

    /// 가장 한가한 서버. 같으면 차례대로.
    fn least_busy(&self) -> usize {
        let count = self.backends.len();
        let start = self.next.fetch_add(1, Ordering::Relaxed);
        (0..count)
            .map(|offset| (start + offset) % count)
            .min_by_key(|&index| self.backends[index].busy.load(Ordering::Acquire))
            .unwrap_or(0)
    }

    // ANCHOR: choose
    /// 이 글을 어느 서버로 보낼지 고른다.
    pub fn choose(&self, text: &str) -> usize {
        let count = self.backends.len();
        match self.policy {
            Policy::RoundRobin => self.next.fetch_add(1, Ordering::Relaxed) % count,
            Policy::LeastBusy => self.least_busy(),
            Policy::PrefixHash { bytes } => {
                // FNV-1a. 같은 앞부분은 같은 수가 되고, 수는 서버들에 고르게 퍼진다.
                let mut hash = 0xcbf2_9ce4_8422_2325_u64;
                for byte in text.bytes().take(bytes) {
                    hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
                }
                (hash % count as u64) as usize
            }
            Policy::LongestPrefix { min } => {
                // 서버마다, 기억해 둔 글 가운데 이 글과 가장 길게 겹치는 길이.
                let overlap = |backend: &Backend| {
                    let seen = backend.seen.lock().expect("기억을 쥔 스레드가 죽었다");
                    seen.iter().map(|old| common(old, text)).max().unwrap_or(0)
                };
                let best = (0..count)
                    .map(|index| (overlap(&self.backends[index]), index))
                    .max_by_key(|&(length, index)| {
                        // 겹침이 같으면 덜 바쁜 쪽.
                        let busy = self.backends[index].busy.load(Ordering::Acquire);
                        (length, std::cmp::Reverse(busy))
                    });
                match best {
                    Some((length, index)) if length >= min => index,
                    _ => self.least_busy(),
                }
            }
        }
    }
    // ANCHOR_END: choose

    /// 이 서버로 이 글을 보냈다고 적어 둔다.
    fn remember(&self, index: usize, text: &str) {
        let mut seen = self.backends[index]
            .seen
            .lock()
            .expect("기억을 쥔 스레드가 죽었다");
        // 새 글의 앞부분일 뿐인 옛 글은 지운다. 대화가 이어지면 그렇게 된다.
        seen.retain(|old| !text.starts_with(old.as_str()));
        seen.push_back(text.to_owned());
        while seen.len() > REMEMBERED {
            seen.pop_front();
        }
    }

    /// 서버마다 보낸 요청의 수.
    pub fn served(&self) -> Vec<usize> {
        self.backends
            .iter()
            .map(|backend| backend.served.load(Ordering::Acquire))
            .collect()
    }

    // ANCHOR: forward
    /// 요청 하나를 고른 서버로 넘기고, 응답을 오는 대로 돌려보낸다.
    fn forward(&self, request: &Request, client: &mut TcpStream) -> std::io::Result<()> {
        let text = conversation(&request.body);
        let index = self.choose(&text);
        let backend = &self.backends[index];
        self.remember(index, &text);
        backend.busy.fetch_add(1, Ordering::AcqRel);
        backend.served.fetch_add(1, Ordering::AcqRel);

        let mut relay = || -> std::io::Result<()> {
            let mut upstream = TcpStream::connect(&backend.address)?;
            upstream.set_nodelay(true)?;
            write!(
                upstream,
                "{} {} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n",
                request.method,
                request.path,
                backend.address,
                request.body.len()
            )?;
            upstream.write_all(&request.body)?;
            upstream.flush()?;
            // 서버가 보내는 바이트를 받는 대로 넘긴다. 토큰이 흘러나오는
            // 응답이 라우터에서 고이지 않게, 모으지 않는다.
            let mut buffer = [0_u8; 16 * 1024];
            loop {
                let read = upstream.read(&mut buffer)?;
                if read == 0 {
                    return Ok(());
                }
                client.write_all(&buffer[..read])?;
            }
        };
        let outcome = relay();
        backend.busy.fetch_sub(1, Ordering::AcqRel);
        outcome
    }
    // ANCHOR_END: forward

    fn connection(&self, stream: TcpStream) -> std::io::Result<()> {
        stream.set_nodelay(true)?;
        let mut client = stream.try_clone()?;
        let mut reader = BufReader::new(stream);
        let request = match http::read_request(&mut reader, &mut client) {
            Ok(Some(request)) => request,
            Ok(None) => return Ok(()),
            Err(http::Error::Io(error)) => return Err(error),
            Err(_) => {
                return http::respond(&mut client, 400, "application/json", b"{}");
            }
        };
        if request.method == "GET" && request.path == "/router" {
            let served: Vec<String> = self.served().iter().map(usize::to_string).collect();
            let body = format!("{{\"served\":[{}]}}", served.join(","));
            return http::respond(&mut client, 200, "application/json", body.as_bytes());
        }
        if request.method == "GET" && request.path == "/health" {
            return http::respond(&mut client, 200, "application/json", b"{\"status\":\"ok\"}");
        }
        if let Err(error) = self.forward(&request, &mut client) {
            // 서버에 닿지 못했으면 502로 알린다. 응답을 넘기던 중이었다면
            // 이 줄은 쓰이지 못하고 연결이 끊긴다.
            let _ = http::respond(&mut client, 502, "application/json", b"{}");
            return Err(error);
        }
        Ok(())
    }

    /// 연결을 받는 대로 스레드 하나에 맡긴다. 돌아오지 않는다.
    pub fn serve(self: &Arc<Self>, listener: &TcpListener) -> std::io::Result<()> {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { continue };
            let router = Arc::clone(self);
            std::thread::spawn(move || {
                let _ = router.connection(stream);
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn router(policy: Policy) -> Router {
        let addresses: Vec<String> = (0..4)
            .map(|index| format!("127.0.0.1:{}", 9000 + index))
            .collect();
        Router::new(&addresses, policy)
    }

    #[test]
    fn reads_the_conversation_in_the_order_the_server_will() {
        let body = br#"{"messages":[{"role":"system","content":"Be brief."},{"role":"user","content":"Hi"}]}"#;
        assert_eq!(conversation(body), "system\nBe brief.\nuser\nHi\n");
        assert_eq!(conversation(b"not json"), "");
    }

    #[test]
    fn round_robin_spreads_and_prefix_hash_sticks() {
        let spread = router(Policy::RoundRobin);
        let chosen: Vec<usize> = (0..8).map(|_| spread.choose("same text")).collect();
        assert_eq!(chosen, [0, 1, 2, 3, 0, 1, 2, 3]);

        let sticky = router(Policy::PrefixHash { bytes: 16 });
        let first = sticky.choose("tenant A: rules and more rules, question one");
        // 앞 16바이트가 같으면 뒤가 달라도 같은 서버다.
        assert_eq!(
            sticky.choose("tenant A: rules and more rules, question two"),
            first
        );
        let others: Vec<usize> = (0..32)
            .map(|tenant| sticky.choose(&format!("tenant {tenant:02}: rules")))
            .collect();
        assert!((0..4).all(|index| others.contains(&index)), "{others:?}");
    }

    #[test]
    fn longest_prefix_follows_the_server_that_saw_the_text() {
        let aware = router(Policy::LongestPrefix { min: 8 });
        let first = "system\nRules of shop A.\nuser\nHello\n";
        let home = aware.choose(first);
        aware.remember(home, first);
        // 다른 가게의 대화는 다른(한가한) 서버로 간다. 겹치는 것이 임계값에 못 미친다.
        let other = "system\nA different shop.\nuser\nHello\n";
        aware.backends[home].busy.store(1, Ordering::Release);
        let elsewhere = aware.choose(other);
        assert_ne!(elsewhere, home);
        aware.remember(elsewhere, other);
        // 첫 대화가 이어지면, 바쁘더라도 그 대화를 본 서버로 돌아간다.
        let next = format!("{first}assistant\nHi!\nuser\nAnd then?\n");
        assert_eq!(aware.choose(&next), home);
        aware.remember(home, &next);
        // 이어진 글이 옛 글을 대신한다.
        assert_eq!(aware.backends[home].seen.lock().unwrap().len(), 1);
    }
}
