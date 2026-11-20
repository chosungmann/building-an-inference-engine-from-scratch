//! `badak`의 추론 서버 (39장).
//!
//! 지금까지의 엔진은 프로그램 하나가 불러 쓰는 함수였다. 이 크레이트는 그
//! 엔진을 **다른 프로그램이 네트워크로 부르는 것**으로 만든다. HTTP를 직접
//! 읽고 쓰고, 토큰을 쓰는 대로 흘려보내고, 널리 쓰이는 API의 꼴을 따른다.

pub mod api;
pub mod client;
pub mod engine;
pub mod http;
pub mod router;
pub mod scheduler;
pub mod server;
