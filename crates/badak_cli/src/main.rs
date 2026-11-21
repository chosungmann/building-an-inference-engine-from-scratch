//! 실행 파일 `badak` (39장).
//!
//! 명령은 둘이다. `serve`는 추론 서버를 띄우고, `route`(42장)는 서버 여러
//! 대 앞에 서서 요청을 나눠 준다.
//!
//! ```text
//! badak route --backend 주소:포트 [--backend …] [--host 주소] [--port 번호]
//!             [--policy round_robin|least_busy|prefix_hash|longest_prefix]
//! ```
//!
//! ```text
//! badak serve [--model 이름이나 경로] [--host 주소] [--port 번호]
//!             [--cache-dir 디렉터리] [--context 토큰 수] [--no-warm]
//!             [--parallel 개] [--static 개] [--wait 밀리초] [--chunk 토큰 수]
//!             [--pages 개] [--keep 개]
//! ```
//!
//! `--model`에 이름을 주면 `models/` 아래에서 찾는다(기본은
//! `smollm2_135m`). `--host`의 기본은 `127.0.0.1`, 곧 이 기계 안에서만
//! 닿는 주소다. 인증이 없는 서버이므로 밖에서 닿는 주소(`0.0.0.0`)는
//! 쓰는 사람이 스스로 골라야 한다.
//!
//! 40장: `--parallel`을 주면 요청을 그 수까지 함께 처리한다(연속 배칭).
//! `--static`을 주면 그 수가 모이거나 `--wait`만큼 기다린 뒤에 함께
//! 시작한다(정적 배칭). `--chunk`는 글을 읽는 요청이 한 스텝에 넣는
//! 토큰의 한도, `--pages`는 KV 캐시의 페이지(토큰 16개씩)의 한도,
//! `--keep`은 끝난 요청의 캐시를 남겨 두는 개수다.

use std::net::TcpListener;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use badak_models::models_dir;
use badak_server::engine::Engine;
use badak_server::router::{Policy as RoutePolicy, Router};
use badak_server::scheduler::{Policy, Settings, pages_for};
use badak_server::server::Server;

const USAGE: &str = "사용법: badak serve [--model 이름이나 경로] [--host 주소] [--port 번호] \
                     [--cache-dir 디렉터리] [--context 토큰 수] [--no-warm] [--parallel 개] \
                     [--static 개] [--wait 밀리초] [--chunk 토큰 수] [--pages 개] [--keep 개]";

// ANCHOR: serve
fn serve(mut arguments: impl Iterator<Item = String>) -> Result<(), Box<dyn std::error::Error>> {
    let (mut model, mut host, mut port) =
        ("smollm2_135m".to_owned(), "127.0.0.1".to_owned(), 8080u16);
    let (mut cache_dir, mut context, mut warm) = (None, None, true);
    let (mut parallel, mut fixed, mut wait) = (None::<usize>, None::<usize>, 50u64);
    let (mut chunk, mut pages, mut keep) = (64usize, None::<usize>, 8usize);
    while let Some(flag) = arguments.next() {
        let mut value = || arguments.next().ok_or(format!("{flag} 뒤에 값이 없습니다"));
        match flag.as_str() {
            "--model" => model = value()?,
            "--host" => host = value()?,
            "--port" => port = value()?.parse()?,
            "--cache-dir" => cache_dir = Some(PathBuf::from(value()?)),
            "--context" => context = Some(value()?.parse()?),
            "--no-warm" => warm = false,
            "--parallel" => parallel = Some(value()?.parse()?),
            "--static" => fixed = Some(value()?.parse()?),
            "--wait" => wait = value()?.parse()?,
            "--chunk" => chunk = value()?.parse()?,
            "--pages" => pages = Some(value()?.parse()?),
            "--keep" => keep = value()?.parse()?,
            _ => return Err(format!("모르는 인자입니다: {flag}\n{USAGE}").into()),
        }
    }
    let directory = if model.contains('/') {
        PathBuf::from(&model)
    } else {
        models_dir().join(&model)
    };
    let mut engine = Engine::load(&directory)?;
    if let Some(context) = context {
        engine.context = context;
    }
    if let Some(directory) = &cache_dir {
        std::fs::create_dir_all(directory)?;
    }
    // 요청을 받기 전에 모델을 한 번 돌려 둔다. 첫 요청이 느리지 않게.
    if warm {
        println!("데우기:  {:.0}ms", engine.warm()? * 1e3);
    }
    let listener = TcpListener::bind((host.as_str(), port))?;
    // 포트를 0으로 주면 운영체제가 빈 포트를 고른다. 고른 것을 알려 준다.
    println!("듣는 곳: http://{}", listener.local_addr()?);
    println!("모델:    {}", engine.name);
    // 40장: 묶어 처리하라고 했으면 스케줄러를 띄운다.
    let policy = match (fixed, parallel) {
        (Some(size), _) => Some(Policy::Static {
            size: size.max(1),
            wait: Duration::from_millis(wait),
        }),
        (None, Some(max)) => Some(Policy::Continuous { max: max.max(1) }),
        (None, None) => None,
    };
    let context = engine.context;
    let mut server = Server::new(engine, cache_dir);
    if let Some(policy) = policy {
        let slots = fixed.or(parallel).unwrap_or(1) + keep;
        let settings = Settings {
            policy,
            chunk: chunk.max(1),
            pages: pages.unwrap_or_else(|| pages_for(context, slots)),
            keep,
        };
        println!(
            "묶기:    {policy:?}, 한 스텝에 읽는 토큰 {chunk}, 페이지 {}",
            settings.pages
        );
        server = server.with_scheduler(settings);
    }
    Arc::new(server).serve(&listener)?;
    Ok(())
}
// ANCHOR_END: serve

// ANCHOR: route
fn route(mut arguments: impl Iterator<Item = String>) -> Result<(), Box<dyn std::error::Error>> {
    let (mut host, mut port, mut backends) = ("127.0.0.1".to_owned(), 8080u16, Vec::new());
    let mut policy = RoutePolicy::LongestPrefix { min: 256 };
    while let Some(flag) = arguments.next() {
        let mut value = || arguments.next().ok_or(format!("{flag} 뒤에 값이 없습니다"));
        match flag.as_str() {
            "--backend" => backends.push(value()?),
            "--host" => host = value()?,
            "--port" => port = value()?.parse()?,
            "--policy" => {
                policy = match value()?.as_str() {
                    "round_robin" => RoutePolicy::RoundRobin,
                    "least_busy" => RoutePolicy::LeastBusy,
                    "prefix_hash" => RoutePolicy::PrefixHash { bytes: 256 },
                    "longest_prefix" => RoutePolicy::LongestPrefix { min: 256 },
                    other => return Err(format!("모르는 방법입니다: {other}").into()),
                }
            }
            _ => return Err(format!("모르는 인자입니다: {flag}\n{USAGE}").into()),
        }
    }
    if backends.is_empty() {
        return Err("--backend로 서버의 주소를 하나 넘게 주십시오".into());
    }
    let listener = TcpListener::bind((host.as_str(), port))?;
    println!("듣는 곳: http://{}", listener.local_addr()?);
    println!("서버:    {}", backends.join(", "));
    println!("방법:    {policy:?}");
    Arc::new(Router::new(&backends, policy)).serve(&listener)?;
    Ok(())
}
// ANCHOR_END: route

fn main() -> ExitCode {
    let mut arguments = std::env::args().skip(1);
    let done = match arguments.next().as_deref() {
        Some("serve") => serve(arguments),
        Some("route") => route(arguments),
        _ => Err(USAGE.into()),
    };
    match done {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}
