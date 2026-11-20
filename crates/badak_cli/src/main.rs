//! 실행 파일 `badak` (39장).
//!
//! 지금은 명령이 하나다.
//!
//! ```text
//! badak serve [--model 이름이나 경로] [--host 주소] [--port 번호]
//!             [--cache-dir 디렉터리] [--context 토큰 수] [--no-warm]
//! ```
//!
//! `--model`에 이름을 주면 `models/` 아래에서 찾는다(기본은
//! `smollm2_135m`). `--host`의 기본은 `127.0.0.1`, 곧 이 기계 안에서만
//! 닿는 주소다. 인증이 없는 서버이므로 밖에서 닿는 주소(`0.0.0.0`)는
//! 쓰는 사람이 스스로 골라야 한다.

use std::net::TcpListener;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use badak_models::models_dir;
use badak_server::engine::Engine;
use badak_server::server::Server;

const USAGE: &str = "사용법: badak serve [--model 이름이나 경로] [--host 주소] [--port 번호] \
                     [--cache-dir 디렉터리] [--context 토큰 수] [--no-warm]";

// ANCHOR: serve
fn serve(mut arguments: impl Iterator<Item = String>) -> Result<(), Box<dyn std::error::Error>> {
    let (mut model, mut host, mut port) =
        ("smollm2_135m".to_owned(), "127.0.0.1".to_owned(), 8080u16);
    let (mut cache_dir, mut context, mut warm) = (None, None, true);
    while let Some(flag) = arguments.next() {
        let mut value = || arguments.next().ok_or(format!("{flag} 뒤에 값이 없습니다"));
        match flag.as_str() {
            "--model" => model = value()?,
            "--host" => host = value()?,
            "--port" => port = value()?.parse()?,
            "--cache-dir" => cache_dir = Some(PathBuf::from(value()?)),
            "--context" => context = Some(value()?.parse()?),
            "--no-warm" => warm = false,
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
    Arc::new(Server::new(engine, cache_dir)).serve(&listener)?;
    Ok(())
}
// ANCHOR_END: serve

fn main() -> ExitCode {
    let mut arguments = std::env::args().skip(1);
    let done = match arguments.next().as_deref() {
        Some("serve") => serve(arguments),
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
