//! 32장: ViT가 사진 한 장을 분류하는 시간과, 그 시간이 어디에 드는지.
//!
//! 실행: `cargo run --release -p badak_models --example ch32_vit_bench [base] [fast]`
//!
//! `fast`를 주면 31장의 SME 커널과 근사한 지수 함수를 쓴다.
//!
//! 스레드의 수는 환경 변수 `BADAK_THREADS`로 정한다.

use std::hint::black_box;

use badak_bench::stats::Summary;
use badak_bench::timing::{Plan, measure};
use badak_core::{profile, threads};
use badak_formats::safetensors::SafeTensors;
use badak_models::models_dir;
use badak_models::vit::Vit;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let directory = models_dir().join("vision_transformer");
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let fast = arguments.iter().any(|argument| argument == "fast");
    // `base`를 주면 ViT-B/16(헤드 12개)을, 아니면 ViT-Tiny(헤드 3개)를 잰다.
    let (name, heads) = if arguments.iter().any(|argument| argument == "base") {
        ("vit_base", 12)
    } else {
        ("vit_tiny", 3)
    };
    let model = Vit::load(directory.join(format!("{name}.safetensors")), heads)?;
    let model = if fast { model.fast()? } else { model };
    let reference = SafeTensors::open(directory.join("reference.safetensors"))?;
    let input = reference.tensor("vit.cup.input")?.to_vec::<f32>()?;
    let plan = Plan {
        warmup: 5,
        repetitions: 51,
    };
    let samples = measure(plan, || {
        black_box(model.forward(&input).expect("실행").logits);
    });
    let summary = Summary::new(&samples);
    println!(
        "{name} {} 스레드 {:>2} 중앙값 {:>7.2}ms 최소 {:>7.2}ms",
        if fast { "빠른 길" } else { "기본   " },
        threads::global().pool().threads(),
        summary.median * 1e3,
        summary.min * 1e3
    );

    // 14장의 프로파일러로 연산마다의 시간을 모은다.
    let rounds = 20;
    profile::reset();
    profile::enable(true);
    for _ in 0..rounds {
        black_box(model.forward(&input)?.logits);
    }
    profile::enable(false);
    for (name, record) in profile::report() {
        println!(
            "  {name:<12} {:>6}번 {:>7.2}ms",
            record.calls / rounds,
            record.total.as_secs_f64() * 1e3 / rounds as f64
        );
    }
    Ok(())
}
