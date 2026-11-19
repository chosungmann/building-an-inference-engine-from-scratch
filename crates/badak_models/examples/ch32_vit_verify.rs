//! 32장: ViT의 계산을 PyTorch의 정답과 단계마다 견준다.
//!
//! 실행: `cargo run --release -p badak_models --example ch32_vit_verify`

use std::time::Instant;

use badak_core::Tensor;
use badak_formats::safetensors::SafeTensors;
use badak_models::models_dir;
use badak_models::vit::Vit;

const IMAGES: [&str; 5] = ["cup", "golden_retriever", "guitar", "pelican", "tabby_cat"];

type Outcome<T> = Result<T, Box<dyn std::error::Error>>;

/// 두 텐서의 가장 큰 차이와, 정답에서 가장 큰 절댓값.
fn gap(ours: &Tensor, theirs: &Tensor) -> Outcome<(f32, f32)> {
    let (ours, theirs) = (ours.to_vec::<f32>()?, theirs.to_vec::<f32>()?);
    if ours.len() != theirs.len() {
        return Err(format!("값의 수가 다릅니다: {} 대 {}", ours.len(), theirs.len()).into());
    }
    let largest = ours
        .iter()
        .zip(&theirs)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    let scale = theirs
        .iter()
        .fold(0.0_f32, |peak, value| peak.max(value.abs()));
    Ok((largest, scale))
}

fn main() -> Outcome<()> {
    let directory = models_dir().join("vision_transformer");
    let reference = SafeTensors::open(directory.join("reference.safetensors"))?;
    let labels = std::fs::read_to_string(models_dir().join("vision_tflite/labels.txt"))?;
    let labels: Vec<&str> = labels.lines().collect();
    let start = Instant::now();
    let model = Vit::load(directory.join("vit_tiny.safetensors"), 3)?;
    println!(
        "ViT-Tiny: 사진 {0} × {0}, 토큰 {1}개, 폭 {2}, 블록 {3}개 (읽는 데 {4:.1}ms)",
        model.image_size(),
        model.tokens(),
        model.width(),
        model.depth(),
        start.elapsed().as_secs_f64() * 1e3
    );

    // 첫 사진은 단계마다 견준다.
    let input = reference.tensor("vit.cup.input")?.to_vec::<f32>()?;
    let trace = model.forward(&input)?;
    println!("\ncup: 단계마다의 가장 큰 차이");
    println!("  {:<22} {:>10} {:>12}", "단계", "차이", "값의 크기");
    let last = model.depth() - 1;
    let steps = [
        ("조각을 벡터로", "patches", &trace.patches),
        ("위치를 더한 뒤", "embedded", &trace.embedded),
        ("첫 블록 뒤", "block0", &trace.blocks[0]),
        ("마지막 블록 뒤", "block11", &trace.blocks[last]),
        ("마지막 정규화 뒤", "normed", &trace.normed),
        ("점수", "logits", &trace.logits),
    ];
    for (label, key, ours) in steps {
        let (largest, scale) = gap(ours, &reference.tensor(&format!("vit.cup.{key}"))?)?;
        println!("  {label:<22} {largest:>10.2e} {scale:>12.2}");
    }

    println!("\n사진마다");
    println!(
        "  {:<18} {:<28} {:>10} {:>9}",
        "사진", "고른 이름", "PyTorch와", "시간"
    );
    for image in IMAGES {
        let input = reference
            .tensor(&format!("vit.{image}.input"))?
            .to_vec::<f32>()?;
        let start = Instant::now();
        let trace = model.forward(&input)?;
        let elapsed = start.elapsed();
        let (largest, _) = gap(
            &trace.logits,
            &reference.tensor(&format!("vit.{image}.logits"))?,
        )?;
        let scores = trace.logits.to_vec::<f32>()?;
        let best = (0..scores.len())
            .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
            .expect("점수가 있다");
        println!(
            "  {image:<18} {:<28} {largest:>10.2e} {:>7.1}ms",
            labels[best],
            elapsed.as_secs_f64() * 1e3
        );
        if largest.is_nan() || largest > 2e-3 {
            return Err(format!("{image}: PyTorch와 다릅니다").into());
        }
    }

    // 빠른 길(SME 커널, 근사한 지수 함수)의 답도 견준다.
    let model = model.fast()?;
    println!("\n빠른 길");
    for image in IMAGES {
        let input = reference
            .tensor(&format!("vit.{image}.input"))?
            .to_vec::<f32>()?;
        let start = Instant::now();
        let trace = model.forward(&input)?;
        let elapsed = start.elapsed();
        let expected = reference.tensor(&format!("vit.{image}.logits"))?;
        let (largest, _) = gap(&trace.logits, &expected)?;
        println!(
            "  {image:<18} {:<28} {largest:>10.2e} {:>7.1}ms",
            "",
            elapsed.as_secs_f64() * 1e3
        );
        if largest.is_nan() || largest > 2e-3 {
            return Err(format!("{image}: 빠른 길이 PyTorch와 다릅니다").into());
        }
    }

    // ViT-B/16. 같은 코드가 더 큰 가중치를 읽는다. 파일이 있을 때만 한다
    // (sh scripts/download_ch32.sh base).
    let path = directory.join("vit_base.safetensors");
    if !path.exists() {
        println!("\nViT-B/16: {}이 없어 건너뜁니다", path.display());
        return Ok(());
    }
    let base = Vit::load(&path, 12)?;
    println!(
        "\nViT-B/16: 토큰 {}개, 폭 {}, 블록 {}개",
        base.tokens(),
        base.width(),
        base.depth()
    );
    let quick = Vit::load(&path, 12)?.fast()?;
    println!(
        "  {:<18} {:<28} {:>10} {:>9} {:>10} {:>9}",
        "사진", "고른 이름", "PyTorch와", "시간", "빠른 길", "시간"
    );
    for image in IMAGES {
        let input = reference
            .tensor(&format!("vit.{image}.input"))?
            .to_vec::<f32>()?;
        let expected = reference.tensor(&format!("vit_base.{image}.logits"))?;
        let mut gaps = [0.0_f32; 2];
        let mut times = [0.0_f64; 2];
        let mut best = 0;
        for (index, model) in [&base, &quick].into_iter().enumerate() {
            let start = Instant::now();
            let trace = model.forward(&input)?;
            times[index] = start.elapsed().as_secs_f64() * 1e3;
            gaps[index] = gap(&trace.logits, &expected)?.0;
            let scores = trace.logits.to_vec::<f32>()?;
            best = (0..scores.len())
                .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
                .expect("점수가 있다");
        }
        println!(
            "  {image:<18} {:<28} {:>10.2e} {:>7.1}ms {:>10.2e} {:>7.1}ms",
            labels[best], gaps[0], times[0], gaps[1], times[1]
        );
        if gaps.iter().any(|&gap| gap.is_nan() || gap > 2e-3) {
            return Err(format!("{image}: ViT-B/16이 PyTorch와 다릅니다").into());
        }
    }
    Ok(())
}
