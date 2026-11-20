//! 44장: 모델 파일을 새로 받았을 때 생기는 일들을 차례로 겪어 본다.
//!
//! 실행: `cargo run --release -p badak_ffi --example ch44_update`
//!
//! 내려받기는 흉내만 낸다(파일을 복사한다). 받은 뒤의 일, 곧 확인하고,
//! 놓고, 돌려 보고, 안 되면 되돌리는 일은 진짜다. 모델은 27장의 것을 쓴다.

use std::path::{Path, PathBuf};
use std::time::Instant;

use badak_ffi::{Classifier, Score};
use badak_formats::store::{ENGINE_GENERATION, Manifest, Store};
use badak_graph::models_dir;

/// ImageNet의 부류 번호: 컵.
const CUP: u32 = 968;

/// 놓인 모델로 아는 사진을 분류해 본다. 새 모델을 믿기 전에 하는 시험이다.
fn self_test(store: &Store, jpeg: &[u8]) -> Result<Score, Box<dyn std::error::Error>> {
    let (_, path) = store.current("classifier")?.ok_or("놓인 모델이 없습니다")?;
    let mut classifier = Classifier::open(path)?;
    let mut top = [Score::default(); 1];
    classifier.classify_jpeg(jpeg, &mut top)?;
    if top[0].index != CUP {
        return Err(format!("컵 사진을 {}번 부류라고 합니다", top[0].index).into());
    }
    Ok(top[0])
}

/// 파일을 "받는다". 서버에 있는 파일을 받아 둘 자리로 복사한다.
fn download(from: &Path, scratch: &Path) -> std::io::Result<PathBuf> {
    let arrived = scratch.join("arrived.partial");
    std::fs::copy(from, &arrived)?;
    Ok(arrived)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let models = models_dir().join("vision_tflite");
    let jpeg = std::fs::read(models_dir().join("images/cup.jpg"))?;
    let scratch = std::env::temp_dir().join(format!("badak_ch44_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&scratch);
    std::fs::create_dir_all(&scratch)?;
    let store = Store::new(scratch.join("models"));
    let megabytes = |path: &Path| std::fs::metadata(path).map(|m| m.len() as f64 / 1e6);

    // 서버 쪽: 두 판의 명세를 만든다. 파일 전체의 해시를 계산한다.
    let small = models.join("mobilenetv2.tflite");
    let large = models.join("resnet50.tflite");
    let started = Instant::now();
    let first = Manifest::describe("classifier", "2026-11-01", &small)?;
    let hashed_small = started.elapsed();
    let started = Instant::now();
    let second = Manifest::describe("classifier", "2026-11-21", &large)?;
    let hashed_large = started.elapsed();
    println!("명세 (서버가 만든다)");
    print!("  {}", first.to_json());
    print!("  {}", second.to_json());
    println!(
        "  해시를 계산하는 데: {:.1}MB에 {:.0}ms, {:.1}MB에 {:.0}ms\n",
        megabytes(&small)?,
        hashed_small.as_secs_f64() * 1e3,
        megabytes(&large)?,
        hashed_large.as_secs_f64() * 1e3
    );

    println!("1. 첫 판을 받아 놓는다");
    store.install(&first, download(&small, &scratch)?)?;
    let score = self_test(&store, &jpeg)?;
    println!(
        "   놓았습니다. 컵 사진: 컵 {:.1}%\n",
        100.0 * score.probability
    );

    println!("2. 새 판을 받다가 끊겼다");
    let arrived = download(&large, &scratch)?;
    let bytes = std::fs::read(&arrived)?;
    std::fs::write(&arrived, &bytes[..bytes.len() * 2 / 3])?;
    let error = store.install(&second, &arrived).unwrap_err();
    println!("   거절: {error}");

    println!("3. 다 받았는데 비트 하나가 다르다");
    let mut flipped = bytes.clone();
    flipped[50_000_000] ^= 0x01;
    std::fs::write(&arrived, &flipped)?;
    let started = Instant::now();
    let error = store.install(&second, &arrived).unwrap_err();
    println!("   거절: {error}");
    println!(
        "   (알아내는 데 {:.0}ms)",
        started.elapsed().as_secs_f64() * 1e3
    );

    println!("4. 더 새로운 엔진을 위한 파일이다");
    std::fs::write(&arrived, &bytes)?;
    let newer = Manifest {
        requires: ENGINE_GENERATION + 1,
        ..second.clone()
    };
    let error = store.install(&newer, &arrived).unwrap_err();
    println!("   거절: {error}");
    println!(
        "   그동안 쓰는 판: {}\n",
        store.current_version("classifier")?.unwrap_or_default()
    );

    println!("5. 해시는 맞는데 돌지 않는 파일이다");
    // 서버에서 누가 파일을 잘못 올렸다. 명세는 그 파일로 만들었으므로
    // 해시는 맞는다.
    let wrong = scratch.join("mobilenetv2.tflite");
    std::fs::copy(models.join("labels.txt"), &wrong)?;
    let broken = Manifest::describe("classifier", "2026-11-15", &wrong)?;
    store.install(&broken, &wrong)?;
    println!(
        "   놓았습니다 (해시가 맞으므로). 지금 쓰는 판: {}",
        broken.version
    );
    match self_test(&store, &jpeg) {
        Ok(_) => println!("   돌았습니다?"),
        Err(error) => println!("   돌려 보니: {error}"),
    }
    let rolled = store.rollback("classifier")?;
    println!(
        "   되돌렸습니다({rolled}). 지금 쓰는 판: {}",
        store.current_version("classifier")?.unwrap_or_default()
    );
    let score = self_test(&store, &jpeg)?;
    println!("   컵 사진: 컵 {:.1}%\n", 100.0 * score.probability);

    println!("6. 제대로 된 새 판");
    store.install(&second, &arrived)?;
    let score = self_test(&store, &jpeg)?;
    println!(
        "   놓았습니다. 지금 쓰는 판: {}. 컵 사진: 컵 {:.1}%",
        second.version,
        100.0 * score.probability
    );
    println!("   지운 옛 판의 폴더: {}개", store.prune("classifier")?);
    let mut left: Vec<String> = std::fs::read_dir(scratch.join("models/classifier"))?
        .map(|entry| entry.map(|entry| entry.file_name().to_string_lossy().into_owned()))
        .collect::<Result<_, _>>()?;
    left.sort();
    println!("   남은 것: {}", left.join(", "));

    std::fs::remove_dir_all(&scratch)?;
    Ok(())
}
