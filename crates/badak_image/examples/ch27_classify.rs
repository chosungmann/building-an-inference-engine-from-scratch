//! 27장: 사진 파일에서 이름까지. 전처리의 단계마다 Pillow와 견준다.
//!
//! 실행: `cargo run --release -p badak_image --example ch27_classify -- [모델]`
//!
//! 사진마다 세 가지를 본다. 파일을 푼 픽셀이 Pillow와 같은가, 줄이고 자른
//! 사진이 같은가, 그리고 그 차이가 모델의 점수를 얼마나 바꾸는가. 크기를
//! 줄이는 방법을 바꿔서도 해 본다.

use std::time::Instant;

use badak_formats::safetensors::SafeTensors;
use badak_formats::tflite::Model;
use badak_graph::{Graph, Interpreter, Registry, models_dir};
use badak_image::{CHANNELS, Image, Preprocess, Resampling};

const IMAGES: [&str; 5] = ["cup", "golden_retriever", "guitar", "pelican", "tabby_cat"];
/// 정답 파일이 담아 둔, 푼 사진의 왼쪽 위 귀퉁이의 한 변.
const CORNER: usize = 64;

/// 두 바이트열에서 다른 값의 비율과 가장 큰 차이.
fn differences(ours: &[u8], theirs: &[u8]) -> (f64, u8) {
    let different = ours.iter().zip(theirs).filter(|(a, b)| a != b).count();
    let worst = ours
        .iter()
        .zip(theirs)
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap_or(0);
    (100.0 * different as f64 / ours.len() as f64, worst)
}

/// 점수를 확률로 바꿔 높은 것부터 `count`개를 고른다.
fn top(scores: &[f32], count: usize) -> Vec<(usize, f32)> {
    let probabilities = badak_core::prob::softmax(scores);
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| probabilities[b].total_cmp(&probabilities[a]));
    order
        .into_iter()
        .take(count)
        .map(|index| (index, probabilities[index]))
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let name = std::env::args().nth(1).unwrap_or_else(|| "resnet50".into());
    let models = models_dir();
    let directory = models.join("vision_tflite");
    let reference = SafeTensors::open(directory.join("reference.safetensors"))?;
    let labels = std::fs::read_to_string(directory.join("labels.txt"))?;
    let labels: Vec<&str> = labels.lines().collect();
    let model = Model::open(directory.join(format!("{name}.tflite")))?;
    let graph = Graph::from_tflite(&model)?;
    let mut interpreter = Interpreter::new(&graph, &Registry::builtin())?;
    let settings = Preprocess::IMAGENET;
    println!("{name}.tflite\n");

    for image_name in IMAGES {
        let path = models.join(format!("images/{image_name}.jpg"));
        let started = Instant::now();
        let image = Image::open(&path)?;
        let decode = started.elapsed();
        println!("{image_name}.jpg ({} × {})", image.width, image.height);

        // 1. 파일을 푼 픽셀.
        let corner: Vec<u8> = (0..CORNER)
            .flat_map(|row| &image.pixels[row * image.width * CHANNELS..][..CORNER * CHANNELS])
            .copied()
            .collect();
        let expected = reference
            .tensor(&format!("{image_name}.decoded"))?
            .to_vec::<u8>()?;
        let (share, worst) = differences(&corner, &expected);
        println!("  푼 픽셀 (귀퉁이 64 × 64)   다른 값 {share:>5.1}%, 가장 큰 차이 {worst}");

        // 2. 줄이고 자른 사진.
        let started = Instant::now();
        let shaped = settings.shape(&image);
        let shape = started.elapsed();
        let expected = reference
            .tensor(&format!("{image_name}.shaped"))?
            .to_vec::<u8>()?;
        let (share, worst) = differences(&shaped.pixels, &expected);
        println!("  줄이고 자른 사진           다른 값 {share:>5.1}%, 가장 큰 차이 {worst}");

        // 3. 점수. Pillow가 만든 입력으로 LiteRT가 낸 점수와 견준다.
        let expected = reference
            .tensor(&format!("{name}.{image_name}.litert"))?
            .to_vec::<f32>()?;
        let theirs = top(&expected, 1)[0];
        println!(
            "  Pillow와 LiteRT            {:<26} {:>5.1}%",
            labels[theirs.0],
            100.0 * theirs.1
        );
        for (label, resampling) in [
            ("넓이를 보고 줄이기", Resampling::Area),
            ("둘만 보고 줄이기", Resampling::TwoTaps),
        ] {
            let input = Preprocess {
                resampling,
                ..settings
            }
            .apply(&image);
            interpreter.input_mut(0).copy_from_slice(&input);
            let started = Instant::now();
            interpreter.run()?;
            let run = started.elapsed();
            let scores = interpreter.output(0);
            let gap = scores
                .iter()
                .zip(&expected)
                .map(|(ours, theirs)| (ours - theirs).abs())
                .fold(0.0_f32, f32::max);
            let best = top(scores, 3);
            println!(
                "  {label:<10}       {:<26} {:>5.1}%  점수의 차이 {gap:.3}",
                labels[best[0].0],
                100.0 * best[0].1
            );
            if resampling == Resampling::Area {
                for &(index, probability) in &best[1..] {
                    println!(
                        "  {:<24} {:<26} {:>5.1}%",
                        "",
                        labels[index],
                        100.0 * probability
                    );
                }
                println!(
                    "  시간: 파일 풀기 {:.1}ms, 줄이고 자르기 {:.1}ms, 모델 {:.1}ms",
                    decode.as_secs_f64() * 1e3,
                    shape.as_secs_f64() * 1e3,
                    run.as_secs_f64() * 1e3
                );
            }
        }
        println!();
    }

    // 단계마다의 시간을 한 사진으로 여러 번 잰다.
    let path = models.join("images/tabby_cat.jpg");
    let bytes = std::fs::read(&path)?;
    let image = Image::decode_jpeg(&bytes)?;
    let shaped = settings.shape(&image);
    let input = settings.apply(&image);
    let median = |work: &mut dyn FnMut()| {
        let mut samples: Vec<f64> = (0..9)
            .map(|_| {
                let started = Instant::now();
                work();
                started.elapsed().as_secs_f64() * 1e3
            })
            .collect();
        samples.sort_by(f64::total_cmp);
        samples[samples.len() / 2]
    };
    println!(
        "tabby_cat.jpg ({} × {}), 아홉 번의 중앙값",
        image.width, image.height
    );
    println!(
        "  파일 풀기              {:>7.2} ms",
        median(&mut || {
            std::hint::black_box(Image::decode_jpeg(&bytes).unwrap());
        })
    );
    let (width, height) = settings.resized(image.width, image.height);
    println!(
        "  줄이기 (넓이를 보고)   {:>7.2} ms",
        median(&mut || {
            std::hint::black_box(image.resize(width, height, Resampling::Area));
        })
    );
    println!(
        "  줄이기 (둘만 보고)     {:>7.2} ms",
        median(&mut || {
            std::hint::black_box(image.resize(width, height, Resampling::TwoTaps));
        })
    );
    println!(
        "  자르고 숫자로 바꾸기   {:>7.2} ms",
        median(&mut || {
            std::hint::black_box(
                Preprocess {
                    resize: shaped.width,
                    ..settings
                }
                .apply(&shaped),
            );
        })
    );
    println!(
        "  모델                   {:>7.2} ms",
        median(&mut || {
            interpreter.input_mut(0).copy_from_slice(&input);
            interpreter.run().unwrap();
        })
    );
    Ok(())
}
