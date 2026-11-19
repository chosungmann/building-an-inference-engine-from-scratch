//! ViT-Tiny를 PyTorch의 정답과 견준다 (32장).
//!
//! 파일이 없으면 건너뛴다. `sh scripts/download_ch32.sh`로 받는다.

use badak_formats::safetensors::SafeTensors;
use badak_models::models_dir;
use badak_models::vit::Vit;

const IMAGES: [&str; 5] = ["cup", "golden_retriever", "guitar", "pelican", "tabby_cat"];

fn largest_gap(ours: &[f32], theirs: &[f32]) -> f32 {
    assert_eq!(ours.len(), theirs.len());
    ours.iter()
        .zip(theirs)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

#[test]
fn matches_pytorch_on_both_paths() {
    let directory = models_dir().join("vision_transformer");
    if !directory.join("vit_tiny.safetensors").exists() {
        eprintln!("건너뜀: {}에 모델이 없습니다", directory.display());
        return;
    }
    let reference = SafeTensors::open(directory.join("reference.safetensors")).unwrap();
    let tensor = |key: &str| reference.tensor(key).unwrap().to_vec::<f32>().unwrap();
    let model = Vit::load(directory.join("vit_tiny.safetensors"), 3).unwrap();
    assert_eq!(
        (
            model.image_size(),
            model.tokens(),
            model.width(),
            model.depth()
        ),
        (224, 197, 192, 12)
    );

    // 첫 사진은 단계마다 견준다.
    let trace = model.forward(&tensor("vit.cup.input")).unwrap();
    assert_eq!(trace.patches.shape(), [196, 192]);
    assert_eq!(trace.blocks.len(), 12);
    let steps = [
        ("patches", &trace.patches, 1e-4),
        ("embedded", &trace.embedded, 1e-4),
        ("block0", &trace.blocks[0], 2e-4),
        // 마지막 블록의 값은 수백까지 커진다.
        ("block11", &trace.blocks[11], 2e-2),
        ("normed", &trace.normed, 2e-3),
        ("logits", &trace.logits, 2e-4),
    ];
    for (key, ours, tolerance) in steps {
        let gap = largest_gap(
            &ours.to_vec::<f32>().unwrap(),
            &tensor(&format!("vit.cup.{key}")),
        );
        assert!(gap < tolerance, "{key}: {gap}");
    }

    let fast = Vit::load(directory.join("vit_tiny.safetensors"), 3)
        .unwrap()
        .fast()
        .unwrap();
    for image in IMAGES {
        let (input, expected) = (
            tensor(&format!("vit.{image}.input")),
            tensor(&format!("vit.{image}.logits")),
        );
        let best = |scores: &[f32]| {
            (0..scores.len())
                .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
                .unwrap()
        };
        for model in [&model, &fast] {
            let scores = model
                .forward(&input)
                .unwrap()
                .logits
                .to_vec::<f32>()
                .unwrap();
            assert_eq!(best(&scores), best(&expected), "{image}");
            assert!(largest_gap(&scores, &expected) < 2e-4, "{image}");
        }
    }
    // 크기가 맞지 않는 사진은 거절한다.
    assert!(model.forward(&[0.0; 10]).is_err());
}

#[test]
fn the_same_code_runs_the_base_model() {
    let directory = models_dir().join("vision_transformer");
    let path = directory.join("vit_base.safetensors");
    if !path.exists() {
        eprintln!("건너뜀: {}이 없습니다", path.display());
        return;
    }
    let reference = SafeTensors::open(directory.join("reference.safetensors")).unwrap();
    let tensor = |key: &str| reference.tensor(key).unwrap().to_vec::<f32>().unwrap();
    let model = Vit::load(&path, 12).unwrap();
    assert_eq!(
        (
            model.image_size(),
            model.tokens(),
            model.width(),
            model.depth()
        ),
        (224, 197, 768, 12)
    );
    let fast = Vit::load(&path, 12).unwrap().fast().unwrap();
    for image in ["cup", "tabby_cat"] {
        let input = tensor(&format!("vit.{image}.input"));
        let expected = tensor(&format!("vit_base.{image}.logits"));
        for model in [&model, &fast] {
            let scores = model
                .forward(&input)
                .unwrap()
                .logits
                .to_vec::<f32>()
                .unwrap();
            assert!(largest_gap(&scores, &expected) < 5e-4, "{image}");
        }
    }
}
