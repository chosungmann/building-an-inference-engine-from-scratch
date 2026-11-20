//! 모델을 줄이는 도구들이 엔진 안에서 맞게 도는지 본다 (46장).
//!
//! `models/smollm2_135m`이 있어야 돈다. 없으면 건너뛴다.

use std::sync::Arc;

use badak_core::Tensor;
use badak_core::packed::{Delegated, Matrix};
use badak_core::sparse::{Sparse, prune_magnitude};
use badak_models::llama::WeightFormat;
use badak_models::llama::cache::KvCache;
use badak_models::llama::model::LlamaModel;
use badak_models::models_dir;
use badak_models::tokenizer::Tokenizer;

fn open() -> Option<(LlamaModel, Vec<u32>)> {
    let directory = models_dir().join("smollm2_135m");
    if !directory.join("model.safetensors").exists() {
        eprintln!("건너뜀: {}에 모델이 없습니다", directory.display());
        return None;
    }
    let tokenizer = Tokenizer::from_tokenizer_json(directory.join("tokenizer.json")).unwrap();
    let mut ids = tokenizer.prefix().to_vec();
    ids.extend(tokenizer.encode("The capital of France is", false));
    Some((
        LlamaModel::load_as(&directory, WeightFormat::F32).unwrap(),
        ids,
    ))
}

fn logits(model: &LlamaModel, ids: &[u32]) -> Vec<f32> {
    let mut cache = KvCache::new(&model.config);
    let logits = model.forward_cached(ids, &mut cache).unwrap();
    logits.as_slice::<f32>().unwrap().to_vec()
}

fn largest_gap(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

#[test]
fn a_model_with_fewer_blocks_still_runs() {
    let Some((mut model, ids)) = open() else {
        return;
    };
    let whole = logits(&model, &ids);
    let count = model.config.num_layers;
    model.retain_blocks(|index| index != 14);
    assert_eq!(model.config.num_layers, count - 1);
    let fewer = logits(&model, &ids);
    assert_eq!(fewer.len(), whole.len());
    assert!(fewer.iter().all(|value| value.is_finite()));
    // 블록 하나가 빠졌으니 점수가 달라진다. 그래도 같은 모델의 것이다.
    let gap = largest_gap(&whole, &fewer);
    assert!(gap > 1e-3 && gap < 20.0, "{gap}");
}

#[test]
fn the_sparse_kernel_gives_the_scores_of_the_dense_kernel() {
    let Some((_, ids)) = open() else { return };
    let directory = models_dir().join("smollm2_135m");
    // 같은 가중치를 깎아, 한쪽은 보통의 행렬로 다른 쪽은 희소 커널로 돌린다.
    let build = |sparse: bool| {
        let mut model = LlamaModel::load_as(&directory, WeightFormat::F32).unwrap();
        model
            .replace_matrices(|kind, _, matrix| {
                if kind == "embedding" {
                    return Ok(matrix.clone());
                }
                let shape = matrix.shape().to_vec();
                let mut values = matrix.to_f32()?.to_vec::<f32>()?;
                prune_magnitude(&mut values, 0.3);
                let dense: Matrix = Tensor::from_vec(values.clone(), &shape)?.into();
                if sparse {
                    let kernel = Sparse::from_dense(&values, shape[0], shape[1])?;
                    assert!(kernel.nonzeros() * 10 <= values.len() * 7 + 10);
                    Ok(Matrix::Delegated(Delegated::new(dense, Arc::new(kernel))))
                } else {
                    Ok(dense)
                }
            })
            .unwrap();
        model
    };
    let dense = logits(&build(false), &ids);
    let sparse = logits(&build(true), &ids);
    // 더하는 순서가 달라 비트까지 같지는 않다.
    let gap = largest_gap(&dense, &sparse);
    assert!(gap < 2e-2, "{gap}");
    let best = |scores: &[f32]| (0..scores.len()).max_by(|&a, &b| scores[a].total_cmp(&scores[b]));
    assert_eq!(best(&dense), best(&sparse));
}
