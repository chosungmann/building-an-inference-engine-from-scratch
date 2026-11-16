//! 언어 모델의 첫 블록을 이루는 부품을 PyTorch의 계산과 대조한다.
//!
//! 모델 파일이 없으면 건너뛴다. `sh scripts/download_smollm2.sh` 또는
//! `sh scripts/download_llama32.sh`로 받는다.

use badak_core::approx::compare;
use badak_core::{Tensor, ops};
use badak_formats::gguf::Gguf;
use badak_formats::safetensors::SafeTensors;
use badak_models::llama::attention::attention;
use badak_models::llama::{LlamaFiles, mlp, rope};
use badak_models::models_dir;
use badak_models::tokenizer::Tokenizer;

fn assert_close(name: &str, actual: &Tensor, expected: &Tensor) {
    let (actual, expected) = (
        actual.to_vec::<f32>().unwrap(),
        expected.to_vec::<f32>().unwrap(),
    );
    let scale = expected
        .iter()
        .fold(0.0_f32, |max, value| max.max(value.abs()));
    let report = compare(&actual, &expected, 1e-5, 1e-5 * scale);
    assert!(report.is_close(), "{name}: {report}");
}

/// 모델과 정답 파일을 연다. 없으면 `None`.
fn open(name: &str) -> Option<(LlamaFiles, SafeTensors)> {
    let directory = models_dir().join(name);
    let reference = directory.join("reference.safetensors");
    if !directory.join("model.safetensors").exists() || !reference.exists() {
        eprintln!(
            "건너뜀: {}에 모델이나 정답 파일이 없습니다",
            directory.display()
        );
        return None;
    }
    Some((
        LlamaFiles::open(&directory).unwrap(),
        SafeTensors::open(reference).unwrap(),
    ))
}

fn check_first_block(name: &str) {
    let Some((files, reference)) = open(name) else {
        return;
    };
    let expected = |tensor: &str| reference.tensor(tensor).unwrap();
    let c = &files.config;
    let block = files.block(0).unwrap();
    let frequencies = rope::inverse_frequencies(c);

    let embedded = ops::embedding(&files.embedding().unwrap(), &expected("ids")).unwrap();
    assert_close("임베딩", &embedded, &expected("embed"));

    let normed = ops::rms_norm(&embedded, &block.attention_norm, c.rms_norm_eps).unwrap();
    assert_close("정규화 1", &normed, &expected("l0.norm1"));
    assert_close(
        "질의",
        &ops::linear(&normed, &block.q_proj, None).unwrap(),
        &expected("l0.q"),
    );
    assert_close(
        "키",
        &ops::linear(&normed, &block.k_proj, None).unwrap(),
        &expected("l0.k"),
    );
    assert_close(
        "값",
        &ops::linear(&normed, &block.v_proj, None).unwrap(),
        &expected("l0.v"),
    );

    let attended = attention(&normed, &block, c, &frequencies).unwrap();
    assert_close(
        "어텐션 비중",
        &attended.weights,
        &expected("l0.attn_weights"),
    );
    assert_close("어텐션 출력", &attended.output, &expected("l0.attn"));

    // 이번에는 부품을 이어서 블록 전체를 계산한다. 오차가 쌓여도 허용 범위
    // 안이어야 한다.
    let after_attention = ops::add(&embedded, &attended.output).unwrap();
    let normed = ops::rms_norm(&after_attention, &block.mlp_norm, c.rms_norm_eps).unwrap();
    assert_close("정규화 2", &normed, &expected("l0.norm2"));
    let mlp_out = mlp(&normed, &block).unwrap();
    assert_close("MLP", &mlp_out, &expected("l0.mlp"));
    assert_close(
        "블록 출력",
        &ops::add(&after_attention, &mlp_out).unwrap(),
        &expected("l0.out"),
    );

    // 어텐션 비중은 토큰마다 합이 1이고, 뒤의 토큰에는 0이다.
    let tokens = expected("ids").len();
    let weights = attended.weights.to_vec::<f32>().unwrap();
    for row in weights
        .chunks_exact(tokens)
        .enumerate()
        .map(|(index, row)| (index % tokens, row))
    {
        let (token, row) = row;
        assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(row[token + 1..].iter().all(|&weight| weight == 0.0));
    }
}

#[test]
fn smollm2_first_block_matches_pytorch() {
    check_first_block("smollm2_135m");
}

#[test]
fn llama32_first_block_matches_pytorch() {
    check_first_block("llama32_1b");
}

#[test]
fn chapter_8_tokenizer_agrees_with_the_model_authors_tokenizer() {
    // 정답 파일의 토큰 번호는 Hugging Face의 tokenizers 라이브러리가 만든 것이다.
    let Some((_, reference)) = open("llama32_1b") else {
        return;
    };
    let vocab = models_dir().join("llama_cpp_vocab/ggml-vocab-llama-bpe.gguf");
    if !vocab.exists() {
        eprintln!("건너뜀: {}가 없습니다", vocab.display());
        return;
    }
    let tokenizer = Tokenizer::from_gguf(&Gguf::open(vocab).unwrap()).unwrap();
    let ids: Vec<i32> = tokenizer
        .encode(&reference.metadata()["prompt"], false)
        .iter()
        .map(|&id| id as i32)
        .collect();
    assert_eq!(
        ids,
        reference.tensor("ids").unwrap().to_vec::<i32>().unwrap()
    );
}
