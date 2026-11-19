//! all-MiniLM-L6-v2를 transformers의 정답과 견준다 (34장).
//!
//! 파일이 없으면 건너뛴다. `sh scripts/download_ch34.sh`로 받는다.

use badak_formats::safetensors::SafeTensors;
use badak_models::bert::{Batching, Bert, Plan};
use badak_models::models_dir;
use badak_models::search::Index;
use badak_models::tokenizer::TokenId;
use badak_models::tokenizer::wordpiece::WordPiece;

fn largest_gap(ours: &[f32], theirs: &[f32]) -> f32 {
    assert_eq!(ours.len(), theirs.len());
    ours.iter()
        .zip(theirs)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

#[test]
fn tokens_and_embeddings_match_the_reference() {
    let directory = models_dir().join("minilm");
    if !directory.join("model.safetensors").exists() {
        eprintln!("건너뜀: {}에 모델이 없습니다", directory.display());
        return;
    }
    let reference = SafeTensors::open(directory.join("reference.safetensors")).unwrap();
    let floats = |key: &str| reference.tensor(key).unwrap().to_vec::<f32>().unwrap();
    let tokenizer = WordPiece::from_vocab(directory.join("vocab.txt")).unwrap();
    assert_eq!(tokenizer.vocab_size(), 30522);
    assert_eq!(tokenizer.padding(), 0);

    // 토크나이저. 악센트, 한글, 한자, 그림 문자, 긴 글이 들어 있다.
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference/ch34_sentences.txt"),
    )
    .unwrap();
    let mut encoded: Vec<Vec<TokenId>> = Vec::new();
    for (index, sentence) in text.lines().enumerate() {
        let ids = tokenizer.encode(sentence, 256);
        let expected = reference
            .tensor(&format!("ids.{index}"))
            .unwrap()
            .to_vec::<i32>()
            .unwrap();
        assert!(
            ids.iter().map(|&id| id as i32).eq(expected.iter().copied()),
            "{index}번 글의 토큰이 다릅니다: {sentence:?}"
        );
        encoded.push(ids);
    }
    assert_eq!(encoded.len(), 16);
    assert_eq!(encoded[15].len(), 256);

    let model = Bert::load(&directory).unwrap();
    assert_eq!(
        (model.width(), model.depth(), model.longest()),
        (384, 6, 512)
    );

    // 첫 글은 단계마다.
    let trace = model.forward(&[&encoded[0]], Batching::Ragged, 0).unwrap();
    assert_eq!(trace.embedded.shape(), [encoded[0].len(), 384]);
    let embedded = trace.embedded.to_vec::<f32>().unwrap();
    assert!(largest_gap(&embedded, &floats("embedded.0")) < 1e-5);
    for (layer, output) in trace.layers.iter().enumerate() {
        let output = output.to_vec::<f32>().unwrap();
        assert!(largest_gap(&output, &floats(&format!("layer{layer}.0"))) < 5e-5);
    }

    // 묶는 방법이 달라도 임베딩은 같다.
    let expected = floats("pooled");
    let plan = |batch, batching, sorted, across| Plan {
        batch,
        batching,
        sorted,
        across,
    };
    let one = model
        .embed_many(&encoded, 0, plan(1, Batching::Ragged, false, false))
        .unwrap();
    assert!(largest_gap(&one, &expected) < 1e-5);
    for plan in [
        plan(16, Batching::Padded, false, false),
        plan(16, Batching::Ragged, false, false),
        plan(4, Batching::Padded, true, false),
        plan(4, Batching::Ragged, true, true),
        plan(5, Batching::Padded, false, true),
    ] {
        let batched = model.embed_many(&encoded, 0, plan).unwrap();
        assert!(largest_gap(&batched, &one) < 1e-5, "{plan:?}");
    }
    let padded = model
        .embed_many(&encoded, 0, plan(16, Batching::Padded, false, false))
        .unwrap();
    assert!(largest_gap(&padded, &floats("pooled_padded")) < 1e-5);

    // 빠른 길.
    let fast = Bert::load(&directory).unwrap().fast().unwrap();
    let quick = fast
        .embed_many(&encoded, 0, plan(8, Batching::Ragged, true, true))
        .unwrap();
    assert!(largest_gap(&quick, &one) < 1e-4);

    // 임베딩은 길이가 1이다.
    for vector in one.as_chunks::<384>().0 {
        let length: f32 = vector.iter().map(|value| value * value).sum();
        assert!((length - 1.0).abs() < 1e-5);
    }

    // 검색. "프랑스의 수도는?"(12번)에 가장 가까운 글은 자신 다음으로
    // "파리는 프랑스의 수도다"(13번)다.
    let mut index = Index::new(384);
    index.extend(&one);
    let hits = index.search(&one[12 * 384..][..384], 2);
    assert_eq!((hits[0].index, hits[1].index), (12, 13));
    assert!((hits[0].score - 1.0).abs() < 1e-5);

    // 너무 긴 글과 단어 사전 밖의 번호는 거절한다.
    let long = vec![0; 513];
    assert!(model.forward(&[&long], Batching::Ragged, 0).is_err());
    assert!(model.forward(&[&[40_000]], Batching::Ragged, 0).is_err());
    // 빈 묶음.
    assert!(
        model
            .embed_many(&[], 0, plan(4, Batching::Padded, true, true))
            .unwrap()
            .is_empty()
    );
}
