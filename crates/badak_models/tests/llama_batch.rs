//! 여러 글을 묶은 순전파가 글 하나씩의 순전파와 같은지 본다 (40장).
//!
//! `models/smollm2_135m`이 있어야 돈다. 없으면 건너뛴다.

use badak_models::llama::cache::KvCache;
use badak_models::llama::model::LlamaModel;
use badak_models::llama::paged::{PAGE_TOKENS, PagePool, PagedCache, Step};
use badak_models::models_dir;
use badak_models::tokenizer::{TokenId, Tokenizer};

fn open() -> Option<(LlamaModel, Tokenizer)> {
    let directory = models_dir().join("smollm2_135m");
    if !directory.join("model.safetensors").exists() {
        eprintln!("건너뜀: {}에 모델이 없습니다", directory.display());
        return None;
    }
    Some((
        LlamaModel::load(&directory).unwrap(),
        Tokenizer::from_tokenizer_json(directory.join("tokenizer.json")).unwrap(),
    ))
}

/// 글 하나를 11장의 캐시로 읽고 `steps`걸음을 더 간다. 걸음마다의 점수.
fn alone(model: &LlamaModel, ids: &[TokenId], steps: usize) -> Vec<Vec<f32>> {
    let mut cache = KvCache::new(&model.config);
    let mut input = ids.to_vec();
    let mut all = Vec::new();
    for _ in 0..steps {
        let logits = model.forward_cached(&input, &mut cache).unwrap();
        let logits = logits.as_slice::<f32>().unwrap().to_vec();
        input = vec![argmax(&logits)];
        all.push(logits);
    }
    all
}

fn argmax(logits: &[f32]) -> TokenId {
    let best = logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1));
    best.unwrap().0 as TokenId
}

#[test]
fn a_batch_gives_each_text_the_bits_it_gets_alone() {
    let Some((model, tokenizer)) = open() else {
        return;
    };
    // 길이가 서로 다른 글 셋. 하나는 페이지 여럿에 걸친다.
    let texts = [
        "The capital of France is",
        "Once upon a time, in a small village by the sea, there lived an old fisherman who had \
         never caught a fish larger than his own hand, and yet every morning he",
        "1 2 3 4 5 6 7",
    ];
    let prompts: Vec<Vec<TokenId>> = texts
        .iter()
        .map(|text| tokenizer.encode(text, false))
        .collect();
    assert!(prompts[1].len() > 2 * PAGE_TOKENS);
    let steps = 6;
    let expected: Vec<Vec<Vec<f32>>> = prompts
        .iter()
        .map(|ids| alone(&model, ids, steps))
        .collect();

    // 셋을 함께 읽고 함께 쓴다. 첫 걸음에는 글마다 프롬프트 전체를, 그
    // 뒤로는 방금 고른 토큰 하나씩을 넣는다.
    let pool = PagePool::new(&model.config, 64);
    let mut caches: Vec<PagedCache> = prompts.iter().map(|_| PagedCache::new(&pool)).collect();
    let mut inputs = prompts.clone();
    for step in 0..steps {
        let mut batch: Vec<Step<'_>> = inputs
            .iter()
            .zip(&mut caches)
            .map(|(ids, cache)| Step { ids, cache })
            .collect();
        let logits = model.forward_batch(&mut batch).unwrap();
        for (index, logits) in logits.iter().enumerate() {
            let logits = logits.as_slice::<f32>().unwrap();
            assert!(
                Some(logits) == expected[index].get(step).map(Vec::as_slice),
                "글 {index}의 {step}째 걸음이 혼자 계산한 것과 다릅니다"
            );
            inputs[index] = vec![argmax(logits)];
        }
    }
}

#[test]
fn reading_in_pieces_and_from_a_fork_changes_nothing() {
    let Some((model, tokenizer)) = open() else {
        return;
    };
    let shared = "You are a careful assistant. Answer in one short sentence, never guess, and say so \
                  plainly when you do not know the answer. ";
    let first = tokenizer.encode(&format!("{shared}What is the capital of France?"), false);
    let second = tokenizer.encode(&format!("{shared}Name a river in Egypt."), false);
    let expected = [alone(&model, &first, 1), alone(&model, &second, 1)];

    let pool = PagePool::new(&model.config, 64);
    // 첫 글을 일곱 토큰씩 끊어 읽는다(청크 단위 프롬프트 처리).
    let mut cache = PagedCache::new(&pool);
    let mut logits = Vec::new();
    for piece in first.chunks(7) {
        let mut batch = [Step {
            ids: piece,
            cache: &mut cache,
        }];
        logits = model.forward_batch(&mut batch).unwrap();
    }
    assert!(logits[0].as_slice::<f32>().unwrap() == expected[0][0].as_slice());

    // 둘째 글은 첫 글의 캐시에서 같은 앞부분을 빌려 시작한다.
    let common = first
        .iter()
        .zip(&second)
        .take_while(|(a, b)| a == b)
        .count();
    // 같은 앞부분이 페이지의 가운데에서 끝나게 한다. 반쯤 찬 페이지를
    // 함께 쓰다가 베끼는 길을 지나가야 한다.
    let common = common - usize::from(common.is_multiple_of(PAGE_TOKENS));
    assert!(common > PAGE_TOKENS);
    let before = pool.used();
    let mut forked = cache.fork(common);
    assert_eq!(pool.used(), before);
    let mut batch = [Step {
        ids: &second[common..],
        cache: &mut forked,
    }];
    let logits = model.forward_batch(&mut batch).unwrap();
    assert!(logits[0].as_slice::<f32>().unwrap() == expected[1][0].as_slice());
    assert!(forked.shared_pages() >= common / PAGE_TOKENS);

    // 빌려준 쪽은 그대로다. 이어서 한 토큰을 더 넣어도 혼자일 때와 같다.
    let next = [argmax(expected[0][0].as_slice())];
    let mut batch = [Step {
        ids: &next,
        cache: &mut cache,
    }];
    let logits = model.forward_batch(&mut batch).unwrap();
    let mut alone_cache = KvCache::new(&model.config);
    model.forward_cached(&first, &mut alone_cache).unwrap();
    let again = model.forward_cached(&next, &mut alone_cache).unwrap();
    assert!(logits[0].as_slice::<f32>().unwrap() == again.as_slice::<f32>().unwrap());
}
