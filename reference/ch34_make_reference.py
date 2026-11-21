"""34장: all-MiniLM-L6-v2의 정답을 남긴다.

실행:
    sh scripts/download_ch34.sh
    pip install -r reference/requirements_ch34.txt
    python reference/ch34_make_reference.py

출력 (models/minilm/):
    reference.safetensors  reference/ch34_sentences.txt의 글마다 토큰 번호,
                           앞의 몇 글은 블록마다의 출력, 글마다의 임베딩(하나씩
                           계산한 것과 한 묶음으로 채워서 계산한 것), STS 벤치마크
                           시험 쌍들의 코사인 유사도와 스피어먼 상관

모델은 sentence-transformers의 all-MiniLM-L6-v2(Apache-2.0)이고 Hugging
Face의 원래 저장소에서 받는다. 토크나이저와 계산은 transformers
라이브러리가 한다. 평균을 내고 길이를 1로 맞추는 일은 모델 카드에 적힌
코드를 따랐다. 이 스크립트는 가중치를 바꾸지 않는다.
"""

from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models/minilm"
REPO, REVISION = "sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
# sentence_bert_config.json의 max_seq_length.
LIMIT = 256
# 블록마다의 출력까지 남길 글의 수.
TRACED = 4


def pool(hidden, mask):
    """진짜 토큰들의 평균을 내고 길이를 1로 맞춘다."""
    mask = mask.unsqueeze(-1).float()
    mean = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return torch.nn.functional.normalize(mean, p=2, dim=1)


def ranks(values):
    """순위. 같은 값은 순위의 평균을 나눠 갖는다."""
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values))
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and values[order[end + 1]] == values[order[start]]:
            end += 1
        result[order[start : end + 1]] = (start + end) / 2
        start = end + 1
    return result


def spearman(a, b):
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def main():
    tokenizer = AutoTokenizer.from_pretrained(REPO, revision=REVISION)
    model = AutoModel.from_pretrained(REPO, revision=REVISION).eval()
    sentences = (ROOT / "reference/ch34_sentences.txt").read_text(encoding="utf-8").splitlines()
    reference = {}

    # 1. 글을 하나씩. 채울 것이 없다.
    singles = []
    for index, sentence in enumerate(sentences):
        encoded = tokenizer(sentence, truncation=True, max_length=LIMIT, return_tensors="pt")
        reference[f"ids.{index}"] = encoded["input_ids"][0].numpy().astype(np.int32)
        with torch.no_grad():
            output = model(**encoded, output_hidden_states=True)
        if index < TRACED:
            reference[f"embedded.{index}"] = output.hidden_states[0][0].numpy()
            for layer, hidden in enumerate(output.hidden_states[1:]):
                reference[f"layer{layer}.{index}"] = hidden[0].numpy()
        singles.append(pool(output.last_hidden_state, encoded["attention_mask"])[0].numpy())
    reference["pooled"] = np.stack(singles)

    # 2. 한 묶음으로. 가장 긴 글에 맞춰 채우고 마스크를 넘긴다.
    batch = tokenizer(sentences, padding=True, truncation=True, max_length=LIMIT, return_tensors="pt")
    with torch.no_grad():
        output = model(**batch)
    reference["pooled_padded"] = pool(output.last_hidden_state, batch["attention_mask"]).numpy()
    gap = np.abs(reference["pooled"] - reference["pooled_padded"]).max()
    print(f"글 {len(sentences)}개, 하나씩과 한 묶음의 차이 {gap:.2e}")

    # 3. STS 벤치마크의 시험 쌍. 사람이 매긴 점수와 코사인 유사도의 순위 상관.
    rows = [line.split("\t") for line in (OUT / "stsbenchmark.tsv").read_text(encoding="utf-8").splitlines()]
    pairs = [(row[6], row[7], float(row[5])) for row in rows[1:] if row[0] == "test"]
    vectors = []
    for side in (0, 1):
        texts = [pair[side] for pair in pairs]
        parts = []
        for start in range(0, len(texts), 64):
            encoded = tokenizer(
                texts[start : start + 64], padding=True, truncation=True, max_length=LIMIT, return_tensors="pt"
            )
            with torch.no_grad():
                hidden = model(**encoded).last_hidden_state
            parts.append(pool(hidden, encoded["attention_mask"]).numpy())
        vectors.append(np.concatenate(parts))
    cosine = (vectors[0] * vectors[1]).sum(1)
    score = spearman(cosine, np.array([pair[2] for pair in pairs]))
    reference["stsb.cosine"] = cosine.astype(np.float32)
    reference["stsb.spearman"] = np.array([score], dtype=np.float32)
    print(f"STS 벤치마크 시험 쌍 {len(pairs)}개, 스피어먼 상관 {score:.4f}")

    save_file(reference, str(OUT / "reference.safetensors"))
    print(f"{OUT / 'reference.safetensors'} 기록")


if __name__ == "__main__":
    main()
