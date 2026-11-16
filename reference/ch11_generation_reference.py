"""11장: PyTorch로 글을 이어 써서, 뽑힌 토큰을 기록한다.

실행:
    python reference/ch11_generation_reference.py llama32_1b
    python reference/ch11_generation_reference.py smollm2_135m

출력: models/<모델 디렉터리>/generation.json

가중치를 f32로 펼쳐 CPU에서 계산하고, 매번 점수가 가장 높은 토큰을 고른다.
글마다 넣은 토큰, 새로 뽑힌 토큰, 그리고 뽑을 때마다의 1위와 2위의 점수
차이를 기록한다. 점수 차이가 작은 자리는 엔진이 달라지면 결과가 갈릴 수
있는 자리다.
"""

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MAX_NEW_TOKENS = 40

# 글과, 특수 토큰의 표기를 특수 토큰으로 읽을지. 대화는 모델마다 형식이
# 다르므로 8.9절의 형식으로 직접 적는다.
PLAIN = "The capital of France is"
CHATS = {
    "llama32_1b": (
        "<|start_header_id|>user<|end_header_id|>\n\n"
        "대한민국의 수도는 어디인가요? 한 문장으로 답해 주세요.<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    ),
    "smollm2_135m": (
        "<|im_start|>user\nWhat is the capital of France? Answer in one sentence.<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
}


def generate(model, ids, eos):
    """캐시 없이, 매번 글 전체를 넣어 다음 토큰을 고른다."""
    new, margins = [], []
    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            logits = model(torch.tensor([ids + new]), use_cache=False).logits[0, -1]
            top = torch.topk(logits, 2)
            token = top.indices[0].item()
            if token in eos:
                break
            new.append(token)
            margins.append(round((top.values[0] - top.values[1]).item(), 4))
    return new, margins


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in CHATS:
        sys.exit(f"모델 디렉터리의 이름을 주십시오: {', '.join(CHATS)}")
    name = sys.argv[1]
    directory = ROOT / "models" / name
    tokenizer = AutoTokenizer.from_pretrained(directory)
    model = AutoModelForCausalLM.from_pretrained(directory, dtype=torch.float32)
    model.eval()
    eos = model.config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])

    cases = []
    for kind, text in (("plain", PLAIN), ("chat", CHATS[name])):
        # 토크나이저의 설정대로 글 앞에 토큰을 붙인다(10.4절).
        ids = tokenizer(text, add_special_tokens=True).input_ids
        new, margins = generate(model, ids, eos)
        cases.append({"kind": kind, "text": text, "ids": ids, "generated": new, "margins": margins})
        print(f"{kind}: 토큰 {len(ids)}개 -> 새 토큰 {len(new)}개, 가장 작은 점수 차이 {min(margins)}")
        print(f"  {tokenizer.decode(new)!r}")

    out = directory / "generation.json"
    out.write_text(json.dumps(cases, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {out.name} ({out.stat().st_size}바이트)")


if __name__ == "__main__":
    main()
