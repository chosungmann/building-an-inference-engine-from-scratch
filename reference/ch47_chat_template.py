"""47장: 모델의 대화 틀이 내는 글을 적는다.

실행:
    sh scripts/download_smollm2.sh
    python reference/ch47_chat_template.py

transformers의 `apply_chat_template`이 모델과 함께 온 틀(Jinja)을 실행해
내는 글이다. badak의 `ChatTemplate`이 같은 글을 내는지 견줄 때 쓴다
(`crates/badak_models/tests/chat_template.rs`). 가중치는 읽지 않는다.
"""

import json
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
CONVERSATIONS = [
    [{"role": "user", "content": "Hi"}],
    [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": " 안녕? "}],
    [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris."},
        {"role": "user", "content": "And of Korea?"},
    ],
]


def main():
    for name in ["smollm2_135m", "smollm2_360m", "smollm2_1_7b"]:
        directory = ROOT / "models" / name
        if not (directory / "tokenizer_config.json").exists():
            print(f"{name}: 없습니다")
            continue
        tokenizer = AutoTokenizer.from_pretrained(directory)
        for messages in CONVERSATIONS:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ids = tokenizer(text, add_special_tokens=False).input_ids
            print(json.dumps({"model": name, "messages": messages, "text": text, "ids": ids}, ensure_ascii=False))


if __name__ == "__main__":
    main()
