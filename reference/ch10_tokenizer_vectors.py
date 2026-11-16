"""10장: 모델을 만든 쪽의 토크나이저로 시험 자료를 만든다.

실행:
    sh scripts/download_ch08.sh
    python reference/ch10_tokenizer_vectors.py llama32_1b
    python reference/ch10_tokenizer_vectors.py smollm2_135m

출력: models/<모델 디렉터리>/tokenizer_vectors.json

Hugging Face의 tokenizers 라이브러리로 `tokenizer.json`을 읽어, 여러 글의
토큰 번호를 기록한다. 입력은 llama.cpp의 토크나이저 시험 입력에 숫자와
한글이 든 글을 더한 것이다.
"""

import json
import sys
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
SEPARATOR = "\n__ggml_vocab_test__\n"
EXTRA = [
    "The capital of France is",
    "The capital of France is Paris. 추론 엔진",
    "밑바닥부터 만드는 추론 엔진: Rust로 배우는 AI 인퍼런스 엔지니어링",
    "x = 3.14159 + 42 * 1000000; // 2026-11-17",
    "v1.2.3  7 8  9\n10\t11",
    "fn main() {\n    println!(\"Hello, {}!\", 2 + 2);\n}\n",
    "½ ٣ Ⅳ ① 2²",
    "I'LL don't WE'RE y'all",
    "<|eot_id|><|im_start|>user<|endoftext|>",
    # 단어 사전에 토큰이 없을 수 있는 제어 문자.
    "a\x04b\x13c",
]


def main():
    if len(sys.argv) != 2:
        sys.exit("모델 디렉터리의 이름을 주십시오. 예: llama32_1b")
    model = ROOT / "models" / sys.argv[1]
    tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))
    inputs = (ROOT / "models/llama_cpp_vocab/ggml-vocab-llama-bpe.gguf.inp").read_text(encoding="utf-8")
    texts = inputs.split(SEPARATOR) + EXTRA
    # 글의 시작을 알리는 토큰은 붙이지 않는다. 특수 토큰의 표기는 특수 토큰으로 읽힌다.
    vectors = [
        {
            "text": text,
            "ids": tokenizer.encode(text, add_special_tokens=False).ids,
        }
        for text in texts
    ]
    out = model / "tokenizer_vectors.json"
    out.write_text(json.dumps(vectors, ensure_ascii=False, indent=0), encoding="utf-8")
    total = sum(len(vector["ids"]) for vector in vectors)
    print(f"{sys.argv[1]}: 글 {len(vectors)}개, 토큰 {total}개 -> {out.name} ({out.stat().st_size}바이트)")


if __name__ == "__main__":
    main()
