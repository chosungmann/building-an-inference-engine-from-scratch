"""11장: 프롬프트를 처리하는 시간과 토큰 하나를 만드는 시간을 PyTorch로 잰다.

실행: python reference/ch11_bench_pytorch.py llama32_1b
      python reference/ch11_bench_pytorch.py smollm2_135m

badak의 `ch11_prompt_bench` 예제와 같은 토큰을 넣는다. 가중치를 f32로 펼쳐
CPU에서 계산하고, KV 캐시(`past_key_values`)를 쓴다. 세 번 실행해 가장
빠른 시간을 취한다.
"""

import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SENTENCE = "The quick brown fox jumps over the lazy dog. "
LENGTHS = [16, 64, 256]
NEW_TOKENS = 4


def measure(model, ids, length):
    """(프롬프트 처리 시간, 새 토큰 하나의 시간)을 초로 돌려준다."""
    start = time.perf_counter()
    out = model(ids[:, :length], use_cache=True)
    prompt = time.perf_counter() - start

    past = out.past_key_values
    start = time.perf_counter()
    for step in range(NEW_TOKENS):
        out = model(ids[:, length + step : length + step + 1], past_key_values=past, use_cache=True)
        past = out.past_key_values
    return prompt, (time.perf_counter() - start) / NEW_TOKENS


def main():
    if len(sys.argv) != 2:
        sys.exit("모델 디렉터리의 이름을 주십시오. 예: llama32_1b")
    directory = ROOT / "models" / sys.argv[1]
    tokenizer = AutoTokenizer.from_pretrained(directory)
    model = AutoModelForCausalLM.from_pretrained(directory, dtype=torch.float32)
    model.eval()
    ids = tokenizer(SENTENCE * 40, return_tensors="pt").input_ids

    print(f"모델 {sys.argv[1]}, PyTorch {torch.__version__}")
    print(f"{'프롬프트':>10} {'프롬프트 처리':>14} {'토큰당':>14} {'새 토큰 하나':>14} {'비율':>8}")
    with torch.no_grad():
        for length in LENGTHS:
            prompt, step = min(measure(model, ids, length) for _ in range(3))
            per_token = prompt / length
            print(f"{length:>8}개 {prompt:>11.3f} 초 {per_token:>11.4f} 초 {step:>11.4f} 초 {step / per_token:>7.1f}배")


if __name__ == "__main__":
    main()
