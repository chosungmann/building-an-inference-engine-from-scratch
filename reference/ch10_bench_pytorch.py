"""10장: 같은 모델의 순전파를 PyTorch로 실행하는 시간을 잰다.

실행: python reference/ch10_bench_pytorch.py llama32_1b
      python reference/ch10_bench_pytorch.py smollm2_135m

badak의 `ch10_next_token` 예제와 같은 토큰을 넣고, 가중치를 f32로 펼쳐
CPU에서 계산한다. 세 번 실행해 가장 빠른 시간을 취한다.
"""

import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
PROMPT = "The capital of France is"


def best_of(repeats, f):
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        f()
        best = min(best, time.perf_counter() - start)
    return best


def main():
    if len(sys.argv) != 2:
        sys.exit("모델 디렉터리의 이름을 주십시오. 예: llama32_1b")
    directory = ROOT / "models" / sys.argv[1]
    tokenizer = AutoTokenizer.from_pretrained(directory)
    model = AutoModelForCausalLM.from_pretrained(directory, dtype=torch.float32)
    model.eval()
    ids = tokenizer(PROMPT, return_tensors="pt").input_ids
    print(f"모델 {sys.argv[1]}, 토큰 {ids.shape[1]}개: {ids[0].tolist()}")

    default_threads = torch.get_num_threads()
    for threads in (1, default_threads):
        torch.set_num_threads(threads)
        with torch.no_grad():
            # use_cache=False: 뒤에 쓸 값을 따로 모아 두지 않는다 (11장).
            seconds = best_of(3, lambda: model(ids, use_cache=False))
            logits = model(ids, use_cache=False).logits[0, -1]
        token = tokenizer.decode(logits.argmax().item())
        print(f"PyTorch {torch.__version__}, 스레드 {threads}개: 순전파 {seconds * 1e3:8.1f} ms, 다음 토큰 {token!r}")


if __name__ == "__main__":
    main()
