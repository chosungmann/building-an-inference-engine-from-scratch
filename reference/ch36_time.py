"""36장: PyTorch(transformers)가 SmolVLM-256M의 단계마다 쓰는 시간.

실행:
    python reference/ch36_time.py

입력은 정답 파일에 든 고양이 사진이다. 재기 전마다 REST초 쉰다(22.12절).
"""

import os
import statistics
import time
from pathlib import Path

import torch
from safetensors.numpy import load_file
from transformers import AutoModelForImageTextToText

ROOT = Path(__file__).resolve().parents[1]
REPO, REVISION = "HuggingFaceTB/SmolVLM-256M-Instruct", "d039ab0d4b9311b617e8ffec1d06c8464364aeb3"
REST = float(os.environ.get("REST", "3"))


def median_ms(rounds, work):
    time.sleep(REST)
    work()
    samples = []
    for _ in range(rounds):
        start = time.perf_counter()
        work()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples) * 1e3


def main():
    reference = load_file(ROOT / "models/smolvlm/reference.safetensors")
    model = AutoModelForImageTextToText.from_pretrained(REPO, revision=REVISION, dtype=torch.float32).eval()
    pixels = torch.from_numpy(reference["tabby_cat.pixels"])[None]
    ids = torch.from_numpy(reference["tabby_cat.ids"]).long()[None]
    mask = torch.ones_like(ids)
    print(f"PyTorch {torch.__version__}")
    for threads in (1, 4, 12):
        torch.set_num_threads(threads)
        with torch.inference_mode():
            vision = median_ms(11, lambda: model.model.vision_model(pixel_values=pixels).last_hidden_state)
            print(f"vision  PyTorch 스레드 {threads:>2} 중앙값 {vision:>8.2f}ms", flush=True)
            inputs = dict(input_ids=ids, attention_mask=mask, pixel_values=pixels[None])
            # 한 토큰만 쓰게 한 시간과 48토큰을 쓰게 한 시간의 차이에서 토큰마다의 시간을 낸다.
            one = median_ms(5, lambda: model.generate(**inputs, do_sample=False, max_new_tokens=1))
            many = median_ms(5, lambda: model.generate(**inputs, do_sample=False, max_new_tokens=48))
            print(f"first   PyTorch 스레드 {threads:>2} 중앙값 {one:>8.2f}ms (사진 보기와 글 읽기를 합친 것)", flush=True)
            print(f"token   PyTorch 스레드 {threads:>2} 중앙값 {(many - one) / 47:>8.2f}ms", flush=True)


if __name__ == "__main__":
    main()
