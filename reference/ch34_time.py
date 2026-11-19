"""34장: ONNX Runtime과 PyTorch가 all-MiniLM-L6-v2를 실행하는 시간.

실행:
    python reference/ch34_time.py [latency|bulk]

`latency`는 길이가 다른 글 하나씩의 시간을, `bulk`는 STS 벤치마크의 앞
4,096글을 임베딩으로 바꾸는 처리량을 잰다. 토큰으로 바꾸는 시간은 넣지
않는다. 평균을 내고 길이를 맞추는 일은 넣는다. ONNX Runtime은 모델
저장소에 함께 올라 있는 `onnx/model.onnx`를 쓴다. 재기 전마다 REST초
쉰다(22.12절).
"""

import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models/minilm"
REPO, REVISION = "sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
LIMIT, BULK = 256, 4096
REST = float(os.environ.get("REST", "3"))


def pool(hidden, mask):
    mask = mask[..., None].astype(np.float32)
    mean = (hidden * mask).sum(1) / mask.sum(1)
    return mean / np.linalg.norm(mean, axis=1, keepdims=True)


def engines(threads):
    """(이름, 토큰 묶음을 받아 임베딩을 내는 함수)들."""
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = threads
    session = onnxruntime.InferenceSession(str(MODELS / "model.onnx"), options, providers=["CPUExecutionProvider"])

    def run_onnx(batch):
        hidden = session.run(None, dict(batch))[0]
        return pool(hidden, batch["attention_mask"])

    torch.set_num_threads(threads)
    model = AutoModel.from_pretrained(REPO, revision=REVISION).eval()

    def run_torch(batch):
        with torch.inference_mode():
            inputs = {name: torch.from_numpy(value) for name, value in batch.items()}
            hidden = model(**inputs).last_hidden_state.numpy()
        return pool(hidden, batch["attention_mask"])

    return [("ONNX Runtime", run_onnx), ("PyTorch", run_torch)]


def latency(tokenizer):
    long = (ROOT / "reference/ch34_sentences.txt").read_text(encoding="utf-8").splitlines()[-1]
    for threads in (1, 4, 12):
        for name, run in engines(threads):
            for length in (8, 16, 32, 64, 128, 256):
                batch = dict(tokenizer([long], truncation=True, max_length=length, return_tensors="np"))
                time.sleep(REST)
                samples = []
                for index in range(5 + 101):
                    start = time.perf_counter()
                    run(batch)
                    if index >= 5:
                        samples.append(time.perf_counter() - start)
                median = statistics.median(samples) * 1e3
                print(f"latency {name:<12} 스레드 {threads:>2} 토큰 {length:>3} 중앙값 {median:>7.3f}ms", flush=True)


def bulk(tokenizer):
    rows = (MODELS / "stsbenchmark.tsv").read_text(encoding="utf-8").splitlines()[1:]
    texts = [text for row in rows for text in row.split("\t")[6:8]][:BULK]
    lengths = [len(ids) for ids in tokenizer(texts, truncation=True, max_length=LIMIT)["input_ids"]]
    by_length = sorted(range(len(texts)), key=lambda index: -lengths[index])
    for threads in (1, 12):
        for name, run in engines(threads):
            for size, order_name in ((1, "given"), (32, "given"), (32, "sorted")):
                order = by_length if order_name == "sorted" else list(range(len(texts)))
                batches = [
                    dict(
                        tokenizer(
                            [texts[index] for index in order[start : start + size]],
                            padding=True,
                            truncation=True,
                            max_length=LIMIT,
                            return_tensors="np",
                        )
                    )
                    for start in range(0, len(order), size)
                ]
                rows_used = sum(batch["input_ids"].size for batch in batches)
                run(batches[0])
                best = float("inf")
                for _ in range(3):
                    time.sleep(REST)
                    start = time.perf_counter()
                    for batch in batches:
                        run(batch)
                    best = min(best, time.perf_counter() - start)
                print(
                    f"bulk {name:<12} 스레드 {threads:>2} 묶음 {size:>3} padded {order_name:<6} "
                    f"줄/토큰 {rows_used / sum(lengths):.2f} 초당 {len(texts) / best:>8.1f}글 ({best:.2f}초)",
                    flush=True,
                )


def main():
    tokenizer = AutoTokenizer.from_pretrained(REPO, revision=REVISION)
    print(f"ONNX Runtime {onnxruntime.__version__}, PyTorch {torch.__version__}")
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("latency", "all"):
        latency(tokenizer)
    if mode in ("bulk", "all"):
        bulk(tokenizer)


if __name__ == "__main__":
    main()
