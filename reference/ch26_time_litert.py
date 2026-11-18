"""26장: LiteRT가 같은 .tflite 파일로 사진 한 장을 추론하는 시간을 잰다.

실행:
    python reference/ch26_time_litert.py

Python에서 `invoke()`를 부르는 시간을 잰다. 호출 한 번에 Python이 쓰는
시간이 섞여 있으므로, 아무 일도 하지 않는 호출의 시간도 함께 적는다.
"""

import statistics
import time
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "models/mnist"
MODELS = ROOT / "models/mnist_tflite"
SINGLES = 1000
ROUNDS = 31


def read_images():
    raw = (DATA / "t10k-images-idx3-ubyte").read_bytes()
    pixels = np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(-1, 784)
    return pixels.astype(np.float32) / 255.0


def median_per_call(work):
    """`work`를 SINGLES번 부르는 일을 ROUNDS번 재서, 한 번의 중앙값(초)을 낸다."""
    samples = []
    for _ in range(3 + ROUNDS):
        start = time.perf_counter()
        for index in range(SINGLES):
            work(index)
        samples.append((time.perf_counter() - start) / SINGLES)
    return statistics.median(samples[3:])


def main():
    images = read_images()
    empty = median_per_call(lambda index: None)
    print(f"아무 일도 하지 않는 Python 호출  {empty * 1e6:8.3f} µs")
    for name in ("mlp", "residual"):
        for threads in (1, None):
            interpreter = Interpreter(model_path=str(MODELS / f"{name}.tflite"), num_threads=threads)
            interpreter.allocate_tensors()
            source = interpreter.get_input_details()[0]["index"]
            interpreter.set_tensor(source, images[:1])
            only = median_per_call(lambda index: interpreter.invoke())

            def whole(index):
                interpreter.set_tensor(source, images[index : index + 1])
                interpreter.invoke()

            both = median_per_call(whole)
            label = "기본값" if threads is None else f"{threads}개"
            print(
                f"{name:<9} 스레드 {label:<4} invoke만 {only * 1e6:8.2f} µs, "
                f"입력을 넣는 것까지 {both * 1e6:8.2f} µs"
            )


if __name__ == "__main__":
    main()
