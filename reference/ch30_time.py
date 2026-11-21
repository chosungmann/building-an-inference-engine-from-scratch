"""30장: ONNX Runtime이 그래프를 얼마나 고치고, 그래서 얼마나 빨라지는지 잰다.

실행:
    python reference/ch30_time.py

ONNX Runtime은 그래프 최적화를 네 단계로 켜고 끌 수 있다. 단계마다 고친
그래프를 파일로 받아 노드를 세고, 사진 한 장의 시간을 잰다. 입력은 27장의
정답 파일에 든 첫 사진이다.
"""

import statistics
import tempfile
import time
from collections import Counter
from pathlib import Path

import onnx
import onnxruntime
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models/vision_onnx"
INPUTS = ROOT / "models/vision_tflite/reference.safetensors"
ROUNDS = 51
LEVELS = [
    ("끔", onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL),
    ("기본", onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC),
    ("확장", onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED),
    ("모두", onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL),
]


def median_ms(work):
    samples = []
    for _ in range(5 + ROUNDS):
        start = time.perf_counter()
        work()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples[5:]) * 1e3


def main():
    x = load_file(INPUTS)["cup.input"]
    print(f"ONNX Runtime {onnxruntime.__version__}")
    print(f"{'모델':<12} {'최적화':<4} {'노드':>4} {'스레드 1':>9} {'스레드 12':>9}  남은 연산")
    with tempfile.TemporaryDirectory() as scratch:
        for file in ("resnet50", "mobilenetv2"):
            for label, level in LEVELS:
                times = []
                for threads in (1, 12):
                    options = onnxruntime.SessionOptions()
                    options.graph_optimization_level = level
                    options.intra_op_num_threads = threads
                    saved = Path(scratch) / f"{file}_{label}.onnx"
                    options.optimized_model_filepath = str(saved)
                    session = onnxruntime.InferenceSession(
                        str(MODELS / f"{file}.onnx"), options, providers=["CPUExecutionProvider"]
                    )
                    times.append(median_ms(lambda: session.run(None, {"input": x})))
                nodes = onnx.load(str(saved)).graph.node
                counts = Counter(node.op_type for node in nodes)
                kinds = ", ".join(f"{op} {count}" for op, count in sorted(counts.items()))
                print(f"{file:<12} {label:<4} {len(nodes):>4} {times[0]:>9.2f} {times[1]:>9.2f}  {kinds}")


if __name__ == "__main__":
    main()
