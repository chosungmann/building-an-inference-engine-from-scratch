"""29장: LiteRT가 정수 모델로 사진 한 장을 분류하는 시간을 잰다.

실행:
    python reference/ch29_time.py

두 길을 잰다. XNNPACK에 넘기는 기본 경로와, LiteRT 자신의 커널(builtin)이다.
"""

import statistics
import time
from pathlib import Path

from safetensors.numpy import load_file

from ai_edge_litert.interpreter import Interpreter, OpResolverType

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models/vision_tflite"
ROUNDS = 31


def median_ms(work):
    samples = []
    for _ in range(3 + ROUNDS):
        start = time.perf_counter()
        work()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples[3:]) * 1e3


def main():
    x = load_file(MODELS / "reference_int8.safetensors")["cup.input"].view("int8")
    floats = load_file(MODELS / "reference.safetensors")["cup.input"]
    print(f"{'모델':<10} {'경로':<10} {'스레드':>6} {'한 장 (ms)':>10}")
    for label, resolver in (("XNNPACK", OpResolverType.AUTO), ("builtin", OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)):
        for file, data in (("int8", x), ("f32", floats)):
            name = "mobilenetv2_int8.tflite" if file == "int8" else "mobilenetv2.tflite"
            for threads in (1, 4):
                interpreter = Interpreter(
                    model_path=str(MODELS / name), num_threads=threads, experimental_op_resolver_type=resolver
                )
                interpreter.allocate_tensors()
                interpreter.set_tensor(interpreter.get_input_details()[0]["index"], data)
                print(f"{file:<10} {label:<10} {threads:>6} {median_ms(interpreter.invoke):>10.2f}")


if __name__ == "__main__":
    main()
