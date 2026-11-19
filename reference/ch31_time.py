"""31장: 네 엔진이 사진 한 장을 분류하는 시간을 같은 조건으로 잰다.

실행:
    python reference/ch31_convert.py
    python reference/ch31_time.py [스레드 수 ...]

엔진마다, 설정마다 따로 재고 그 앞에서 REST초 쉰다(22.12절). 벽시계 시간과
함께 프로세스가 쓴 CPU 시간을 적는다. 둘의 비가 "실제로 일한 코어의 수"다.
입력은 27장의 정답 파일에 든 첫 사진이다.
"""

import os
import statistics
import sys
import time
from pathlib import Path

import MNN
import ncnn
import numpy as np
import onnxruntime
from ai_edge_litert.interpreter import Interpreter, OpResolverType
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
ROUNDS = 51
REST = float(os.environ.get("REST", "3"))


def measure(work):
    """(벽시계 시간의 중앙값, CPU 시간의 중앙값)을 밀리초로 돌려준다."""
    time.sleep(REST)
    walls, cpus = [], []
    for index in range(5 + ROUNDS):
        wall, cpu = time.perf_counter(), time.process_time()
        work()
        if index >= 5:
            walls.append(time.perf_counter() - wall)
            cpus.append(time.process_time() - cpu)
    return statistics.median(walls) * 1e3, statistics.median(cpus) * 1e3


def litert(name, threads, x, resolver):
    interpreter = Interpreter(
        model_path=str(MODELS / f"vision_tflite/{name}.tflite"), num_threads=threads, experimental_op_resolver_type=resolver
    )
    interpreter.allocate_tensors()
    index = interpreter.get_input_details()[0]["index"]

    def work():
        interpreter.set_tensor(index, x)
        interpreter.invoke()

    return work


def ort(name, threads, x):
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = threads
    session = onnxruntime.InferenceSession(str(MODELS / f"vision_onnx/{name}.onnx"), options, providers=["CPUExecutionProvider"])
    return lambda: session.run(None, {"input": x})


def ncnn_net(name, threads, x, fp16):
    net = ncnn.Net()
    net.opt.num_threads = threads
    net.opt.use_fp16_packed = net.opt.use_fp16_storage = net.opt.use_fp16_arithmetic = fp16
    net.load_param(str(MODELS / f"vision_mobile/{name}.ncnn.param"))
    net.load_model(str(MODELS / f"vision_mobile/{name}.ncnn.bin"))
    mat = ncnn.Mat(x[0]).clone()

    def work():
        extractor = net.create_extractor()
        extractor.input("in0", mat)
        extractor.extract("out0")

    work.keep = net
    return work


def mnn(name, threads, x, precision):
    interpreter = MNN.Interpreter(str(MODELS / f"vision_mobile/{name}.mnn"))
    session = interpreter.createSession({"numThread": threads, "precision": precision, "backend": "CPU"})
    source = MNN.Tensor((1, 3, 224, 224), MNN.Halide_Type_Float, x, MNN.Tensor_DimensionType_Caffe)
    target = interpreter.getSessionInput(session)
    output = interpreter.getSessionOutput(session)
    host = MNN.Tensor((1, 1000), MNN.Halide_Type_Float, np.zeros((1, 1000), np.float32), MNN.Tensor_DimensionType_Caffe)

    def work():
        target.copyFrom(source)
        interpreter.runSession(session)
        output.copyToHostTensor(host)

    return work


ENGINES = [
    ("LiteRT + XNNPACK", lambda n, t, x: litert(n, t, x, OpResolverType.AUTO)),
    ("LiteRT 자체 커널", lambda n, t, x: litert(n, t, x, OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)),
    ("ONNX Runtime", ort),
    ("NCNN f32", lambda n, t, x: ncnn_net(n, t, x, False)),
    ("NCNN fp16 (기본값)", lambda n, t, x: ncnn_net(n, t, x, True)),
    ("MNN", lambda n, t, x: mnn(n, t, x, "normal")),
    ("MNN 낮은 정밀도", lambda n, t, x: mnn(n, t, x, "low")),
]


def main():
    threads = [int(text) for text in sys.argv[1:]] or [1, 4, 12]
    x = load_file(MODELS / "vision_tflite/reference.safetensors")["cup.input"]
    print(f"ONNX Runtime {onnxruntime.__version__}, ncnn {ncnn.__version__}, MNN {MNN.version()}")
    print(f"{'모델':<12} {'엔진':<20} {'스레드':>4} {'한 장 (ms)':>10} {'CPU (ms)':>9} {'코어':>5}")
    for name in ("resnet50", "mobilenetv2"):
        for count in threads:
            for label, build in ENGINES:
                wall, cpu = measure(build(name, count, x))
                print(f"{name:<12} {label:<20} {count:>4} {wall:>10.2f} {cpu:>9.2f} {cpu / wall:>5.1f}", flush=True)


if __name__ == "__main__":
    main()
