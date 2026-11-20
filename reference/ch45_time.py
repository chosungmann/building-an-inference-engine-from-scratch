"""45장: 31장의 엔진들이 가속기로 사진 한 장을 분류하는 시간을 잰다.

실행:
    python reference/ch31_convert.py
    python reference/ch45_time.py

이 맥에서 되는 것만 잰다. ONNX Runtime은 Core ML 실행 제공자, NCNN은
Vulkan, MNN은 Metal이다. LiteRT의 파이썬 꾸러미에는 맥용 GPU 델리게이트가
들어 있지 않아서 재지 못한다. 엔진마다 CPU로 낸 점수와 견준 차이와, 준비
시간과 첫 실행 시간을 함께 적는다.

ONNX Runtime이 그래프를 어떻게 나눴는지는 세션을 만들 때 적는 기록에서
읽는다(표준 오류로 나오는 것을 파일로 받아 낸다).
"""

import os
import re
import statistics
import tempfile
import time
from pathlib import Path

import MNN
import ncnn
import numpy as np
import onnxruntime
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
ROUNDS = 50
REST = float(os.environ.get("REST", "3"))


def captured(work):
    """`work`가 도는 동안 표준 오류에 적힌 글을 함께 돌려준다."""
    with tempfile.TemporaryFile() as file:
        saved = os.dup(2)
        os.dup2(file.fileno(), 2)
        try:
            result = work()
        finally:
            os.dup2(saved, 2)
            os.close(saved)
        file.seek(0)
        return result, file.read().decode("utf-8", "replace")


def measure(build):
    """(준비, 첫 실행, 그 뒤의 중앙값, 점수)를 밀리초로 돌려준다."""
    time.sleep(REST)
    started = time.perf_counter()
    work, note = build()
    prepared = time.perf_counter() - started
    started = time.perf_counter()
    scores = work()
    first = time.perf_counter() - started
    walls = []
    for index in range(5 + ROUNDS):
        started = time.perf_counter()
        scores = work()
        if index >= 5:
            walls.append(time.perf_counter() - started)
    return prepared * 1e3, first * 1e3, statistics.median(walls) * 1e3, np.asarray(scores).reshape(-1), note


def ort(name, x, units):
    def build():
        options = onnxruntime.SessionOptions()
        providers = ["CPUExecutionProvider"]
        if units:
            # 나눈 결과를 적게 한다(1은 "알림" 수준).
            options.log_severity_level = 1
            providers.insert(0, ("CoreMLExecutionProvider", {"MLComputeUnits": units, "ModelFormat": "NeuralNetwork"}))
        path = str(MODELS / f"vision_onnx/{name}.onnx")
        session, log = captured(lambda: onnxruntime.InferenceSession(path, options, providers=providers))
        found = re.search(r"number of partitions supported by CoreML: (\d+) number of nodes in the graph: (\d+) number of nodes supported by CoreML: (\d+)", log)
        note = f"조각 {found[1]} 마디 {found[3]}/{found[2]}" if found else ""
        return (lambda: session.run(None, {"input": x})[0]), note

    return build


def ncnn_net(name, x, vulkan):
    def build():
        net = ncnn.Net()
        net.opt.use_vulkan_compute = vulkan
        net.load_param(str(MODELS / f"vision_mobile/{name}.ncnn.param"))
        net.load_model(str(MODELS / f"vision_mobile/{name}.ncnn.bin"))
        mat = ncnn.Mat(x[0]).clone()

        def work():
            extractor = net.create_extractor()
            extractor.input("in0", mat)
            return np.array(extractor.extract("out0")[1])

        work.keep = net
        return work, ""

    return build


def mnn(name, x, backend):
    def build():
        interpreter = MNN.Interpreter(str(MODELS / f"vision_mobile/{name}.mnn"))
        session = interpreter.createSession({"backend": backend, "precision": "normal"})
        source = MNN.Tensor((1, 3, 224, 224), MNN.Halide_Type_Float, x, MNN.Tensor_DimensionType_Caffe)
        target = interpreter.getSessionInput(session)
        output = interpreter.getSessionOutput(session)
        host = MNN.Tensor((1, 1000), MNN.Halide_Type_Float, np.zeros((1, 1000), np.float32), MNN.Tensor_DimensionType_Caffe)

        def work():
            target.copyFrom(source)
            interpreter.runSession(session)
            output.copyToHostTensor(host)
            return np.array(host.getData(), dtype=np.float32)

        work.keep = (interpreter, session)
        return work, ""

    return build


# (엔진, 설정, 만드는 함수). 엔진마다 첫 줄이 CPU이고 나머지를 그것과 견준다.
ENGINES = [
    ("ONNX Runtime", "CPU", lambda n, x: ort(n, x, None)),
    ("ONNX Runtime", "Core ML: CPU만", lambda n, x: ort(n, x, "CPUOnly")),
    ("ONNX Runtime", "Core ML: CPU와 GPU", lambda n, x: ort(n, x, "CPUAndGPU")),
    ("ONNX Runtime", "Core ML: CPU와 뉴럴 엔진", lambda n, x: ort(n, x, "CPUAndNeuralEngine")),
    ("NCNN", "CPU", lambda n, x: ncnn_net(n, x, False)),
    ("NCNN", "Vulkan", lambda n, x: ncnn_net(n, x, True)),
    ("MNN", "CPU", lambda n, x: mnn(n, x, "CPU")),
    ("MNN", "Metal", lambda n, x: mnn(n, x, "METAL")),
]


def main():
    x = load_file(MODELS / "vision_tflite/reference.safetensors")["cup.input"]
    print(f"# ONNX Runtime {onnxruntime.__version__}, ncnn {ncnn.__version__}, MNN {MNN.version()}")
    for name in ("mobilenetv2", "resnet50"):
        baseline = {}
        for engine, setting, make in ENGINES:
            prepared, first, median, scores, note = measure(make(name, x))
            baseline.setdefault(engine, scores)
            gap = float(np.abs(scores - baseline[engine]).max())
            print(
                f"engine {name} {engine.replace(' ', '_')} {setting.replace(' ', '_')} 준비 {prepared:.1f} 첫실행 {first:.2f} "
                f"가운데 {median:.3f} 차이 {gap:.4f} 답 {int(scores.argmax())} {note}",
                flush=True,
            )


if __name__ == "__main__":
    main()
