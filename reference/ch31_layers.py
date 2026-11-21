"""31장: 층 하나짜리 모델로 엔진들을 견준다.

실행:
    python reference/ch31_layers.py [스레드 수 ...]

모델 전체의 시간은 여러 종류의 층이 섞인 값이다. 층을 하나씩 떼어 재면
엔진마다 어떤 층에서 빠르고 느린지가 보인다. 층의 모양은 ResNet-50과
MobileNetV2에서 골랐다. 가중치는 무작위다(시간은 값과 무관하다).

층마다 PyTorch 모듈을 만들어 ONNX, .tflite, NCNN, MNN으로 바꾼 뒤 잰다.
바꾼 파일은 임시 디렉터리에 두고 버린다. 우리 엔진의 같은 층은
`cargo run --release -p badak_graph --example ch31_layers`가 잰다.
"""

import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import MNN
import ncnn
import numpy as np
import onnxruntime
import torch
from ai_edge_litert.interpreter import Interpreter

import litert_torch

ROUNDS = 51
REST = float(os.environ.get("REST", "1"))
BIN = Path(sys.executable).parent

# (이름, 채널, 한 변, 모듈). 이름은 Rust 쪽의 표와 같다.
LAYERS = [
    ("conv 7×7 스트라이드 2, 3→64 @224", 3, 224, lambda: torch.nn.Conv2d(3, 64, 7, 2, 3)),
    ("conv 3×3, 64→64 @56", 64, 56, lambda: torch.nn.Conv2d(64, 64, 3, 1, 1)),
    ("conv 1×1, 64→256 @56", 64, 56, lambda: torch.nn.Conv2d(64, 256, 1)),
    ("conv 3×3, 256→256 @14", 256, 14, lambda: torch.nn.Conv2d(256, 256, 3, 1, 1)),
    ("conv 1×1, 1024→256 @14", 1024, 14, lambda: torch.nn.Conv2d(1024, 256, 1)),
    ("conv 3×3, 512→512 @7", 512, 7, lambda: torch.nn.Conv2d(512, 512, 3, 1, 1)),
    ("conv 1×1, 16→96 @112", 16, 112, lambda: torch.nn.Conv2d(16, 96, 1)),
    ("depthwise 3×3, 144 @56", 144, 56, lambda: torch.nn.Conv2d(144, 144, 3, 1, 1, groups=144)),
    ("depthwise 3×3, 960 @7", 960, 7, lambda: torch.nn.Conv2d(960, 960, 3, 1, 1, groups=960)),
]


def measure(work):
    time.sleep(REST)
    samples = []
    for index in range(5 + ROUNDS):
        start = time.perf_counter()
        work()
        if index >= 5:
            samples.append(time.perf_counter() - start)
    return statistics.median(samples) * 1e6


def engines(module, x, threads, scratch):
    """이 층을 엔진마다의 형식으로 바꾸고, (이름, 실행하는 함수)들을 돌려준다."""
    scratch = Path(scratch)
    torch.onnx.export(module, (torch.from_numpy(x),), str(scratch / "layer.onnx"), dynamo=False, opset_version=18,
                      input_names=["input"], output_names=["output"])
    litert_torch.convert(module, (torch.from_numpy(x),)).export(str(scratch / "layer.tflite"))
    subprocess.run([BIN / "mnnconvert", "-f", "ONNX", "--modelFile", "layer.onnx", "--MNNModel", "layer.mnn", "--bizCode", "badak"],
                   check=True, capture_output=True, cwd=scratch)
    shape = ",".join(str(dim) for dim in x.shape)
    subprocess.run([BIN / "pnnx", "layer.onnx", f"inputshape=[{shape}]", "fp16=0"], check=True, capture_output=True, cwd=scratch)

    interpreter = Interpreter(model_path=str(scratch / "layer.tflite"), num_threads=threads)
    interpreter.allocate_tensors()
    index = interpreter.get_input_details()[0]["index"]

    def litert():
        interpreter.set_tensor(index, x)
        interpreter.invoke()

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = threads
    session = onnxruntime.InferenceSession(str(scratch / "layer.onnx"), options, providers=["CPUExecutionProvider"])

    nets = []
    for fp16 in (False, True):
        net = ncnn.Net()
        net.opt.num_threads = threads
        net.opt.use_fp16_packed = net.opt.use_fp16_storage = net.opt.use_fp16_arithmetic = fp16
        net.load_param(str(scratch / "layer.ncnn.param"))
        net.load_model(str(scratch / "layer.ncnn.bin"))
        nets.append(net)
    mat = ncnn.Mat(x[0]).clone()

    def run_ncnn(net):
        extractor = net.create_extractor()
        extractor.input("in0", mat)
        extractor.extract("out0")

    mnn = MNN.Interpreter(str(scratch / "layer.mnn"))
    session_mnn = mnn.createSession({"numThread": threads, "precision": "normal", "backend": "CPU"})
    source = MNN.Tensor(x.shape, MNN.Halide_Type_Float, x, MNN.Tensor_DimensionType_Caffe)
    target = mnn.getSessionInput(session_mnn)

    def run_mnn():
        target.copyFrom(source)
        mnn.runSession(session_mnn)

    return [
        ("XNNPACK", litert),
        ("ORT", lambda: session.run(None, {"input": x})),
        ("NCNN", lambda: run_ncnn(nets[0])),
        ("NCNN16", lambda: run_ncnn(nets[1])),
        ("MNN", run_mnn),
    ]


def main():
    counts = [int(text) for text in sys.argv[1:]] or [1, 12]
    torch.manual_seed(0)
    print("층 하나의 시간 (마이크로초). NCNN16은 16비트로 계산하는 기본 설정이다.")
    for threads in counts:
        print(f"\n스레드 {threads}개")
        print(f"{'층':<28} {'XNNPACK':>9} {'ORT':>9} {'NCNN':>9} {'NCNN16':>9} {'MNN':>9}")
        for name, channels, side, build in LAYERS:
            module = build().eval()
            x = torch.randn(1, channels, side, side).numpy()
            with tempfile.TemporaryDirectory() as scratch:
                times = [measure(work) for _, work in engines(module, x, threads, scratch)]
            print(f"{name:<28} " + " ".join(f"{value:>9.0f}" for value in times), flush=True)


if __name__ == "__main__":
    main()
