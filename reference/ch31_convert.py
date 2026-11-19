"""31장: 30장의 ONNX 모델을 NCNN과 MNN의 형식으로 바꾸고 답을 확인한다.

실행:
    sh scripts/download_ch27.sh
    sh scripts/download_ch30.sh
    pip install -r reference/requirements_ch31.txt
    python reference/ch31_convert.py

출력 (models/vision_mobile/):
    resnet50.ncnn.param, resnet50.ncnn.bin        NCNN (가중치 f32)
    mobilenetv2.ncnn.param, mobilenetv2.ncnn.bin
    resnet50.mnn, mobilenetv2.mnn                 MNN

두 변환기 모두 30장에서 우리가 한 일(배치 정규화 접기, 활성화 함수 붙이기)을
스스로 한다. NCNN의 변환기 pnnx는 기본값으로 가중치를 16비트 실수로
저장한다. 엔진끼리 같은 가중치로 견주려고 그것을 끈다(fp16=0).
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import MNN
import ncnn
import numpy as np
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parents[1]
ONNX = ROOT / "models/vision_onnx"
INPUTS = ROOT / "models/vision_tflite/reference.safetensors"
OUT = ROOT / "models/vision_mobile"
MODELS = ("resnet50", "mobilenetv2")
BIN = Path(sys.executable).parent


def run_ncnn(name, x, fp16):
    net = ncnn.Net()
    net.opt.num_threads = 1
    net.opt.use_fp16_packed = net.opt.use_fp16_storage = net.opt.use_fp16_arithmetic = fp16
    net.load_param(str(OUT / f"{name}.ncnn.param"))
    net.load_model(str(OUT / f"{name}.ncnn.bin"))
    extractor = net.create_extractor()
    extractor.input("in0", ncnn.Mat(x[0]).clone())
    return np.array(extractor.extract("out0")[1])


def run_mnn(name, x, precision):
    interpreter = MNN.Interpreter(str(OUT / f"{name}.mnn"))
    session = interpreter.createSession({"numThread": 1, "precision": precision, "backend": "CPU"})
    source = MNN.Tensor((1, 3, 224, 224), MNN.Halide_Type_Float, x, MNN.Tensor_DimensionType_Caffe)
    interpreter.getSessionInput(session).copyFrom(source)
    interpreter.runSession(session)
    host = MNN.Tensor((1, 1000), MNN.Halide_Type_Float, np.zeros((1, 1000), np.float32), MNN.Tensor_DimensionType_Caffe)
    interpreter.getSessionOutput(session).copyToHostTensor(host)
    return np.array(host.getData()).reshape(-1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in MODELS:
        subprocess.run(
            [BIN / "mnnconvert", "-f", "ONNX", "--modelFile", ONNX / f"{name}.onnx", "--MNNModel", OUT / f"{name}.mnn", "--bizCode", "badak"],
            check=True,
            capture_output=True,
        )
        # pnnx는 입력 파일 옆에 중간 파일을 여럿 만든다. 임시 디렉터리에서
        # 돌리고 필요한 둘만 옮긴다.
        with tempfile.TemporaryDirectory() as scratch:
            shutil.copy(ONNX / f"{name}.onnx", scratch)
            subprocess.run(
                [BIN / "pnnx", f"{name}.onnx", "inputshape=[1,3,224,224]", "fp16=0"],
                check=True,
                capture_output=True,
                cwd=scratch,
            )
            for suffix in ("param", "bin"):
                shutil.copy(Path(scratch) / f"{name}.ncnn.{suffix}", OUT)

    inputs = load_file(INPUTS)
    reference = load_file(ONNX / "reference.safetensors")
    images = sorted(key.removesuffix(".input") for key in inputs if key.endswith(".input"))
    print(f"ncnn {ncnn.__version__}, MNN {MNN.version()}")
    print("ONNX Runtime의 점수와의 가장 큰 차이 (사진 다섯 장)")
    print(f"{'모델':<12} {'NCNN f32':>10} {'NCNN fp16':>10} {'MNN 보통':>10} {'MNN 낮음':>10}")
    for name in MODELS:
        layers = sum(1 for _ in open(OUT / f"{name}.ncnn.param", encoding="utf-8")) - 2
        gaps = np.zeros(4)
        for image in images:
            x, want = inputs[f"{image}.input"], reference[f"{name}.{image}.onnxruntime"]
            outputs = [run_ncnn(name, x, False), run_ncnn(name, x, True), run_mnn(name, x, "normal"), run_mnn(name, x, "low")]
            for index, output in enumerate(outputs):
                assert output.argmax() == want.argmax(), (name, image, index)
                gaps[index] = max(gaps[index], np.abs(output - want).max())
        print(f"{name:<12} " + " ".join(f"{gap:>10.2e}" for gap in gaps) + f"   (NCNN의 층 {layers}개)")


if __name__ == "__main__":
    main()
