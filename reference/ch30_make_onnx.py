"""30장: 이미지 분류 모델 둘을 손대지 않은 ONNX 파일로 만들고 정답을 남긴다.

실행:
    sh scripts/download_ch27.sh
    pip install -r reference/requirements_ch30.txt
    python reference/ch30_make_onnx.py

출력 (models/vision_onnx/):
    resnet50.onnx          ResNet-50 (timm/resnet50.tv_in1k, BSD-3-Clause)
    mobilenetv2.onnx       MobileNetV2 (timm/mobilenetv2_100.ra_in1k, Apache-2.0)
    reference.safetensors  시험 사진마다 ONNX Runtime이 낸 점수

27장과 같은 가중치다. 다른 점은 변환기가 아무것도 고치지 않게 했다는
것이다. PyTorch의 ONNX 내보내기는 보통 배치 정규화를 합성곱에 접어 넣고
상수끼리의 계산을 미리 해 둔다. 여기서는 둘 다 끈다. 30장이 그 일을 직접
한다.

입력은 27장의 정답 파일(models/vision_tflite/reference.safetensors)에 든,
전처리를 마친 사진 (1, 3, 224, 224)를 그대로 쓴다.
"""

from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import timm
import torch
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file as load_numpy
from safetensors.numpy import save_file
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "models/vision_tflite/reference.safetensors"
OUT = ROOT / "models/vision_onnx"

# (파일 이름, timm의 모델 이름, Hub 저장소, 고정한 판). 27장과 같다.
MODELS = [
    ("resnet50", "resnet50.tv_in1k", "timm/resnet50.tv_in1k", "78f3ecfdb38e"),
    ("mobilenetv2", "mobilenetv2_100.ra_in1k", "timm/mobilenetv2_100.ra_in1k", "5afa12513b04"),
]
OPSET = 18


def session(path, level):
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = level
    return onnxruntime.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = load_numpy(INPUTS)
    names = sorted(key.removesuffix(".input") for key in source if key.endswith(".input"))
    inputs = [source[f"{name}.input"] for name in names]
    reference = {}

    for file, name, repo, revision in MODELS:
        weights = hf_hub_download(repo, "model.safetensors", revision=revision)
        model = timm.create_model(name, pretrained=False)
        model.load_state_dict(load_file(weights))
        model.eval()
        path = OUT / f"{file}.onnx"
        # dynamo=False: 모델을 한 번 실행하며 지나간 연산을 받아 적는 예전
        # 내보내기. training=PRESERVE와 do_constant_folding=False가 배치
        # 정규화 접기와 상수 접기를 끈다. 모델은 eval() 상태이므로 배치
        # 정규화는 저장된 평균과 분산을 쓴다.
        torch.onnx.export(
            model,
            (torch.from_numpy(inputs[0]),),
            str(path),
            dynamo=False,
            opset_version=OPSET,
            training=torch.onnx.TrainingMode.PRESERVE,
            do_constant_folding=False,
            input_names=["input"],
            output_names=["output"],
        )
        graph = onnx.load(str(path)).graph
        onnx.checker.check_model(str(path))
        counts = Counter(node.op_type for node in graph.node)
        print(f"{file}: {path.stat().st_size:,}바이트, 마디 {len(graph.node)}개")
        print("  " + ", ".join(f"{op} {count}" for op, count in sorted(counts.items())))

        with torch.no_grad():
            expected = np.concatenate([model(torch.from_numpy(x)).numpy() for x in inputs])
        # ONNX Runtime도 그래프를 고치지 않게 하고 실행한다.
        plain = session(path, onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL)
        actual = np.concatenate([plain.run(None, {"input": x})[0] for x in inputs])
        for index, image in enumerate(names):
            reference[f"{file}.{image}.onnxruntime"] = actual[index]
            gap = np.abs(actual[index] - expected[index]).max()
            print(f"  {image:<18} {int(actual[index].argmax()):>4}  PyTorch와의 차이 {gap:.2e}")
    save_file(reference, OUT / "reference.safetensors")


if __name__ == "__main__":
    main()
