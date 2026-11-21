"""27장: 이미지 분류 모델 둘을 .tflite 파일로 만들고 정답을 남긴다.

실행:
    sh scripts/download_ch27_images.sh
    pip install -r reference/requirements_ch27.txt
    python reference/ch27_make_tflite.py

출력 (models/vision_tflite/):
    resnet50.tflite        ResNet-50 (timm/resnet50.tv_in1k, BSD-3-Clause)
    mobilenetv2.tflite     MobileNetV2 (timm/mobilenetv2_100.ra_in1k, Apache-2.0)
    labels.txt             ImageNet의 1,000가지 이름. 한 줄에 하나
    reference.safetensors  시험 사진마다의 입력과, PyTorch와 LiteRT의 점수.
                           전처리의 중간 결과도 담는다. 파일을 푼 픽셀의 한
                           귀퉁이(decoded)와 줄이고 자른 사진(shaped)이다

가중치는 Hugging Face Hub의 원래 저장소에서 버전을 고정해 받는다. 두 모델
모두 입력은 정규화한 사진 (1, 3, 224, 224)이고 출력은 점수 (1, 1000)이다.
전처리는 그래프에 넣지 않았다. 27장이 그것을 직접 만든다.
"""

from pathlib import Path

import numpy as np
import timm
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.numpy import save_file
from safetensors.torch import load_file

import litert_torch
from ai_edge_litert.interpreter import Interpreter

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "models/images"
OUT = ROOT / "models/vision_tflite"

# (파일 이름, timm의 모델 이름, Hub 저장소, 고정한 버전)
MODELS = [
    ("resnet50", "resnet50.tv_in1k", "timm/resnet50.tv_in1k", "78f3ecfdb38e"),
    ("mobilenetv2", "mobilenetv2_100.ra_in1k", "timm/mobilenetv2_100.ra_in1k", "5afa12513b04"),
]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
RESIZE, CROP = 256, 224
# 디코더끼리 견주려고 남겨 두는, 푼 사진의 왼쪽 위 귀퉁이의 한 변.
CORNER = 64


def preprocess(path):
    """사진을 모델의 입력으로 바꾼다. 27장의 Rust 코드가 같은 일을 한다.

    짧은 변이 256이 되도록 줄이고(쌍선형, Pillow의 방법), 가운데 224 × 224를
    잘라, 0과 1 사이로 바꾸고, 채널마다 평균을 빼고 표준편차로 나눈다.
    """
    image = Image.open(path).convert("RGB")
    decoded = np.asarray(image)[:CORNER, :CORNER].copy()
    width, height = image.size
    scale = RESIZE / min(width, height)
    size = (max(RESIZE, round(width * scale)), max(RESIZE, round(height * scale)))
    image = image.resize(size, Image.Resampling.BILINEAR)
    left, top = (size[0] - CROP) // 2, (size[1] - CROP) // 2
    image = image.crop((left, top, left + CROP, top + CROP))
    shaped = np.asarray(image).copy()
    pixels = shaped.astype(np.float32) / 255.0
    return decoded, shaped, ((pixels - MEAN) / STD).transpose(2, 0, 1)[None]


def synthetic(width, height):
    """식으로 만든 사진. Rust 쪽 시험이 같은 식으로 같은 사진을 만든다."""
    row, column, channel = np.meshgrid(np.arange(height), np.arange(width), np.arange(3), indexing="ij")
    values = (row * 7 + column * 13 + channel * 31 + (row * column) % 251) % 256
    return Image.fromarray(values.astype(np.uint8), "RGB")


def run_litert(path, inputs):
    interpreter = Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    source = interpreter.get_input_details()[0]["index"]
    target = interpreter.get_output_details()[0]["index"]
    outputs = []
    for x in inputs:
        interpreter.set_tensor(source, x)
        interpreter.invoke()
        outputs.append(interpreter.get_tensor(target)[0].copy())
    return np.stack(outputs)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names = sorted(path.stem for path in IMAGES.glob("*.jpg"))
    reference, inputs = {}, []
    for name in names:
        decoded, shaped, x = preprocess(IMAGES / f"{name}.jpg")
        reference[f"{name}.decoded"] = decoded
        reference[f"{name}.shaped"] = shaped
        reference[f"{name}.input"] = x
        inputs.append(x)

    # 파일을 푸는 일을 빼고 크기를 바꾸는 일만 견주기 위한 것.
    for width, height, new_width, new_height in [(517, 389, 341, 256), (300, 1000, 256, 853), (97, 61, 256, 161)]:
        resized = synthetic(width, height).resize((new_width, new_height), Image.Resampling.BILINEAR)
        reference[f"synthetic.{width}x{height}.{new_width}x{new_height}"] = np.asarray(resized).copy()

    info = timm.data.ImageNetInfo()
    labels = [info.index_to_description(index) for index in range(1000)]
    (OUT / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")

    for file, name, repo, revision in MODELS:
        weights = hf_hub_download(repo, "model.safetensors", revision=revision)
        model = timm.create_model(name, pretrained=False)
        model.load_state_dict(load_file(weights))
        model.eval()
        path = OUT / f"{file}.tflite"
        litert_torch.convert(model, (torch.from_numpy(inputs[0]),)).export(str(path))

        with torch.no_grad():
            expected = np.concatenate([model(torch.from_numpy(x)).numpy() for x in inputs])
        actual = run_litert(path, inputs)
        print(f"{file}: {path.stat().st_size:,}바이트")
        for index, image in enumerate(names):
            reference[f"{file}.{image}.pytorch"] = expected[index]
            reference[f"{file}.{image}.litert"] = actual[index]
            best = int(actual[index].argmax())
            gap = np.abs(actual[index] - expected[index]).max()
            print(f"  {image:<18} {labels[best]:<32} PyTorch와의 차이 {gap:.2e}")
    save_file(reference, OUT / "reference.safetensors")


if __name__ == "__main__":
    main()
