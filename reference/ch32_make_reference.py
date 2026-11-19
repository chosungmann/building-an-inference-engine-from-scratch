"""32장: ViT와 YOLOX의 정답을 남긴다.

실행:
    sh scripts/download_ch27.sh
    sh scripts/download_ch32.sh
    pip install -r reference/requirements_ch32.txt
    python reference/ch32_make_reference.py

출력 (models/vision_transformer/):
    reference.safetensors   ViT-Tiny와 ViT-B/16: 사진마다 입력과 PyTorch의 점수.
                            ViT-Tiny의 첫 사진은 중간 결과도.
                            YOLOX: 사진마다 입력과 ONNX Runtime의 출력.

두 모델의 가중치는 원래 배포처에서 받는다(scripts/download_ch32.sh).
ViT-Tiny는 timm의 저장소(Apache-2.0), YOLOX-Tiny는 YOLOX 프로젝트가 올려 둔
ONNX 파일(Apache-2.0)이다. 이 스크립트는 가중치를 바꾸지 않는다.

ViT의 입력은 27장의 정답 파일에 든 "줄이고 자른 사진"(`*.shaped`,
224 × 224)에서 만든다. 0~1로 바꾸고 0.5를 빼고 0.5로 나눈다. 이 모델을
학습할 때 쓴 정규화다.

YOLOX의 입력은 사진을 비율을 지켜 416 × 416 안에 들어가게 줄이고, 남는
자리를 114로 채운 것이다. 채널은 BGR 순서이고 값은 0~255 그대로다. YOLOX
프로젝트의 `preproc`과 같다. 줄이는 방법만 다르다(Pillow의 쌍선형).
"""

from pathlib import Path

import numpy as np
import onnxruntime
import timm
import torch
from PIL import Image
from safetensors.numpy import load_file as load_numpy
from safetensors.numpy import save_file
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "models/images"
SHAPED = ROOT / "models/vision_tflite/reference.safetensors"
OUT = ROOT / "models/vision_transformer"
SIDE = 416


def vit_trace(model, x):
    """timm의 `forward`를 풀어 쓴 것. 단계마다의 값을 남긴다."""
    trace = {}
    with torch.no_grad():
        tokens = model.patch_embed(x)
        trace["patches"] = tokens
        tokens = model._pos_embed(tokens)
        trace["embedded"] = tokens
        for index, block in enumerate(model.blocks):
            tokens = block(tokens)
            if index in (0, 11):
                trace[f"block{index}"] = tokens
        tokens = model.norm(tokens)
        trace["normed"] = tokens
        trace["logits"] = model.head(tokens[:, 0])
    return {key: value.numpy() for key, value in trace.items()}


def letterbox(path):
    """사진을 비율을 지켜 줄여 왼쪽 위에 놓고, 나머지를 114로 채운다."""
    image = Image.open(path).convert("RGB")
    ratio = min(SIDE / image.width, SIDE / image.height)
    size = (int(image.width * ratio), int(image.height * ratio))
    small = np.asarray(image.resize(size, Image.Resampling.BILINEAR))
    canvas = np.full((SIDE, SIDE, 3), 114, dtype=np.uint8)
    canvas[: size[1], : size[0]] = small
    # RGB를 BGR로, (높이, 너비, 채널)을 (채널, 높이, 너비)로.
    return canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32), ratio


def main():
    shaped = load_numpy(SHAPED)
    names = sorted(key.removesuffix(".shaped") for key in shaped if key.endswith(".shaped") and "synthetic" not in key)
    reference = {}

    info = timm.data.ImageNetInfo()
    # (정답의 이름, 파일, timm의 모델 이름). ViT-B/16은 파일이 있을 때만 한다
    # (sh scripts/download_ch32.sh base).
    models = [
        ("vit", "vit_tiny.safetensors", "vit_tiny_patch16_224.augreg_in21k_ft_in1k"),
        ("vit_base", "vit_base.safetensors", "vit_base_patch16_224.augreg2_in21k_ft_in1k"),
    ]
    for prefix, file, timm_name in models:
        if not (OUT / file).exists():
            print(f"건너뜀: {file}이 없습니다")
            continue
        model = timm.create_model(timm_name, pretrained=False)
        model.load_state_dict(load_file(OUT / file))
        model.eval()
        print(f"{timm_name} (PyTorch)")
        for index, name in enumerate(names):
            pixels = shaped[f"{name}.shaped"].astype(np.float32) / 255.0
            x = ((pixels - 0.5) / 0.5).transpose(2, 0, 1)[None]
            trace = vit_trace(model, torch.from_numpy(x))
            with torch.no_grad():
                direct = model(torch.from_numpy(x)).numpy()
            assert np.array_equal(direct, trace["logits"])
            # 두 모델의 입력은 같다. 한 번만 담는다.
            reference[f"vit.{name}.input"] = x
            reference[f"{prefix}.{name}.logits"] = trace["logits"][0]
            if index == 0 and prefix == "vit":
                for key, value in trace.items():
                    reference[f"{prefix}.{name}.{key}"] = value[0]
            best = int(trace["logits"].argmax())
            print(f"  {name:<18} {info.index_to_description(best)}")

    session = onnxruntime.InferenceSession(str(OUT / "yolox_tiny.onnx"), providers=["CPUExecutionProvider"])
    print("YOLOX-Tiny (ONNX Runtime)")
    for name in names:
        x, ratio = letterbox(IMAGES / f"{name}.jpg")
        output = session.run(None, {"images": x})[0][0]
        reference[f"yolox.{name}.input"] = x
        reference[f"yolox.{name}.output"] = output
        reference[f"yolox.{name}.ratio"] = np.array([ratio], dtype=np.float32)
        # 물체가 있을 점수 × 종류의 점수가 가장 큰 칸.
        scores = output[:, 4:5] * output[:, 5:]
        row, kind = np.unravel_index(scores.argmax(), scores.shape)
        print(f"  {name:<18} 가장 높은 점수 {scores[row, kind]:.3f} (칸 {row}, 종류 {kind})")
    save_file(reference, OUT / "reference.safetensors")


if __name__ == "__main__":
    main()
