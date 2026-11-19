"""32장: YOLOX-Tiny가 어떤 입력을 기대하는지 넣어 보고 안다.

실행:
    python reference/ch32_preprocess.py

색의 순서(BGR, RGB)와 값의 범위(0~255 그대로, 평균과 표준편차로 정규화)를
바꿔 가며 같은 사진을 넣고, 가장 높은 점수를 본다. YOLOX 프로젝트의
전처리 코드가 판마다 달라서, 받은 ONNX 파일이 어느 쪽인지 문서만으로는
정할 수 없었다.
"""

from pathlib import Path

import numpy as np
import onnxruntime
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models/vision_transformer"
MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)


def main():
    reference = load_file(MODELS / "reference.safetensors")
    session = onnxruntime.InferenceSession(str(MODELS / "yolox_tiny.onnx"), providers=["CPUExecutionProvider"])
    print(f"{'사진':<18} {'BGR 0~255':>12} {'RGB 0~255':>12} {'RGB 정규화':>12} {'BGR 정규화':>12}")
    for name in ("cup", "golden_retriever", "pelican", "tabby_cat"):
        bgr = reference[f"yolox.{name}.input"]
        rgb = bgr[:, ::-1].copy()
        variants = [
            bgr,
            rgb,
            ((rgb / 255.0 - MEAN) / STD).astype(np.float32),
            ((bgr / 255.0 - MEAN[:, ::-1]) / STD[:, ::-1]).astype(np.float32),
        ]
        best = []
        for x in variants:
            output = session.run(None, {"images": x})[0][0]
            best.append((output[:, 4:5] * output[:, 5:]).max())
        print(f"{name:<18} " + " ".join(f"{score:>12.3f}" for score in best))


if __name__ == "__main__":
    main()
