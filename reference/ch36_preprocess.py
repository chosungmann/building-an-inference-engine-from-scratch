"""36장: 우리 전처리를 Pillow, 그리고 transformers의 전처리와 견준다.

실행:
    cargo build --release -p badak_image --example ch36_squash
    python reference/ch36_preprocess.py

사진마다 세 가지를 만든다. 우리 것(`ch36_squash`), Pillow의 란초스 필터로
두 번 줄인 것, transformers의 처리기가 낸 것(정답 파일에 든 것). 값이 다른
자리의 수와 가장 큰 차이를 적는다.
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ["tabby_cat", "pelican"]


def pillow(path):
    image = Image.open(path).convert("RGB")
    width, height = image.size
    aspect = width / height
    if width >= height:
        width, height = 2048, int(2048 / aspect)
        height += height % 2
    else:
        width, height = int(2048 * aspect), 2048
        width += width % 2
    small = image.resize((width, height), Image.LANCZOS).resize((512, 512), Image.LANCZOS)
    return ((np.asarray(small).astype(np.float32) / 255 - 0.5) / 0.5).transpose(2, 0, 1)


def compare(label, a, b):
    gap = np.abs(a - b)
    print(f"  {label:<28} 다른 값 {int((gap > 1e-6).sum()):>6}개 / {a.size}, 가장 큰 차이 {gap.max() * 255 / 2:.0f}단계")


def main():
    reference = load_file(ROOT / "models/smolvlm/reference.safetensors")
    for name in IMAGES:
        path = ROOT / f"models/images/{name}.jpg"
        with tempfile.NamedTemporaryFile(suffix=".bin") as file:
            subprocess.run([ROOT / "target/release/examples/ch36_squash", path, file.name], check=True)
            ours = np.fromfile(file.name, dtype="<f4").reshape(3, 512, 512)
        print(name)
        compare("우리 ↔ Pillow", ours, pillow(path))
        compare("우리 ↔ transformers", ours, reference[f"{name}.pixels"])
        compare("Pillow ↔ transformers", pillow(path), reference[f"{name}.pixels"])


if __name__ == "__main__":
    main()
