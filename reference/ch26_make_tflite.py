"""26장: 손글씨 숫자 분류 모델 둘을 .tflite 파일로 만들고 LiteRT의 정답을 남긴다.

실행:
    sh scripts/download_mnist.sh
    sh scripts/download_ch07.sh
    pip install -r reference/requirements_ch26.txt
    python reference/ch26_make_tflite.py

출력 (models/mnist_tflite/):
    mlp.tflite             7장의 모델(가중치도 같다)을 그래프로 옮긴 것
    residual.tflite        잔차 연결이 있는 조금 더 깊은 모델. 여기서 학습한다
    reference.safetensors  LiteRT가 두 모델로 계산한 점수와 예측

두 모델 모두 입력은 0과 1 사이의 픽셀 값 (1, 784)이고, 정규화가 그래프
안에 들어 있다. 출력은 숫자 열 개의 확률 (1, 10)이다.
"""

from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file, save_file
from torch import nn

import litert_torch
from ai_edge_litert.interpreter import Interpreter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "models/mnist"
OUT = ROOT / "models/mnist_tflite"

MEAN, STD = 0.1307, 0.3081
SEED = 0
EPOCHS = 4
BATCH = 64
# 점수를 저장해 둘 시험 이미지의 수.
TRACED = 16


def read_idx(path):
    """IDX 파일을 NumPy 배열로 읽는다. 머리의 숫자는 빅 엔디언이다."""
    raw = path.read_bytes()
    ndim = raw[3]
    shape = [int.from_bytes(raw[4 + 4 * i : 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


class Mlp(nn.Module):
    """7장의 모델. 정규화를 그래프 안에 넣었다."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, pixels):
        x = (pixels - MEAN) / STD
        return torch.softmax(self.fc2(torch.relu(self.fc1(x))), dim=-1)


class Residual(nn.Module):
    """잔차 연결이 있는 모델. 블록 하나가 h + fc2(relu(fc1(h)))이다."""

    def __init__(self, width=256, inner=512, blocks=3):
        super().__init__()
        self.stem = nn.Linear(784, width)
        self.down = nn.ModuleList(nn.Linear(width, inner) for _ in range(blocks))
        self.up = nn.ModuleList(nn.Linear(inner, width) for _ in range(blocks))
        self.head = nn.Linear(width, 10)

    def logits(self, pixels):
        h = torch.relu(self.stem((pixels - MEAN) / STD))
        for down, up in zip(self.down, self.up):
            h = torch.relu(h + up(torch.relu(down(h))))
        return self.head(h)

    def forward(self, pixels):
        return torch.softmax(self.logits(pixels), dim=-1)


def train(model, images, labels):
    """잔차 모델을 학습한다. 7장과 같은 방법이다."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(SEED)
    for epoch in range(EPOCHS):
        order = torch.randperm(len(images), generator=generator)
        for start in range(0, len(order), BATCH):
            index = order[start : start + BATCH]
            loss = nn.functional.cross_entropy(model.logits(images[index]), labels[index])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"  {epoch + 1}번째 훑기, 마지막 손실 {loss.item():.4f}")


def run_litert(path, pixels):
    """LiteRT의 해석기로 사진들을 한 장씩 계산한다."""
    interpreter = Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    source = interpreter.get_input_details()[0]["index"]
    target = interpreter.get_output_details()[0]["index"]
    outputs = []
    for image in pixels:
        interpreter.set_tensor(source, image[None, :])
        interpreter.invoke()
        outputs.append(interpreter.get_tensor(target)[0].copy())
    return np.stack(outputs)


def main():
    torch.manual_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    to_pixels = lambda raw: torch.from_numpy(raw.reshape(len(raw), -1).astype(np.float32) / 255.0)
    train_images = to_pixels(read_idx(DATA / "train-images-idx3-ubyte"))
    train_labels = torch.from_numpy(read_idx(DATA / "train-labels-idx1-ubyte").astype(np.int64))
    test_images = to_pixels(read_idx(DATA / "t10k-images-idx3-ubyte"))
    test_labels = read_idx(DATA / "t10k-labels-idx1-ubyte")

    # 7장의 가중치를 그대로 싣는다.
    mlp = Mlp()
    weights = load_file(ROOT / "models/mnist_mlp/model.safetensors")
    mlp.load_state_dict({name: torch.from_numpy(value) for name, value in weights.items()})

    print("잔차 모델을 학습합니다.")
    residual = Residual()
    train(residual, train_images, train_labels)

    reference = {}
    for name, model in (("mlp", mlp), ("residual", residual)):
        model.eval()
        path = OUT / f"{name}.tflite"
        litert_torch.convert(model, (test_images[:1],)).export(str(path))
        probabilities = run_litert(path, test_images.numpy())
        predictions = probabilities.argmax(axis=1)
        accuracy = (predictions == test_labels).mean()
        print(f"{name}: {path.stat().st_size:,}바이트, LiteRT의 정확도 {accuracy:.4f}")
        reference[f"{name}.probabilities"] = probabilities[:TRACED].astype(np.float32)
        reference[f"{name}.predictions"] = predictions.astype(np.int32)
    save_file(reference, OUT / "reference.safetensors")


if __name__ == "__main__":
    main()
