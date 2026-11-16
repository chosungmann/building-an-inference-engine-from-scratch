"""7장: 손글씨 숫자 분류 모델을 학습하고, badak을 검증할 정답을 만든다.

실행:
    sh scripts/download_mnist.sh
    python reference/ch07_train_mnist.py

출력 (models/mnist_mlp/):
    model.safetensors      학습한 가중치
    reference.safetensors  PyTorch가 계산한 층별 출력과 예측

모델은 3장에서 크기를 계산해 본 다층 퍼셉트론이다. 입력 784개, 은닉층
128개, 출력 10개이고 파라미터는 101,770개다.
"""

from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "models/mnist"
OUT = ROOT / "models/mnist_mlp"

# 학습 데이터 전체의 평균과 표준편차. 입력을 이 값으로 정규화한다.
MEAN, STD = 0.1307, 0.3081
SEED = 0
EPOCHS = 5
BATCH = 64
# 층별 출력을 저장해 둘 시험 이미지의 수.
TRACED = 16


def read_idx(path):
    """IDX 파일을 NumPy 배열로 읽는다. 머리의 숫자는 빅 엔디언이다."""
    raw = path.read_bytes()
    ndim = raw[3]
    shape = [int.from_bytes(raw[4 + 4 * i : 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


# ANCHOR: preprocess
def preprocess(images):
    """픽셀(0~255)을 실수로 바꾸고 정규화한 뒤 한 줄로 편다."""
    x = images.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return x.reshape(len(images), 28 * 28)
# ANCHOR_END: preprocess


# ANCHOR: model
class Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))
# ANCHOR_END: model


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(1)

    train_x = torch.from_numpy(preprocess(read_idx(DATA / "train-images-idx3-ubyte")))
    train_y = torch.from_numpy(read_idx(DATA / "train-labels-idx1-ubyte").astype(np.int64))
    test_x = torch.from_numpy(preprocess(read_idx(DATA / "t10k-images-idx3-ubyte")))
    test_y = torch.from_numpy(read_idx(DATA / "t10k-labels-idx1-ubyte").astype(np.int64))

    model = Mlp()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    order = torch.Generator().manual_seed(SEED)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for index in torch.randperm(len(train_x), generator=order).split(BATCH):
            optimizer.zero_grad()
            loss = loss_fn(model(train_x[index]), train_y[index])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(index)
        model.eval()
        with torch.no_grad():
            accuracy = (model(test_x).argmax(1) == test_y).float().mean().item()
        print(f"에폭 {epoch}: 손실 {total / len(train_x):.4f}, 시험 정확도 {accuracy * 100:.2f}%")

    # 정답: 앞의 몇 장에 대한 층별 출력과, 시험 데이터 전체의 예측.
    model.eval()
    with torch.no_grad():
        x = test_x[:TRACED]
        fc1 = model.fc1(x)
        relu = torch.relu(fc1)
        logits = model.fc2(relu)
        all_logits = model(test_x)
        predictions = all_logits.argmax(1)
    correct = int((predictions == test_y).sum())

    OUT.mkdir(parents=True, exist_ok=True)
    weights = {name: value.numpy() for name, value in model.state_dict().items()}
    save_file(
        weights,
        OUT / "model.safetensors",
        metadata={
            "architecture": "mlp 784-128-10, relu",
            "input_mean": str(MEAN),
            "input_std": str(STD),
            "trained_with": f"torch {torch.__version__}",
        },
    )
    save_file(
        {
            "input": x.numpy(),
            "fc1": fc1.numpy(),
            "relu": relu.numpy(),
            "logits": logits.numpy(),
            "all_logits": all_logits.numpy(),
            "predictions": predictions.numpy().astype(np.int32),
        },
        OUT / "reference.safetensors",
        metadata={"correct": str(correct), "total": str(len(test_y))},
    )
    parameters = sum(p.numel() for p in model.parameters())
    print(f"파라미터 {parameters}개, 시험 데이터 {len(test_y)}장 중 {correct}장 정답")
    for name in ("model.safetensors", "reference.safetensors"):
        print(f"{name}: {(OUT / name).stat().st_size}바이트")


if __name__ == "__main__":
    main()
