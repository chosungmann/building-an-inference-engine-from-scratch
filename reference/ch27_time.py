"""27장: LiteRT와 PyTorch가 사진 한 장을 분류하는 시간을 잰다.

실행:
    python reference/ch27_time.py

입력은 reference.safetensors의 첫 사진이다. 스레드 수를 바꿔 가며 잰다.
"""

import statistics
import time
from pathlib import Path

import timm
import torch
from safetensors.numpy import load_file as load_numpy
from safetensors.torch import load_file as load_torch
from huggingface_hub import hf_hub_download

from ai_edge_litert.interpreter import Interpreter

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models/vision_tflite"
ROUNDS = 31
SOURCES = {
    "resnet50": ("resnet50.tv_in1k", "timm/resnet50.tv_in1k", "78f3ecfdb38e"),
    "mobilenetv2": ("mobilenetv2_100.ra_in1k", "timm/mobilenetv2_100.ra_in1k", "5afa12513b04"),
}


def median_ms(work):
    samples = []
    for _ in range(3 + ROUNDS):
        start = time.perf_counter()
        work()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples[3:]) * 1e3


def time_pillow():
    """Pillow가 사진을 풀고 줄이는 시간. 27장의 Rust 코드와 견준다."""
    from PIL import Image

    path = ROOT / "models/images/tabby_cat.jpg"
    image = Image.open(path).convert("RGB")
    print(f"Pillow, tabby_cat.jpg {image.size}")
    print(f"  파일 풀기   {median_ms(lambda: Image.open(path).convert('RGB')):>7.2f} ms")
    print(f"  줄이기      {median_ms(lambda: image.resize((341, 256), Image.Resampling.BILINEAR)):>7.2f} ms")


def main():
    time_pillow()
    x = load_numpy(MODELS / "reference.safetensors")["cup.input"]
    print(f"{'모델':<12} {'엔진':<10} {'스레드':>6} {'한 장 (ms)':>10}")
    for file, (name, repo, revision) in SOURCES.items():
        for threads in (1, 4, 8, 12, None):
            interpreter = Interpreter(model_path=str(MODELS / f"{file}.tflite"), num_threads=threads)
            interpreter.allocate_tensors()
            interpreter.set_tensor(interpreter.get_input_details()[0]["index"], x)
            label = "기본값" if threads is None else str(threads)
            print(f"{file:<12} {'LiteRT':<10} {label:>6} {median_ms(interpreter.invoke):>10.2f}")
        model = timm.create_model(name, pretrained=False)
        model.load_state_dict(load_torch(hf_hub_download(repo, "model.safetensors", revision=revision)))
        model.eval()
        tensor = torch.from_numpy(x)
        for threads in (1, 12):
            torch.set_num_threads(threads)
            with torch.no_grad():
                print(f"{file:<12} {'PyTorch':<10} {threads:>6} {median_ms(lambda: model(tensor)):>10.2f}")


if __name__ == "__main__":
    main()
