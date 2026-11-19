"""32장: PyTorch가 ViT-Tiny를, ONNX Runtime이 YOLOX-Tiny를 실행하는 시간.

실행:
    python reference/ch32_time.py

입력은 정답 파일의 첫 사진이다. 재기 전마다 REST초 쉰다(22.12절).
"""

import os
import statistics
import time
from pathlib import Path

import onnxruntime
import timm
import torch
from safetensors.numpy import load_file as load_numpy
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models/vision_transformer"
ROUNDS = 51
REST = float(os.environ.get("REST", "3"))


def median_ms(work):
    time.sleep(REST)
    samples = []
    for index in range(5 + ROUNDS):
        start = time.perf_counter()
        work()
        if index >= 5:
            samples.append(time.perf_counter() - start)
    return statistics.median(samples) * 1e3


def main():
    reference = load_numpy(MODELS / "reference.safetensors")
    x = torch.from_numpy(reference["vit.cup.input"])
    print(f"PyTorch {torch.__version__}, ONNX Runtime {onnxruntime.__version__}")
    print(f"{'모델':<12} {'엔진':<14} {'스레드':>4} {'한 장 (ms)':>10}")
    names = {"vit_tiny": "vit_tiny_patch16_224.augreg_in21k_ft_in1k", "vit_base": "vit_base_patch16_224.augreg2_in21k_ft_in1k"}
    for file, name in names.items():
        if not (MODELS / f"{file}.safetensors").exists():
            continue
        model = timm.create_model(name, pretrained=False)
        model.load_state_dict(load_file(MODELS / f"{file}.safetensors"))
        model.eval()
        for threads in (1, 4, 12):
            torch.set_num_threads(threads)
            with torch.no_grad():
                print(f"{file:<12} {'PyTorch':<14} {threads:>4} {median_ms(lambda: model(x)):>10.2f}", flush=True)
    image = reference["yolox.cup.input"]
    for threads in (1, 4, 12):
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = threads
        session = onnxruntime.InferenceSession(str(MODELS / "yolox_tiny.onnx"), options, providers=["CPUExecutionProvider"])
        value = median_ms(lambda: session.run(None, {"images": image}))
        print(f"{'yolox_tiny':<12} {'ONNX Runtime':<14} {threads:>4} {value:>10.2f}", flush=True)


if __name__ == "__main__":
    main()
