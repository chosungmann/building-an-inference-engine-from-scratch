"""35장: PyTorch(diffusers)가 U-Net 한 번과 디코더 한 번을 실행하는 시간.

실행:
    python reference/ch35_time.py

입력은 정답 파일에 든 것이다. 재기 전마다 REST초 쉰다(22.12절).
"""

import json
import os
import statistics
import time
from pathlib import Path

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from safetensors.numpy import load_file as load_numpy
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models/stable_diffusion"
REST = float(os.environ.get("REST", "3"))
RENAMES = {".query.": ".to_q.", ".key.": ".to_k.", ".value.": ".to_v.", ".proj_attn.": ".to_out.0."}


def load(kind, config, weights):
    model = kind.from_config(json.loads((OUT / config).read_text()))
    state = load_file(OUT / weights)
    for old, new in RENAMES.items():
        state = {name.replace(old, new) if "attentions" in name else name: value for name, value in state.items()}
    model.load_state_dict(state)
    return model.eval()


def median(rounds, work):
    time.sleep(REST)
    work()
    samples = []
    for _ in range(rounds):
        start = time.perf_counter()
        work()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def main():
    reference = load_numpy(OUT / "reference.safetensors")
    unet = load(UNet2DConditionModel, "unet_config.json", "unet.safetensors")
    vae = load(AutoencoderKL, "vae_config.json", "vae.safetensors")
    start = torch.from_numpy(reference["latent.start"])[None]
    end = torch.from_numpy(reference["latent.end"])[None]
    context = torch.from_numpy(reference["context.0"])[None]
    print(f"PyTorch {torch.__version__}")
    for threads in (1, 4, 12):
        torch.set_num_threads(threads)
        with torch.inference_mode():
            rounds = 5 if threads > 1 else 3
            seconds = median(rounds, lambda: unet(start, 951, encoder_hidden_states=context).sample)
            print(f"unet PyTorch 스레드 {threads:>2} 중앙값 {seconds:>7.3f}초", flush=True)
            seconds = median(rounds, lambda: vae.decode(end / vae.config.scaling_factor).sample)
            print(f"vae  PyTorch 스레드 {threads:>2} 중앙값 {seconds:>7.3f}초", flush=True)


if __name__ == "__main__":
    main()
