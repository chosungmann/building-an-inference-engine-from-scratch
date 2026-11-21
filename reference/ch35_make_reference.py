"""35장: Stable Diffusion 1.5의 정답을 남긴다.

실행:
    sh scripts/download_ch35.sh
    pip install -r reference/requirements_ch35.txt
    python reference/ch35_make_reference.py

출력 (models/stable_diffusion/):
    reference.safetensors  글의 토큰, 텍스트 인코더의 출력, U-Net이 한 번 짐작한
                           잡음, 20스텝을 다 밟은 잠재 그림, 디코더가 푼 그림
    reference.png          그 그림

모델은 Stable Diffusion 1.5(CreativeML OpenRAIL-M)이고 Hugging Face의
저장소에서 받는다. 계산은 diffusers와 transformers가 한다. 부품을 하나씩
읽어 반복문을 직접 돌린다(파이프라인의 안전 검사기는 쓰지 않는다). 이
스크립트는 가중치를 바꾸지 않는다.
"""

import json
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from PIL import Image
from safetensors.numpy import save_file
from safetensors.torch import load_file
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models/stable_diffusion"
PROMPTS = [
    "a watercolor painting of a lighthouse on a cliff at sunset",
    "",
    "A photo of 2 cats, isn't it?! Très bien.",
]
STEPS, GUIDANCE, SEED = 20, 7.5, 35


def load(kind, config, weights):
    model = kind.from_config(json.loads((OUT / config).read_text()))
    state = load_file(OUT / weights)
    # VAE의 파일은 어텐션의 가중치를 옛 이름으로 담고 있다. diffusers가
    # 파일을 읽을 때 하는 것과 같이 지금의 이름으로 바꾼다.
    renames = {".query.": ".to_q.", ".key.": ".to_k.", ".value.": ".to_v.", ".proj_attn.": ".to_out.0."}
    for old, new in renames.items():
        state = {name.replace(old, new) if "attentions" in name else name: value for name, value in state.items()}
    model.load_state_dict(state)
    return model.eval()


def main():
    tokenizer = CLIPTokenizer(str(OUT / "vocab.json"), str(OUT / "merges.txt"))
    text_encoder = CLIPTextModel(CLIPTextConfig.from_json_file(OUT / "text_encoder_config.json"))
    # 파일의 이름은 `text_model.`으로 시작하고 이 버전의 transformers는 그
    # 앞머리 없이 읽는다. 자리 번호의 표(`position_ids`)는 가중치가 아니다.
    # 이름이 하나라도 맞지 않으면 멈춘다.
    state = {
        name.removeprefix("text_model."): value
        for name, value in load_file(OUT / "text_encoder.safetensors").items()
        if not name.endswith("position_ids")
    }
    text_encoder.load_state_dict(state)
    text_encoder.eval()
    unet = load(UNet2DConditionModel, "unet_config.json", "unet.safetensors")
    vae = load(AutoencoderKL, "vae_config.json", "vae.safetensors")
    scheduler = DDIMScheduler.from_config(json.loads((OUT / "scheduler_config.json").read_text()))
    reference = {}

    # 1. 토큰과 텍스트 인코더.
    contexts = []
    for index, prompt in enumerate(PROMPTS):
        ids = tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
        with torch.no_grad():
            context = text_encoder(ids).last_hidden_state
        reference[f"ids.{index}"] = ids[0].numpy().astype(np.int32)
        reference[f"context.{index}"] = context[0].numpy()
        contexts.append(context)
    context, empty = contexts[0], contexts[1]

    # 2. U-Net 한 번. 첫 단계에서 글을 주고 짐작한 잡음.
    generator = torch.Generator().manual_seed(SEED)
    latent = torch.randn((1, 4, 64, 64), generator=generator)
    scheduler.set_timesteps(STEPS)
    first = int(scheduler.timesteps[0])
    with torch.no_grad():
        noise = unet(latent, first, encoder_hidden_states=context).sample
    reference["latent.start"] = latent[0].numpy()
    reference["noise.first"] = noise[0].numpy()
    reference["timesteps"] = scheduler.timesteps.numpy().astype(np.int32)

    # 3. 20스텝.
    x = latent.clone()
    for step in scheduler.timesteps:
        with torch.no_grad():
            conditional = unet(x, step, encoder_hidden_states=context).sample
            unconditional = unet(x, step, encoder_hidden_states=empty).sample
        guided = unconditional + GUIDANCE * (conditional - unconditional)
        x = scheduler.step(guided, step, x).prev_sample
        print(f"단계 {int(step):>3}", flush=True)
    reference["latent.end"] = x[0].numpy()

    # 4. 디코더.
    with torch.no_grad():
        image = vae.decode(x / vae.config.scaling_factor).sample
    reference["image"] = image[0].numpy()
    pixels = ((image[0] / 2 + 0.5).clamp(0, 1) * 255).round().permute(1, 2, 0).numpy().astype(np.uint8)
    Image.fromarray(pixels).save(OUT / "reference.png")

    save_file(reference, str(OUT / "reference.safetensors"))
    print(f"{OUT / 'reference.safetensors'} 기록")


if __name__ == "__main__":
    main()
