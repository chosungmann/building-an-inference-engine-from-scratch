"""36장: SmolVLM-256M의 정답을 남긴다.

실행:
    sh scripts/download_ch36.sh
    pip install -r reference/requirements_ch36.txt
    python reference/ch36_make_reference.py

출력 (models/smolvlm/):
    reference.safetensors  사진마다 전처리한 값, 비전 인코더의 출력, 커넥터의 출력,
                           글의 토큰, 글을 다 읽은 직후의 점수, 탐욕 디코딩이 쓴 토큰,
                           식으로 만든 사진을 Pillow의 란초스 필터로 줄인 것
    reference.txt          쓴 글

모델은 Hugging Face의 SmolVLM-256M-Instruct(Apache-2.0)이고 원래 저장소에서
받는다. 계산은 transformers가 한다. 사진을 여러 조각으로 나누는 기능은
끈다(`do_image_splitting=False`). 사진 한 장이 토큰 64개가 된다. 가중치는
16비트(bfloat16)로 저장되어 있고, 계산은 `f32`로 넓혀서 한다.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.numpy import save_file
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models/smolvlm"
REPO, REVISION = "HuggingFaceTB/SmolVLM-256M-Instruct", "d039ab0d4b9311b617e8ffec1d06c8464364aeb3"
IMAGES = ["tabby_cat", "pelican"]
QUESTION = "Describe this image."
LIMIT = 48


def synthetic(width, height):
    """식으로 만든 사진. crates/badak_image/tests/lanczos.rs 의 것과 같다."""
    rows, columns, channels = np.meshgrid(np.arange(height), np.arange(width), np.arange(3), indexing="ij")
    return ((rows * 7 + columns * 13 + channels * 31 + (rows * columns) % 251) % 256).astype(np.uint8)


# (가로, 세로, 새 가로, 새 세로). 줄이기, 세로로 크게 줄이기, 늘리기, 한 축만.
SIZES = [(517, 389, 341, 256), (300, 1000, 256, 200), (97, 61, 256, 161), (640, 480, 512, 480)]


def main():
    processor = AutoProcessor.from_pretrained(REPO, revision=REVISION)
    model = AutoModelForImageTextToText.from_pretrained(REPO, revision=REVISION, dtype=torch.float32).eval()
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QUESTION}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    reference, lines = {}, [f"글의 틀: {prompt!r}"]
    for name in IMAGES:
        image = Image.open(ROOT / f"models/images/{name}.jpg").convert("RGB")
        inputs = processor(text=prompt, images=[image], return_tensors="pt", do_image_splitting=False)
        pixels = inputs["pixel_values"][0]
        assert pixels.shape[0] == 1, pixels.shape
        with torch.no_grad():
            patches = model.model.vision_model(pixel_values=pixels).last_hidden_state
            seen = model.model.connector(patches)
            logits = model(**inputs).logits[0, -1]
            written = model.generate(**inputs, do_sample=False, max_new_tokens=LIMIT)[0, inputs["input_ids"].shape[1] :]
        reference[f"{name}.pixels"] = pixels[0].numpy()
        reference[f"{name}.patches"] = patches[0].numpy()
        reference[f"{name}.seen"] = seen[0].numpy()
        reference[f"{name}.ids"] = inputs["input_ids"][0].numpy().astype(np.int32)
        reference[f"{name}.logits"] = logits.numpy()
        reference[f"{name}.written"] = written.numpy().astype(np.int32)
        text = processor.decode(written, skip_special_tokens=True)
        lines.append(f"{name}: {text}")
        print(f"{name}: 토큰 {inputs['input_ids'].shape[1]}개를 읽고 {len(written)}개를 썼다")
        print(f"  {text}")
    # Pillow의 란초스 필터로 크기를 바꾼 것. 우리 구현과 바이트까지 견준다.
    for width, height, new_width, new_height in SIZES:
        image = Image.fromarray(synthetic(width, height))
        resized = image.resize((new_width, new_height), Image.LANCZOS)
        reference[f"synthetic.{width}x{height}.{new_width}x{new_height}"] = np.asarray(resized)
    save_file(reference, str(OUT / "reference.safetensors"))
    (OUT / "reference.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{OUT / 'reference.safetensors'} 기록")


if __name__ == "__main__":
    main()
