"""37장: OuteTTS-0.1-350M과 WavTokenizer 디코더의 정답을 남긴다.

실행:
    sh scripts/download_ch37.sh
    git clone https://github.com/edwko/OuteTTS  (커밋 f5eac6e70d79)
    pip install -r reference/requirements_ch37.txt
    OUTETTS=OuteTTS python reference/ch37_make_reference.py

출력 (models/outetts/):
    reference.safetensors  글의 토큰, 글을 읽은 직후의 점수, 탐욕 디코딩이 쓴 토큰,
                           거기서 뽑은 오디오 코드, 디코더 backbone의 출력, 소리
    reference.wav          그 소리 (24kHz, 16비트)

언어 모델은 transformers가 계산한다. 오디오 디코더는 OuteTTS 저장소
(Apache-2.0)에 들어 있는 WavTokenizer의 모듈을 그대로 쓰고, 가중치는
원본 체크포인트(MIT)에서 `weights_only=True`로 읽는다. 이름이 하나라도 맞지
않으면 멈춘다.

글을 고르는 방법은 탐욕 디코딩에 반복 벌점 1.1이다. 무작위가 없어 우리
구현과 토큰 단위로 견줄 수 있다.
"""

import os
import sys
import types
import wave
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models/outetts"
TEXT = "the engine reads a model file and turns text into speech"
LIMIT = 700
PENALTY = 1.1


def decoder_modules():
    """OuteTTS 저장소의 디코더 모듈을 그 패키지의 나머지 없이 불러온다."""
    source = Path(os.environ["OUTETTS"]) / "outetts/wav_tokenizer/decoder"
    package = types.ModuleType("decoder")
    package.__path__ = [str(source)]
    sys.modules["decoder"] = package
    # heads.py 가 이 장에서 쓰지 않는 함수 둘을 torchaudio에서 가져온다.
    stub = types.ModuleType("torchaudio.functional.functional")
    stub._hz_to_mel = stub._mel_to_hz = None
    for name in ("torchaudio", "torchaudio.functional"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["torchaudio.functional.functional"] = stub
    from decoder.heads import ISTFTHead
    from decoder.models import VocosBackbone

    return VocosBackbone, ISTFTHead


def main():
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(OUT / "tokenizer.json"))
    model = AutoModelForCausalLM.from_pretrained(OUT, dtype=torch.float32).eval()
    words = "<|text_sep|>".join(TEXT.split())
    prompt = f"<|im_start|>\n<|text_start|>{words}<|text_end|>\n<|audio_start|>\n"
    ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
    stop = tokenizer.convert_tokens_to_ids("<|audio_end|>")
    with torch.no_grad():
        logits = model(ids).logits[0, -1]
        written = model.generate(
            ids,
            do_sample=False,
            repetition_penalty=PENALTY,
            max_new_tokens=LIMIT,
            eos_token_id=stop,
            pad_token_id=stop,
        )[0, ids.shape[1] :]
    text = tokenizer.decode(written)
    first, last = (tokenizer.convert_tokens_to_ids(f"<|{code}|>") for code in (0, 4095))
    assert last - first == 4095, "오디오 토큰의 번호가 이어져 있지 않다"
    codes = torch.tensor([int(token) - first for token in written if first <= token <= last])
    print(f"토큰 {ids.shape[1]}개를 읽고 {len(written)}개를 썼다. 오디오 코드 {len(codes)}개 ({len(codes) / 75:.2f}초)")
    print(text[:300])

    # 오디오 디코더.
    VocosBackbone, ISTFTHead = decoder_modules()
    state = torch.load(OUT / "original/wavtokenizer.ckpt", map_location="cpu", weights_only=True)["state_dict"]
    backbone = VocosBackbone(input_channels=512, dim=768, intermediate_dim=2304, num_layers=12, adanorm_num_embeddings=4)
    head = ISTFTHead(dim=768, n_fft=1280, hop_length=320, padding="same")
    backbone.load_state_dict({name.removeprefix("backbone."): value for name, value in state.items() if name.startswith("backbone.")})
    head.load_state_dict({name.removeprefix("head."): value for name, value in state.items() if name.startswith("head.")})
    backbone.eval(), head.eval()
    codebook = state["feature_extractor.encodec.quantizer.vq.layers.0._codebook.embed"]
    with torch.no_grad():
        features = codebook[codes].T[None]
        hidden = backbone(features, bandwidth_id=torch.tensor([0]))
        audio = head(hidden)[0]
    print(f"소리 {len(audio)}개 ({len(audio) / 24000:.2f}초), 가장 큰 값 {audio.abs().max():.3f}")

    reference = {
        "ids": ids[0].numpy().astype(np.int32),
        "logits": logits.numpy(),
        "written": written.numpy().astype(np.int32),
        "codes": codes.numpy().astype(np.int32),
        "hidden": hidden[0].numpy(),
        "audio": audio.numpy(),
    }
    save_file(reference, str(OUT / "reference.safetensors"))
    with wave.open(str(OUT / "reference.wav"), "wb") as file:
        file.setnchannels(1), file.setsampwidth(2), file.setframerate(24000)
        file.writeframes((audio.clamp(-1, 1) * 32767).round().short().numpy().tobytes())
    print(f"{OUT / 'reference.safetensors'} 기록")


if __name__ == "__main__":
    main()
