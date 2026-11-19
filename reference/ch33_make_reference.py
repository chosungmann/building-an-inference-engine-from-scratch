"""33장: Whisper-tiny의 정답을 남긴다.

실행:
    sh scripts/download_ch33.sh
    pip install -r reference/requirements_ch33.txt
    python reference/ch33_make_reference.py

출력 (models/whisper/):
    reference.safetensors  소리마다 멜 스펙트로그램, 인코더의 출력, 디코더의 첫 점수,
                           탐욕 디코딩과 빔 서치가 고른 토큰들
    reference.txt          받아쓴 글

모델은 OpenAI의 Whisper-tiny(Apache-2.0)이고 Hugging Face의 원래 저장소에서
받는다. 계산은 transformers 라이브러리가 한다. 이 스크립트는 가중치를
바꾸지 않는다.
"""

import wave
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import WhisperForConditionalGeneration, WhisperProcessor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models/whisper"
REPO, REVISION = "openai/whisper-tiny", "169d4a4341b33bc18d8881c4b69c2e104e1cc0af"
SOUNDS = ["jfk"]
# 시작, 영어, 받아쓰기, 시각 없이.
PREFIX = [50258, 50259, 50359, 50363]


def read_wav(path):
    with wave.open(str(path)) as file:
        assert (file.getnchannels(), file.getsampwidth(), file.getframerate()) == (1, 2, 16000)
        samples = np.frombuffer(file.readframes(file.getnframes()), dtype="<i2")
    return samples.astype(np.float32) / 32768.0


def main():
    processor = WhisperProcessor.from_pretrained(REPO, revision=REVISION)
    model = WhisperForConditionalGeneration.from_pretrained(REPO, revision=REVISION).eval()
    reference, lines = {}, []
    for name in SOUNDS:
        audio = read_wav(OUT / f"{name}.wav")
        features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
        with torch.no_grad():
            encoded = model.model.encoder(features).last_hidden_state
            prefix = torch.tensor([PREFIX])
            logits = model(encoder_outputs=(encoded,), decoder_input_ids=prefix).logits[0, -1]
            options = dict(language="en", task="transcribe", return_timestamps=False, max_new_tokens=200)
            greedy = model.generate(features, num_beams=1, do_sample=False, **options)[0]
            beam = model.generate(features, num_beams=5, do_sample=False, **options)[0]
        reference[f"{name}.mel"] = features[0].numpy()
        reference[f"{name}.encoded"] = encoded[0].numpy()
        reference[f"{name}.logits"] = logits.numpy()
        reference[f"{name}.greedy"] = greedy.numpy().astype(np.int32)
        reference[f"{name}.beam"] = beam.numpy().astype(np.int32)
        for label, ids in (("탐욕", greedy), ("빔 5", beam)):
            text = processor.decode(ids, skip_special_tokens=True)
            lines.append(f"{name} {label}: {text}")
            print(lines[-1])
        print(f"  멜 {tuple(features.shape)}, 인코더 {tuple(encoded.shape)}, 토큰 {greedy.tolist()}")
    save_file(reference, OUT / "reference.safetensors")
    (OUT / "reference.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
