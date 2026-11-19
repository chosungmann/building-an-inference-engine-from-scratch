"""38장: 음성 대화 파이프라인의 정답을 남긴다.

실행:
    sh scripts/download_ch33.sh && sh scripts/download_smollm2.sh && sh scripts/download_ch37.sh
    pip install -r reference/requirements_ch38.txt
    python reference/ch38_make_reference.py

출력 (models/talk/):
    reference.safetensors  받아쓴 토큰, 답의 토큰, 문장마다 음성 합성이 쓴 토큰
    reference.txt          받아쓴 글, 답, 문장들

모델 셋을 transformers로 차례로 실행한다. 받아쓰기는 Whisper-tiny(탐욕),
답은 SmolLM2-135M-Instruct(탐욕, 64토큰까지), 음성 합성은 OuteTTS-0.1-350M
(탐욕, 반복 벌점 1.1)이다. 무작위가 없으므로 우리 구현과 토큰 단위로
견줄 수 있다. 오디오 디코더는 37장에서 따로 대조했으므로 여기서는 오디오
토큰까지만 남긴다.
"""

import re
import wave
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
OUT = MODELS / "talk"
WHISPER, WHISPER_REVISION = "openai/whisper-tiny", "169d4a4341b33bc18d8881c4b69c2e104e1cc0af"
CHAT = "HuggingFaceTB/SmolLM2-135M-Instruct"
SYSTEM = "You are a friendly voice assistant. Answer in two short sentences."


def sentences(text):
    """문장으로 나눈다. 마침표, 물음표, 느낌표 뒤에 빈칸이 오는 곳에서."""
    parts, start = [], 0
    for index in range(1, len(text)):
        if text[index].isspace() and text[index - 1] in ".!?":
            parts.append(text[start:index])
            start = index
    parts.append(text[start:])
    return [part.strip() for part in parts if any(character.isalpha() for character in part)]


def speech_prompt(sentence):
    """37장의 틀. crates/badak_models/src/tts.rs 의 prompt와 같다."""
    cleaned = re.sub(r"[-_/,\.\\]", " ", sentence.lower())
    cleaned = re.sub(r"[^a-z\s]", "", cleaned)
    return "<|im_start|>\n<|text_start|>" + "<|text_sep|>".join(cleaned.split()) + "<|text_end|>\n<|audio_start|>\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reference, lines = {}, []

    # 1. 받아쓰기.
    with wave.open(str(MODELS / "whisper/jfk.wav")) as file:
        audio = np.frombuffer(file.readframes(file.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    processor = WhisperProcessor.from_pretrained(WHISPER, revision=WHISPER_REVISION)
    whisper = WhisperForConditionalGeneration.from_pretrained(WHISPER, revision=WHISPER_REVISION).eval()
    features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
    with torch.no_grad():
        heard_ids = whisper.generate(
            features, num_beams=1, do_sample=False, language="en", task="transcribe", return_timestamps=False, max_new_tokens=200
        )[0]
    heard = processor.decode(heard_ids, skip_special_tokens=True).strip()
    reference["heard"] = heard_ids.numpy().astype(np.int32)
    lines.append(f"들은 말: {heard}")

    # 2. 답.
    tokenizer = AutoTokenizer.from_pretrained(MODELS / "smollm2_135m")
    chat = AutoModelForCausalLM.from_pretrained(MODELS / "smollm2_135m", dtype=torch.float32).eval()
    text = (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{heard}<|im_end|>\n<|im_start|>assistant\n"
    )
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
    with torch.no_grad():
        written = chat.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False, max_new_tokens=64)[0, ids.shape[1] :]
    # 끝 토큰은 모델의 설정에 적힌 번호다. 그 토큰에서 답이 끝난다.
    end = chat.config.eos_token_id
    assert isinstance(end, int), end
    written = torch.tensor([int(token) for token in written if int(token) != end])
    answer = tokenizer.decode(written.tolist(), skip_special_tokens=False)
    # 이 버전의 transformers는 여기서 목록을 돌려주기도 한다.
    answer = answer[0] if isinstance(answer, list) else answer
    reference["prompt"] = ids[0].numpy().astype(np.int32)
    reference["answer"] = written.numpy().astype(np.int32)
    lines.append(f"답: {answer}")

    # 3. 문장마다 오디오 토큰.
    speech_tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(MODELS / "outetts/tokenizer.json"))
    speech = AutoModelForCausalLM.from_pretrained(MODELS / "outetts", dtype=torch.float32).eval()
    stop = speech_tokenizer.convert_tokens_to_ids("<|audio_end|>")
    parts = sentences(answer)
    for index, sentence in enumerate(parts):
        ids = speech_tokenizer(speech_prompt(sentence), add_special_tokens=False, return_tensors="pt").input_ids
        with torch.no_grad():
            tokens = speech.generate(
                ids,
                attention_mask=torch.ones_like(ids),
                do_sample=False,
                repetition_penalty=1.1,
                max_new_tokens=1500,
                eos_token_id=stop,
                pad_token_id=stop,
            )[0, ids.shape[1] :]
        reference[f"speech.{index}"] = tokens.numpy().astype(np.int32)
        lines.append(f"문장 {index + 1} (소리 토큰 {len(tokens)}개): {sentence}")
    reference["sentences"] = np.array([len(parts)], dtype=np.int32)

    save_file(reference, str(OUT / "reference.safetensors"))
    (OUT / "reference.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"{OUT / 'reference.safetensors'} 기록")


if __name__ == "__main__":
    main()
