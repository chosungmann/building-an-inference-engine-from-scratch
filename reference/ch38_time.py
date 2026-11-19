"""38장: PyTorch로 같은 사슬을 돌릴 때 첫 소리까지의 구간마다의 시간.

실행:
    python reference/ch38_time.py

정답 파일의 토큰을 써서, 우리 파이프라인이 문장마다 넘길 때와 같은 양의
일을 시킨다. 받아쓰기 전체, 답의 첫 문장까지(그 문장의 토큰 수 + 1),
첫 문장의 첫 낱말의 오디오 토큰까지. 오디오 디코더의 시간은 넣지 않는다
(37장에서 따로 쟀다). 재기 전마다 REST초 쉰다(22.12절).
"""

import os
import statistics
import time
import wave

import numpy as np
import torch
from safetensors.numpy import load_file
from transformers import (
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from ch38_make_reference import MODELS, OUT, WHISPER, WHISPER_REVISION, speech_prompt

REST = float(os.environ.get("REST", "3"))


def median_ms(rounds, work):
    time.sleep(REST)
    work()
    samples = []
    for _ in range(rounds):
        start = time.perf_counter()
        work()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples) * 1e3


def main():
    reference = load_file(OUT / "reference.safetensors")
    with wave.open(str(MODELS / "whisper/jfk.wav")) as file:
        audio = np.frombuffer(file.readframes(file.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    processor = WhisperProcessor.from_pretrained(WHISPER, revision=WHISPER_REVISION)
    whisper = WhisperForConditionalGeneration.from_pretrained(WHISPER, revision=WHISPER_REVISION).eval()
    chat = AutoModelForCausalLM.from_pretrained(MODELS / "smollm2_135m", dtype=torch.float32).eval()
    speech = AutoModelForCausalLM.from_pretrained(MODELS / "outetts", dtype=torch.float32).eval()
    speech_tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(MODELS / "outetts/tokenizer.json"))
    word_end = speech_tokenizer.convert_tokens_to_ids("<|code_end|>")

    prompt = torch.from_numpy(reference["prompt"]).long()[None]
    answer = reference["answer"].tolist()
    lines = (OUT / "reference.txt").read_text(encoding="utf-8").splitlines()
    first_sentence = lines[2].split(": ", 1)[1]
    first_tokens = reference["speech.0"].tolist()
    first_word = first_tokens.index(word_end) + 1
    speech_ids = speech_tokenizer(speech_prompt(first_sentence), add_special_tokens=False, return_tensors="pt").input_ids
    # 답의 첫 문장은 9토큰이다. 문장이 끝난 줄 알려면 그다음 토큰 하나를 더 봐야 한다.
    sentence_tokens = int(os.environ.get("FIRST_SENTENCE_TOKENS", "10"))
    assert sentence_tokens <= len(answer)
    print(f"PyTorch {torch.__version__}")

    def listen():
        features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
        return whisper.generate(
            features, num_beams=1, do_sample=False, language="en", task="transcribe", return_timestamps=False, max_new_tokens=200
        )

    def write(limit):
        return chat.generate(prompt, attention_mask=torch.ones_like(prompt), do_sample=False, max_new_tokens=limit, eos_token_id=-1, pad_token_id=0)

    def speak(limit):
        return speech.generate(
            speech_ids, attention_mask=torch.ones_like(speech_ids), do_sample=False, repetition_penalty=1.1,
            max_new_tokens=limit, eos_token_id=-1, pad_token_id=0,
        )

    for threads in (1, 4, 12):
        torch.set_num_threads(threads)
        with torch.inference_mode():
            heard = median_ms(5, listen)
            sentence = median_ms(5, lambda: write(sentence_tokens))
            whole = median_ms(3, lambda: write(len(answer)))
            word = median_ms(5, lambda: speak(first_word))
            print(
                f"talk PyTorch 스레드 {threads:>2} 들음 {heard:.0f} 첫문장까지 {sentence:.0f} 답전체 {whole:.0f} "
                f"첫낱말의소리토큰 {word:.0f} 문장마다의합 {heard + sentence + word:.0f} 다쓴뒤의합 {heard + whole + word:.0f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
