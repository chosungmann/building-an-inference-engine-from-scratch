"""37장: PyTorch가 OuteTTS-0.1-350M과 WavTokenizer 디코더를 실행하는 시간.

실행:
    OUTETTS=OuteTTS python reference/ch37_time.py

입력은 정답 파일에 든 것이다. 재기 전마다 REST초 쉰다(22.12절).
"""

import os
import statistics
import time
from pathlib import Path

import torch
from safetensors.numpy import load_file
from transformers import AutoModelForCausalLM

from ch37_make_reference import OUT, PENALTY, decoder_modules

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
    model = AutoModelForCausalLM.from_pretrained(OUT, dtype=torch.float32).eval()
    ids = torch.from_numpy(reference["ids"]).long()[None]
    count = len(reference["written"])
    VocosBackbone, ISTFTHead = decoder_modules()
    state = torch.load(OUT / "original/wavtokenizer.ckpt", map_location="cpu", weights_only=True)["state_dict"]
    backbone = VocosBackbone(input_channels=512, dim=768, intermediate_dim=2304, num_layers=12, adanorm_num_embeddings=4)
    head = ISTFTHead(dim=768, n_fft=1280, hop_length=320, padding="same")
    backbone.load_state_dict({name.removeprefix("backbone."): value for name, value in state.items() if name.startswith("backbone.")})
    head.load_state_dict({name.removeprefix("head."): value for name, value in state.items() if name.startswith("head.")})
    backbone.eval(), head.eval()
    codebook = state["feature_extractor.encodec.quantizer.vq.layers.0._codebook.embed"]
    codes = torch.from_numpy(reference["codes"]).long()
    band = torch.tensor([0])
    print(f"PyTorch {torch.__version__}")

    def write(limit):
        return model.generate(
            ids, attention_mask=torch.ones_like(ids), do_sample=False, repetition_penalty=PENALTY,
            max_new_tokens=limit, eos_token_id=-1, pad_token_id=0,
        )

    for threads in (1, 4, 12):
        torch.set_num_threads(threads)
        with torch.inference_mode():
            one = median_ms(5, lambda: write(1))
            many = median_ms(3, lambda: write(count))
            print(f"read    PyTorch 스레드 {threads:>2} 중앙값 {one:>8.2f}ms (토큰 {ids.shape[1]}개)", flush=True)
            print(f"token   PyTorch 스레드 {threads:>2} 중앙값 {(many - one) / (count - 1):>8.2f}ms", flush=True)
            for length in (75, 300, 1200):
                repeated = codes.repeat(length // len(codes) + 1)[:length]
                features = codebook[repeated].T[None]
                decode = median_ms(7, lambda: head(backbone(features, bandwidth_id=band)))
                print(f"decode  PyTorch 스레드 {threads:>2} 코드 {length:>4} 중앙값 {decode:>8.2f}ms", flush=True)


if __name__ == "__main__":
    main()
