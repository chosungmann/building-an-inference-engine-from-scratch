"""46장: SmolLM2-135M을 줄이는 세 가지 방법을 작게 해 본다.

실행:
    sh scripts/download_smollm2.sh
    sh scripts/download_wikitext2.sh
    pip install -r reference/requirements_ch46.txt
    python reference/ch46_train.py [방법 ...]

방법 (주지 않으면 모두):
    ft          그냥 조금 더 학습시킨다. 견줄 기준이다.
    qat         엔진의 Q4_0 양자화를 흉내 내면서 학습시킨다 (양자화 인식 학습).
    qat_kd      Q4_0을 흉내 낸 모델이 원래 모델의 답을 따라 하게 학습시킨다.
    drop        블록 서른 개 가운데 뒤의 열 개를 덜어 낸다. 학습은 하지 않는다.
    drop_ft     덜어 낸 모델을 조금 더 학습시킨다.
    drop_kd     덜어 낸 모델이 원래 모델의 답을 따라 하게 학습시킨다 (증류).
    pick, pick_ft, pick_kd
                위의 셋과 같되, 덜어 낼 열 개를 엔진이 골랐다. 하나씩 빼
                봤을 때 손실이 가장 적게 오른 블록들이다 (ch46_prune blocks).
    pick_qat_kd pick_kd에 Q4_0 흉내를 더한다. 블록도 덜고 4비트로도 줄인다.
    prune_kd    블록의 행렬마다 절댓값이 작은 절반을 0으로 만들고, 0은 0으로
                둔 채 원래 모델의 답을 따라 하게 학습시킨다.
    vectors     학습은 하지 않고, 흉내 양자화가 엔진의 것과 같은지 견줄 값만
                적는다.

출력: models/smollm2_135m_<방법>/ (config.json, model.safetensors,
tokenizer.json). badak이 그대로 읽는다. 품질은 badak으로 잰다
(crates/badak_models/examples/ch46_quality.rs).

Apple Silicon의 GPU(MPS)나 NVIDIA GPU를 쓴다. 이 책의 맥(M4 Max)에서 방법
하나에 몇 분이 걸린다. 시드를 고정했지만 학습은 기계와 라이브러리의 판에
따라 끝자리가 달라진다. 책의 숫자는 책의 맥에서 한 번 돌린 것이다.
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.func import functional_call
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "models/smollm2_135m"
TEXT = ROOT / "models/wikitext2/wikitext-2-raw/wiki.train.raw"

SEED = 0
# 학습 스텝의 수. 짧게 시험해 보려면 환경 변수 STEPS로 줄인다.
STEPS = int(os.environ.get("STEPS", "400"))
BATCH = 8
LENGTH = 256
RATE = 2e-5
# Q4_0은 값 32개마다 눈금 하나를 둔다 (20장).
BLOCK = 32
# 덜어 낼 블록들. 뒤의 열 개와, 엔진이 고른 열 개.
LAST = list(range(20, 30))
PICKED = [3, 4, 5, 7, 12, 13, 14, 15, 17, 25]
# 깎을 비율.
PRUNED = 0.5


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ANCHOR: fake_q4_0
def fake_q4_0(weight):
    """엔진이 Q4_0으로 줄였다가 되돌린 값을 계산한다.

    crates/badak_core/src/packed/mod.rs 의 quantize_block_q4_0 과 같은 일을
    한다. 값 32개마다 절댓값이 가장 큰 값을 찾아 그것이 -8에 오게 눈금을
    정하고, 0.5를 더해 버림으로 반올림하고, 15에서 자른다. 눈금의 크기는
    16비트 실수로 적히므로 그것도 흉내 낸다.
    """
    rows, cols = weight.shape
    blocks = weight.reshape(rows, cols // BLOCK, BLOCK)
    index = blocks.abs().argmax(dim=-1, keepdim=True)
    signed = blocks.gather(-1, index)
    scale = (signed / -8.0).half().float()
    inverse = torch.where(scale != 0, 1.0 / scale, torch.zeros_like(scale))
    codes = torch.clamp(torch.trunc(blocks * inverse + 8.5), max=15.0)
    return ((codes - 8.0) * scale).reshape(rows, cols)


def quantized(parameters):
    """행렬 가중치를 줄였다 되돌린 값으로 바꾼 사본.

    순전파는 줄인 값으로 하고, 기울기는 줄이지 않은 값으로 그대로 흘린다
    (straight-through estimator). 반올림은 기울기가 어디서나 0이어서, 그대로
    두면 아무것도 배우지 못한다.
    """
    out = {}
    for name, weight in parameters.items():
        if weight.dim() == 2:
            out[name] = weight + (fake_q4_0(weight) - weight).detach()
        else:
            out[name] = weight
    return out
# ANCHOR_END: fake_q4_0


def batches(tokenizer, generator):
    """학습 글에서 무작위 위치의 조각을 뽑아 내는 무한한 열."""
    ids = tokenizer(TEXT.read_text(encoding="utf-8"), return_tensors="pt").input_ids[0]
    while True:
        starts = torch.randint(0, len(ids) - LENGTH - 1, (BATCH,), generator=generator)
        yield torch.stack([ids[start : start + LENGTH] for start in starts])


def drop_blocks(model, dropped):
    """번호가 `dropped`에 든 블록들을 덜어 낸다."""
    kept = [layer for index, layer in enumerate(model.model.layers) if index not in dropped]
    # 블록은 자기 번호를 기억하고 있다(KV 캐시의 자리). 다시 매긴다.
    for index, layer in enumerate(kept):
        layer.self_attn.layer_idx = index
    model.model.layers = torch.nn.ModuleList(kept)
    model.config.num_hidden_layers = len(kept)
    return model


# ANCHOR: masks
def magnitude_masks(model, fraction):
    """블록의 행렬마다, 절댓값이 작은 것부터 `fraction`만큼을 가리는 표.

    임베딩 표는 깎지 않는다. badak의 `prune_magnitude`와 같은 기준이다.
    """
    masks = {}
    for name, weight in model.named_parameters():
        if weight.dim() == 2 and "embed_tokens" not in name:
            cut = round(weight.numel() * fraction)
            threshold = weight.abs().flatten().kthvalue(cut).values
            masks[name] = (weight.abs() > threshold).to(weight.dtype)
    return masks


def apply_masks(model, masks):
    """가린 자리를 0으로 만든다. 학습 스텝마다 다시 부른다."""
    with torch.no_grad():
        for name, weight in model.named_parameters():
            if name in masks:
                weight.mul_(masks[name])
# ANCHOR_END: masks


# ANCHOR: losses
def next_token_loss(logits, ids):
    """다음 토큰을 맞히는 손실. 보통의 학습이 줄이는 것이다."""
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))


def imitation_loss(logits, teacher_logits):
    """선생의 확률 분포를 따라 하는 손실 (KL 발산).

    정답 토큰 하나가 아니라 선생이 모든 토큰에 준 확률을 본다. "이
    자리에는 the가 60%, a가 30%"라는 것까지 배운다.
    """
    student = F.log_softmax(logits.reshape(-1, logits.shape[-1]), dim=-1)
    teacher = F.softmax(teacher_logits.reshape(-1, logits.shape[-1]), dim=-1)
    return F.kl_div(student, teacher, reduction="batchmean")
# ANCHOR_END: losses


def save(model, method):
    out = ROOT / f"models/smollm2_135m_{method}"
    out.mkdir(parents=True, exist_ok=True)
    state = {name: tensor.detach().float().cpu().contiguous() for name, tensor in model.state_dict().items()}
    # 출력 층은 임베딩 표와 같은 텐서다. 한 번만 적는다.
    state.pop("lm_head.weight", None)
    save_file(state, out / "model.safetensors")
    config = json.loads((BASE / "config.json").read_text(encoding="utf-8"))
    config["num_hidden_layers"] = model.config.num_hidden_layers
    config["torch_dtype"] = "float32"
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    shutil.copy(BASE / "tokenizer.json", out / "tokenizer.json")
    print(f"  -> {out.relative_to(ROOT)}")


def train(method, tokenizer, teacher, where):
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)
    if method.startswith("drop"):
        model = drop_blocks(model, LAST)
    if method.startswith("pick"):
        model = drop_blocks(model, PICKED)
    if method in ("drop", "pick"):
        save(model, method)
        return
    model.to(where).train()
    masks = magnitude_masks(model, PRUNED) if method.startswith("prune") else {}
    apply_masks(model, masks)
    fake = "qat" in method
    imitate = method.endswith("_kd")
    optimizer = torch.optim.AdamW(model.parameters(), lr=RATE, weight_decay=0.0)
    started, shown = time.perf_counter(), 0.0
    for step, ids in zip(range(1, STEPS + 1), batches(tokenizer, generator)):
        ids = ids.to(where)
        if fake:
            logits = functional_call(model, quantized(dict(model.named_parameters())), (ids,)).logits
        else:
            logits = model(ids).logits
        if imitate:
            with torch.no_grad():
                target = teacher(ids).logits
            loss = imitation_loss(logits, target)
        else:
            loss = next_token_loss(logits, ids)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        apply_masks(model, masks)
        shown += loss.item()
        if step % 50 == 0:
            print(f"  {step:>4}스텝: 손실 {shown / 50:.4f} ({time.perf_counter() - started:.0f}초)", flush=True)
            shown = 0.0
    save(model.eval(), method)


def vectors():
    """흉내 양자화가 엔진의 것과 같은지 견줄 값을 적는다.

    원래 모델의 행렬 하나의 앞 64행과, 그것을 여기에서 줄였다 되돌린 값이다.
    badak이 같은 행을 자기 코드로 줄였다 되돌려 비트를 견준다.
    """
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)
    weight = model.model.layers[0].mlp.down_proj.weight.detach()[:64].contiguous()
    out = ROOT / "models/smollm2_135m_qat"
    out.mkdir(parents=True, exist_ok=True)
    save_file({"weight": weight, "fake_q4_0": fake_q4_0(weight).contiguous()}, out / "fake_q4_0.safetensors")
    print(f"  -> {(out / 'fake_q4_0.safetensors').relative_to(ROOT)}")


def main():
    everything = ["ft", "qat", "qat_kd", "drop", "drop_ft", "drop_kd", "pick", "pick_ft", "pick_kd", "pick_qat_kd", "prune_kd"]
    methods = sys.argv[1:] or everything
    if methods == ["vectors"]:
        vectors()
        return
    where = device()
    print(f"장치 {where}, {STEPS}스텝, 한 스텝에 {BATCH} × {LENGTH}토큰, 학습률 {RATE}")
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    teacher = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32).to(where).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    for method in methods:
        print(method)
        train(method, tokenizer, teacher, where)


if __name__ == "__main__":
    main()
