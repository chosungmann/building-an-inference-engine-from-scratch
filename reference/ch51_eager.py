"""51장: PyTorch를 그대로 쓰면 무엇을 치르는가.

실행:
    sh scripts/download_smollm2.sh
    pip install -r reference/requirements_ch51.txt
    python reference/ch51_eager.py [size|op|model|profile ...]

적는 줄은 scripts/bench_ch51.sh 가 bench/ch51/ 에 모은다.

    size     설치된 꾸러미의 크기
    op       아주 작은 연산 하나를 부르는 데 드는 시간
    model    SmolLM2-135M으로 프롬프트를 처리하고 토큰을 쓰는 속도
    profile  토큰 하나를 쓸 때 부르는 연산의 수와 시간
    compile  torch.compile로 컴파일한 뒤의 속도와, 컴파일에 드는 시간
    once     모델을 열어 글 한 줄을 이어 쓰고 끝낸다. 시작부터 끝까지의
             시간과 메모리를 밖에서 잴 때 쓴다(/usr/bin/time -l)
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/smollm2_135m"
PROMPT_TOKENS, NEW_TOKENS = 256, 32


def megabytes(path):
    return sum(file.stat().st_size for file in Path(path).rglob("*") if file.is_file()) / 1e6


def size():
    """설치된 꾸러미들이 디스크에서 차지하는 크기."""
    import executorch
    import numpy
    import onnxruntime
    import torch
    import transformers

    for module in (torch, transformers, numpy, onnxruntime, executorch):
        folder = Path(list(module.__path__)[0])
        print(f"size {module.__name__} {megabytes(folder):.0f}MB")
    # 가장 큰 파일 셋. PyTorch의 몸통은 C++로 쓴 라이브러리 하나다.
    files = sorted(Path(torch.__file__).parent.rglob("*"), key=lambda file: file.stat().st_size if file.is_file() else 0)
    for file in files[-3:][::-1]:
        print(f"size-file torch/{file.relative_to(Path(torch.__file__).parent)} {file.stat().st_size / 1e6:.0f}MB")


# ANCHOR: op
def op():
    """값 하나짜리 텐서 둘을 더하는 데 드는 시간.

    계산은 덧셈 한 번이다. 걸리는 시간은 거의 모두 "덧셈을 부르는 데" 든다.
    """
    import numpy
    import torch

    count = 200_000
    a, b = torch.ones(1), torch.ones(1)
    for label, enter in (("grad", torch.enable_grad), ("no_grad", torch.no_grad), ("inference_mode", torch.inference_mode)):
        with enter():
            started = time.perf_counter()
            for _ in range(count):
                a + b
            print(f"op torch-add-1 {label} {(time.perf_counter() - started) / count * 1e9:.0f}ns")
    x, y = numpy.ones(1, dtype=numpy.float32), numpy.ones(1, dtype=numpy.float32)
    started = time.perf_counter()
    for _ in range(count):
        x + y
    print(f"op numpy-add-1 - {(time.perf_counter() - started) / count * 1e9:.0f}ns")
    p, q = 1.0, 1.0
    started = time.perf_counter()
    for _ in range(count):
        p + q
    print(f"op python-add - {(time.perf_counter() - started) / count * 1e9:.0f}ns")
    # 값이 많아지면 부르는 값은 묻힌다.
    for length in (576, 576 * 576):
        a, b = torch.ones(length), torch.ones(length)
        repeats = max(200, 2_000_000 // length)
        with torch.inference_mode():
            started = time.perf_counter()
            for _ in range(repeats):
                a + b
            print(f"op torch-add-{length} inference_mode {(time.perf_counter() - started) / repeats * 1e9:.0f}ns")
# ANCHOR_END: op


def load():
    import torch
    from transformers import AutoModelForCausalLM

    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    return model, time.perf_counter() - started


def tokens(vocabulary):
    """23장의 측정과 같은 꼴: 아무 토큰이나 고정된 씨앗으로 뽑는다."""
    import torch

    generator = torch.Generator().manual_seed(23)
    return torch.randint(0, vocabulary, (1, PROMPT_TOKENS + NEW_TOKENS), generator=generator)


# ANCHOR: model
def model():
    """프롬프트 256토큰을 한꺼번에 처리하고, 이어서 토큰 32개를 하나씩."""
    import statistics

    import torch

    network, seconds = load()
    ids = tokens(network.config.vocab_size)
    print(f"model load-s {seconds:.2f} threads {torch.get_num_threads()} dtype float32")
    for threads in (torch.get_num_threads(), 8, 1):
        torch.set_num_threads(threads)
        prompt, generation = [], []
        with torch.inference_mode():
            for repetition in range(5):
                time.sleep(3)
                started = time.perf_counter()
                out = network(ids[:, :PROMPT_TOKENS], use_cache=True)
                took = time.perf_counter() - started
                cache = out.past_key_values
                started = time.perf_counter()
                for index in range(PROMPT_TOKENS, PROMPT_TOKENS + NEW_TOKENS):
                    out = network(ids[:, index : index + 1], past_key_values=cache, use_cache=True)
                    cache = out.past_key_values
                each = time.perf_counter() - started
                if repetition:
                    prompt.append(PROMPT_TOKENS / took)
                    generation.append(NEW_TOKENS / each)
        print(f"model speed threads {threads} prompt {statistics.median(prompt):.0f} generation {statistics.median(generation):.1f}")
# ANCHOR_END: model


def profile():
    """토큰 하나를 쓰는 동안 PyTorch가 부르는 연산을 센다."""
    import torch
    from torch.profiler import ProfilerActivity
    from torch.profiler import profile as record

    network, _ = load()
    ids = tokens(network.config.vocab_size)
    with torch.inference_mode():
        cache = network(ids[:, :PROMPT_TOKENS], use_cache=True).past_key_values
        # 준비 운동.
        for index in range(PROMPT_TOKENS, PROMPT_TOKENS + 4):
            cache = network(ids[:, index : index + 1], past_key_values=cache, use_cache=True).past_key_values
        started = time.perf_counter()
        steps = 16
        for index in range(PROMPT_TOKENS + 4, PROMPT_TOKENS + 4 + steps):
            cache = network(ids[:, index : index + 1], past_key_values=cache, use_cache=True).past_key_values
        plain = (time.perf_counter() - started) / steps
        with record(activities=[ProfilerActivity.CPU]) as recorded:
            network(ids[:, -1:], past_key_values=cache, use_cache=True)
    events = [event for event in recorded.key_averages() if event.key.startswith("aten::")]
    calls = sum(event.count for event in events)
    print(f"profile token-ms {plain * 1e3:.2f} aten-calls {calls} kinds {len(events)}")
    total = sum(event.self_cpu_time_total for event in events)
    for event in sorted(events, key=lambda event: -event.self_cpu_time_total)[:8]:
        print(f"profile-op {event.key} calls {event.count} share {event.self_cpu_time_total / total * 100:.1f}%")
    linear = sum(event.self_cpu_time_total for event in events if event.key in ("aten::addmm", "aten::linear", "aten::matmul", "aten::mm"))
    print(f"profile-matmul share {linear / total * 100:.1f}%")


# ANCHOR: compile
def compiled():
    """같은 측정을 torch.compile을 거쳐서 한다.

    torch.compile은 Python 안에 머문 채로, 모델이 처음 불릴 때 지나가는
    연산들을 받아 적어 기계어로 옮겨 둔다. 입력의 모양이 달라지면 다시 옮긴다.
    """
    import statistics

    import torch

    network, _ = load()
    ids = tokens(network.config.vocab_size)
    fast = torch.compile(network)
    prompt, generation, first = [], [], []
    with torch.inference_mode():
        for repetition in range(5):
            time.sleep(3)
            started = time.perf_counter()
            out = fast(ids[:, :PROMPT_TOKENS], use_cache=True)
            took = time.perf_counter() - started
            cache = out.past_key_values
            started = time.perf_counter()
            for index in range(PROMPT_TOKENS, PROMPT_TOKENS + NEW_TOKENS):
                out = fast(ids[:, index : index + 1], past_key_values=cache, use_cache=True)
                cache = out.past_key_values
            each = time.perf_counter() - started
            if repetition:
                prompt.append(PROMPT_TOKENS / took)
                generation.append(NEW_TOKENS / each)
            else:
                first = [took, each]
    print(f"compile first-prompt-s {first[0]:.1f} first-generation-s {first[1]:.1f}")
    print(f"compile speed threads {torch.get_num_threads()} prompt {statistics.median(prompt):.0f} generation {statistics.median(generation):.1f}")
# ANCHOR_END: compile


def once():
    """badak의 `ch11_generate smollm2_135m "The capital of France is" 40`과 같은 일."""
    import torch
    from transformers import AutoTokenizer

    network, _ = load()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    ids = tokenizer("The capital of France is", return_tensors="pt").input_ids
    with torch.inference_mode():
        written = network.generate(ids, max_new_tokens=40, do_sample=False)
    print("once " + tokenizer.decode(written[0, ids.shape[1] :]).replace("\n", " "))


if __name__ == "__main__":
    chosen = sys.argv[1:] or ["size", "op", "model", "profile"]
    for name in chosen:
        {"size": size, "op": op, "model": model, "profile": profile, "compile": compiled, "once": once}[name]()
