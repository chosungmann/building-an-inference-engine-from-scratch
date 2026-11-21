"""50장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch50.py
출력: book/src/images/ch50/*.svg

bench/ch50/ 에 남긴 측정값을 그린다.
"""

from common import *  # noqa: F403

OUT = IMAGES / "ch50"  # noqa: F405
BENCH = IMAGES.parents[2] / "bench/ch50"  # noqa: F405
PURPLE = "#7c3aed"
FAMILIES = {"badak": BLUE, "transformers": ORANGE, "onnxruntime": GREEN, "webllm": PURPLE}  # noqa: F405
NAMES = {"badak": "badak", "transformers": "Transformers.js", "onnxruntime": "ONNX Runtime Web", "webllm": "WebLLM"}


def rect(x, y, w, h, fill, rx=2, opacity=1.0):
    return f"<rect x='{x:.1f}' y='{y:.1f}' width='{w:.1f}' height='{h:.1f}' rx='{rx}' fill='{fill}' fill-opacity='{opacity}'/>"


def engines(name):
    """측정 파일에서 엔진의 줄들을 읽는다: (엔진, 설정, 값들)."""
    file = BENCH / name
    rows = []
    if not file.exists():
        return rows
    for line in file.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts or parts[0] != "engine":
            continue
        at = parts.index("load-ms")
        values = {parts[index]: parts[index + 1] for index in range(at, len(parts) - 1, 2) if parts[index] != "text"}
        rows.append((parts[1], " ".join(parts[2:at]), values))
    return rows


def describe(family, setting):
    """설정을 사람이 읽을 글로."""
    words = setting.replace("model_", "").replace(".onnx", "").split()
    where = {"cpu": "CPU", "wasm": "CPU", "gpu": "GPU(큰 일만)", "webgpu": "GPU"}[words[0]]
    rest = [word for word in words[1:] if word not in ("threads",)]
    threads = rest[-1] if "threads" in words else None
    kind = [word for word in rest if not word.isdigit() and word != "default"]
    label = where
    if kind:
        label += ", " + kind[0]
    if threads and where == "CPU":
        label += f", 스레드 {'기본' if threads == 'default' else threads}"
    return label


def bars(name, title, out, note):
    rows = engines(name)
    if not rows:
        return
    body = [text(20, 28, title, INK, 14, "start", "bold")]  # noqa: F405
    x = 20
    for family, color in FAMILIES.items():
        if any(row[0] == family for row in rows):
            body.append(rect(x, 42, 14, 10, color))
            body.append(text(x + 20, 52, NAMES[family], INK, 12, "start"))  # noqa: F405
            x += 150
    body.append(text(880, 52, note, MUTED, 11, "end"))  # noqa: F405
    left, width = 250, 400
    longest = max(float(values["again-per-s"]) for _, _, values in rows) * 1.05
    y = 72
    for family, setting, values in rows:
        rate = float(values["again-per-s"])
        body.append(text(left - 10, y + 11, describe(family, setting), INK, 12, "end"))  # noqa: F405
        body.append(rect(left, y, rate / longest * width, 14, FAMILIES[family]))
        body.append(text(left + rate / longest * width + 6, y + 11, f"{rate:.0f}토큰/초 · 긴 프롬프트 {float(values['long-first-ms']):.0f}ms", INK, 11, "start"))  # noqa: F405
        y += 20
    svg(OUT / out, 900, y + 8, body)  # noqa: F405


if __name__ == "__main__":
    bars("chrome_small.txt", "SmolLM2-135M이 초당 쓰는 토큰 (Chrome)", "small.svg", "막대: 초당 토큰. 오른쪽 숫자: 긴 프롬프트의 첫 토큰까지")
    bars("chrome_large.txt", "Llama 3.2 1B가 초당 쓰는 토큰 (Chrome)", "large.svg", "막대: 초당 토큰. 오른쪽 숫자: 긴 프롬프트의 첫 토큰까지")
