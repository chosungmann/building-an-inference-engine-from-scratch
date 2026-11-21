"""49장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch49.py
출력: book/src/images/ch49/*.svg

bench/ch49/ 에 남긴 측정값을 그린다.
"""

from common import *  # noqa: F403

OUT = IMAGES / "ch49"  # noqa: F405
BENCH = IMAGES.parents[2] / "bench/ch49"  # noqa: F405
PURPLE = "#7c3aed"
PLACES = [
    ("native", "맥에서 그대로 (wgpu)", MUTED),  # noqa: F405
    ("chrome", "Chrome", BLUE),  # noqa: F405
    ("edge", "Edge", ORANGE),  # noqa: F405
    ("firefox", "Firefox", GREEN),  # noqa: F405
    ("safari", "Safari", PURPLE),
]


def rect(x, y, w, h, fill, stroke=None, rx=0, width=1, dash=None):
    edge = f" stroke='{stroke}' stroke-width='{width}'" if stroke else ""
    edge += f" stroke-dasharray='{dash}'" if dash else ""
    return f"<rect x='{x:.1f}' y='{y:.1f}' width='{w:.1f}' height='{h:.1f}' rx='{rx}' fill='{fill}'{edge}/>"


def lines(name):
    file = BENCH / name
    return [row.split() for row in file.read_text(encoding="utf-8").splitlines() if row.strip()] if file.exists() else []


def number(parts, key):
    value = parts[parts.index(key) + 1]
    return None if value == "none" else float(value)


def fig_paths():
    """엔진에서 GPU까지: 맥에서의 길과 브라우저에서의 길."""
    body = [text(20, 28, "같은 셰이더가 GPU에 닿는 두 길", INK, 14, "start", "bold")]  # noqa: F405
    # 함께 쓰는 부분.
    body.append(rect(20, 48, 860, 62, ORANGE_BG, ORANGE, 8, 1.5))  # noqa: F405
    body.append(text(450, 72, "엔진: 선형 층의 커널(WGSL), 선형 층을 GPU에 맡기는 코드", INK, 13, "middle", "bold"))  # noqa: F405
    body.append(text(450, 94, "23장의 것 그대로. 두 길이 함께 쓴다", MUTED, 12))  # noqa: F405
    rows = [
        ("맥에서 그대로", 150, [("Gpu (native.rs)", GRAY_BG, MUTED), ("wgpu", BLUE_BG, BLUE), ("Metal", GREEN_BG, GREEN), ("GPU", GREEN_BG, GREEN)]),  # noqa: F405
        ("브라우저에서", 236, [("Gpu (web.rs)", GRAY_BG, MUTED), ("gpu.js", GRAY_BG, MUTED), ("브라우저의 WebGPU", BLUE_BG, BLUE), ("GPU 프로세스", BLUE_BG, BLUE), ("Metal", GREEN_BG, GREEN), ("GPU", GREEN_BG, GREEN)]),  # noqa: F405
    ]
    for label, y, boxes in rows:
        body.append(text(20, y + 26, label, INK, 13, "start", "bold"))  # noqa: F405
        width = 118
        x = 130
        for index, (name, fill, stroke) in enumerate(boxes):
            body.append(rect(x, y, width, 42, fill, stroke, 6, 1.5))
            body.append(text(x + width / 2, y + 26, name, INK, 12))  # noqa: F405
            if index:
                body.append(line(x - 10, y + 21, x - 2, y + 21, INK, 1.5, True))  # noqa: F405
            x += width + 12
    notes = [
        (189, 208, "들여온 함수 12개", MUTED),  # noqa: F405
        (319, 208, "JavaScript", MUTED),  # noqa: F405
        (449, 208, "Dawn, wgpu, WebKit", MUTED),  # noqa: F405
        (579, 208, "프로세스를 건넌다", MUTED),  # noqa: F405
    ]
    for x, y, label, color in notes:
        body.append(text(x, y + 88, label, color, 11))  # noqa: F405
    body.append(line(189, 110, 189, 146, ORANGE, 1.5, True))  # noqa: F405
    body.append(line(114, 110, 114, 257, ORANGE, 1.5))  # noqa: F405
    body.append(line(114, 257, 127, 257, ORANGE, 1.5, True))  # noqa: F405
    svg(OUT / "paths.svg", 900, 310, body)  # noqa: F405


def fig_kernels():
    """같은 커널의 시간: 맥과 브라우저 넷."""
    shapes = [("64x1536x576", "벡터 64개, 576 → 1536"), ("256x2048x2048", "벡터 256개, 2048 → 2048"), ("256x8192x2048", "벡터 256개, 2048 → 8192")]
    data = {}
    for name, _, _ in PLACES:
        for parts in lines(f"{name}.txt"):
            if parts[0] == "gpu-linear" and parts[1] == "block":
                data[(name, parts[2])] = (number(parts, "wall-us"), number(parts, "gpu-us"))
    if not data:
        return
    body = [text(20, 28, "선형 층 하나를 GPU에서 (값 열여섯 커널): 보내고 기다린 시간", INK, 14, "start", "bold")]  # noqa: F405
    body.append(rect(20, 42, 14, 10, BLUE))  # noqa: F405
    body.append(text(40, 52, "그 가운데 GPU가 스스로 잰 계산 시간 (짙은 부분)", INK, 12, "start"))  # noqa: F405
    left, width = 190, 560
    # 가장 긴 것(Firefox의 0.1초)에 맞추면 나머지가 보이지 않는다. 축은 그
    # 밖의 것들에 맞추고, 넘치는 막대는 끝을 잘라 표시한다.
    longest = max(value[0] for (name, _), value in data.items() if name != "firefox")
    scale = width / (longest * 1.05)
    y = 70
    for shape, label in shapes:
        body.append(text(20, y + 12, label, INK, 12, "start", "bold"))  # noqa: F405
        y += 22
        for name, shown, color in PLACES:
            if (name, shape) not in data:
                continue
            wall, inside = data[(name, shape)]
            body.append(text(left - 10, y + 11, shown, INK, 12, "end"))  # noqa: F405
            shown_wall = min(wall * scale, width)
            body.append(f"<rect x='{left}' y='{y}' width='{shown_wall:.1f}' height='14' rx='2' fill='{color}' fill-opacity='0.35'/>")
            if inside is not None:
                body.append(rect(left, y, min(inside * scale, width), 14, color, None, 2))
            tail = f"{wall / 1000:.2f}ms" + (f" (계산 {inside / 1000:.2f}ms)" if inside is not None else "")
            if wall * scale > width:
                # 잘린 막대: 끝에 빗금 둘을 긋는다.
                for offset in (0, 6):
                    body.append(line(left + width - 14 + offset, y + 16, left + width - 8 + offset, y - 2, "#ffffff", 2))  # noqa: F405
                tail += ", 축 밖"
            body.append(text(left + shown_wall + 6, y + 11, tail, INK, 11, "start"))  # noqa: F405
            y += 18
        y += 10
    svg(OUT / "kernels.svg", 900, y + 4, body)  # noqa: F405


def fig_models():
    """모델의 속도: 프롬프트 처리."""
    models = [("smollm2_135m", "SmolLM2-135M"), ("smollm2_360m", "SmolLM2-360M"), ("llama32_1b", "Llama 3.2 1B")]
    modes = [("cpu", "CPU"), ("gpu-layer", "GPU, 층마다"), ("gpu-mlp", "GPU, MLP를 통째로")]
    data = {}
    for name, _, _ in PLACES:
        for parts in lines(f"{name}.txt"):
            if parts[0] == "llm-speed":
                data[(name, parts[1], parts[2])] = (number(parts, "prompt"), number(parts, "generation"))
    if not data:
        return
    body = [text(20, 28, "프롬프트 256토큰을 처리하는 속도 (토큰/초)", INK, 14, "start", "bold")]  # noqa: F405
    x = 20
    for _, shown, color in PLACES:
        body.append(rect(x, 42, 14, 10, color))
        body.append(text(x + 20, 52, shown, INK, 12, "start"))  # noqa: F405
        x += 170 if shown.startswith("맥") else 100
    panel, top, height = 280, 90, 220
    for column, (model, title) in enumerate(models):
        left = 50 + column * (panel + 10)
        values = [data[key][0] for key in data if key[1] == model]
        if not values:
            continue
        highest = max(values) * 1.12
        body.append(text(left + panel / 2 - 20, top - 12, title, INK, 13, "middle", "bold"))  # noqa: F405
        body.append(line(left, top + height, left + panel - 30, top + height, MUTED, 1))  # noqa: F405
        group = (panel - 40) / len(modes)
        for index, (mode, label) in enumerate(modes):
            x = left + 6 + index * group
            bar = (group - 12) / len(PLACES)
            for place, (name, _, color) in enumerate(PLACES):
                if (name, model, mode) not in data:
                    continue
                value = data[(name, model, mode)][0]
                tall = value / highest * height
                body.append(rect(x + place * bar, top + height - tall, bar - 2, tall, color, None, 1))
            best = max((data[(name, model, mode)][0] for name, _, _ in PLACES if (name, model, mode) in data), default=0)
            body.append(text(x + (group - 12) / 2, top + height - best / highest * height - 6, f"{best:.0f}", MUTED, 10))  # noqa: F405
            body.append(text(x + (group - 12) / 2, top + height + 16, label, INK, 10))  # noqa: F405
    svg(OUT / "models.svg", 900, top + height + 30, body)  # noqa: F405


if __name__ == "__main__":
    fig_paths()
    fig_kernels()
    fig_models()
