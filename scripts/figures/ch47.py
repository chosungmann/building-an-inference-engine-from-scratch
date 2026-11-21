"""47장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch47.py
출력: book/src/images/ch47/*.svg

bench/ch47/ 에 남긴 측정값을 그린다.
"""

import statistics

from common import *  # noqa: F403

OUT = IMAGES / "ch47"  # noqa: F405
BENCH = IMAGES.parents[2] / "bench/ch47"  # noqa: F405
PURPLE, RED = "#7c3aed", "#dc2626"


def rect(x, y, w, h, fill, stroke=None, rx=0, width=1, dash=None):
    edge = f" stroke='{stroke}' stroke-width='{width}'" if stroke else ""
    edge += f" stroke-dasharray='{dash}'" if dash else ""
    return f"<rect x='{x:.1f}' y='{y:.1f}' width='{w:.1f}' height='{h:.1f}' rx='{rx}' fill='{fill}'{edge}/>"


def path(points, stroke, width=2):
    d = " ".join(f"{'M' if index == 0 else 'L'}{x:.1f},{y:.1f}" for index, (x, y) in enumerate(points))
    return f"<path d='{d}' fill='none' stroke='{stroke}' stroke-width='{width}'/>"


def lines(name):
    file = BENCH / name
    return [row.split() for row in file.read_text(encoding="utf-8").splitlines() if row.strip()] if file.exists() else []


def number(parts, key):
    return float(parts[parts.index(key) + 1])


def fig_budget():
    """풋프린트는 캐시와 나머지다."""
    runs = [parts for parts in lines("budget.txt") if parts[0] == "run"]
    if not runs:
        return
    names = {"smollm2_135m": "SmolLM2-135M", "smollm2_360m": "SmolLM2-360M", "llama32_1b": "Llama 3.2 1B", "smollm2_1_7b": "SmolLM2-1.7B"}
    kinds = {"stored": "파일 그대로", "q4_0": "Q4_0"}
    body = [text(20, 28, "LLM의 풋프린트: 가중치를 비추면 남는 것은 KV 캐시다", INK, 14, "start", "bold")]  # noqa: F405
    legend = [("KV 캐시", BLUE), ("나머지", MUTED), ("읽으면서 줄인 가중치", ORANGE)]  # noqa: F405
    x = 20
    for label, color in legend:
        body.append(rect(x, 42, 14, 10, color))
        body.append(text(x + 20, 52, label, INK, 12, "start"))  # noqa: F405
        x += 150
    body.append(text(880, 52, "파일의 가중치", MUTED, 11, "end"))  # noqa: F405
    longest = max(number(parts, "풋프린트") for parts in runs)
    scale = 440 / longest
    y = 70
    for parts in runs:
        model, weights, cache, tokens = parts[1], parts[2], parts[3], int(number(parts, "토큰"))
        footprint, kv, file = number(parts, "풋프린트"), number(parts, "캐시실제"), number(parts, "가중치")
        # 읽으면서 줄인 경우에만 가중치가 풋프린트에 들어간다.
        dirty = file if (weights == "q4_0" and model == "smollm2_135m") else 0.0
        rest = max(footprint - kv - dirty, 0)
        label = f"{names[model]}, {kinds[weights]}, 캐시 {cache}, {tokens:,}토큰"
        body.append(text(330, y + 14, label, INK, 11, "end"))  # noqa: F405
        x = 340
        for value, color in ((kv, BLUE), (dirty, ORANGE), (rest, MUTED)):  # noqa: F405
            if value * scale >= 0.5:
                body.append(rect(x, y, value * scale, 18, color))
                x += value * scale
        body.append(text(x + 6, y + 14, f"{footprint / 1e6:,.0f}MB", INK, 11, "start"))  # noqa: F405
        body.append(text(880, y + 14, f"{file / 1e6:,.0f}MB", MUTED, 11, "end"))  # noqa: F405
        y += 24
    body.append(text(20, y + 16, "오른쪽 끝은 가중치의 크기다. 파일을 메모리에 비추면 그것은 풋프린트에 들지 않는다(43.4절).", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "budget.svg", 900, y + 32, body)  # noqa: F405


def fig_hybrid():
    """넘긴 비율에 따른 정답률과 시간."""
    series = []
    for name, label, color in (("hybrid.txt", "기기: SmolLM2-360M", GREEN), ("hybrid_135m.txt", "기기: SmolLM2-135M", ORANGE)):  # noqa: F405
        rows = [parts for parts in lines(name) if parts[:2] == ["cascade", "평균로그"]]
        if rows:
            series.append((label, color, [(number(p, "넘긴비율"), number(p, "맞음") * 100, number(p, "평균시간")) for p in rows]))
    if not series:
        return
    body = [text(20, 28, "기기가 먼저 답하고 자신 없는 것만 넘긴다 (서버: SmolLM2-1.7B)", INK, 14, "start", "bold")]  # noqa: F405
    x = 20
    for label, color, _ in series:
        body.append(rect(x, 42, 14, 10, color))
        body.append(text(x + 20, 52, label, INK, 12, "start"))  # noqa: F405
        x += 200
    panels = [(70, "정답률 (%)", 1, 60, 100), (500, "질문 하나의 평균 시간 (ms)", 2, 0, 400)]
    for x0, title, column, low, high in panels:
        y0, w, h = 84, 340, 200
        body.append(text(x0 + w / 2, y0 - 10, title, INK, 12, "middle", "bold"))  # noqa: F405
        for tick in range(5):
            value = low + (high - low) * tick / 4
            y = y0 + h - h * tick / 4
            body.append(line(x0, y, x0 + w, y, GRAY_BG, 1))  # noqa: F405
            body.append(text(x0 - 6, y + 4, f"{value:.0f}", MUTED, 11, "end"))  # noqa: F405
        for percent in (0, 20, 40, 60, 80, 100):
            body.append(text(x0 + w * percent / 100, y0 + h + 16, f"{percent}%", MUTED, 11))  # noqa: F405
        for label, color, rows in series:
            points = [(x0 + w * row[0] / 100, y0 + h - h * (row[column] - low) / (high - low)) for row in rows]
            body.append(path(points, color, 2.5))
            for px, py in points:
                body.append(f"<circle cx='{px:.1f}' cy='{py:.1f}' r='3' fill='{color}'/>")
    body.append(text(20, 326, "가로축은 서버에 넘긴 질문의 비율이다. 0%가 기기만, 100%가 전부 넘긴 것이다. 왕복 100ms는 지어낸 값이다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "hybrid.svg", 900, 342, body)  # noqa: F405


def fig_engines():
    """엔진마다 토큰을 쓰는 속도."""
    rows = [parts for parts in lines("engines.txt") if parts[0] == "speed"]
    if not rows:
        return
    marks = [
        ("badak, 16비트, CPU", ("badak", "stored"), BLUE),  # noqa: F405
        ("badak, Q4_0, CPU", ("badak", "q4_0"), BLUE),  # noqa: F405
        ("llama.cpp, Q4_0, CPU", ("llama.cpp", "model_q4_0", "0"), GREEN),  # noqa: F405
        ("llama.cpp, Q4_0, GPU", ("llama.cpp", "model_q4_0", "99"), GREEN),  # noqa: F405
        ("LiteRT-LM, CPU", ("litert-lm", "cpu"), ORANGE),  # noqa: F405
        ("LiteRT-LM, GPU", ("litert-lm", "gpu"), ORANGE),  # noqa: F405
        ("ExecuTorch (올려진 파일), CPU", ("executorch", "f32"), PURPLE),
    ]
    body = [text(20, 28, "같은 모델(SmolLM2-135M)을 여러 엔진으로: 토큰을 쓰는 속도 (M4 Max)", INK, 14, "start", "bold")]  # noqa: F405
    values = []
    for label, key, color in marks:
        if key[0] == "llama.cpp":
            found = [p for p in rows if p[1] == key[0] and p[3] == key[1] and p[5] == key[2]]
        else:
            found = [p for p in rows if p[1] == key[0] and p[3] == key[1]]
        values.append((label, color, statistics.median(number(p, "쓰기") for p in found)))
    scale = 480 / max(value for *_, value in values)
    y = 50
    for label, color, value in values:
        body.append(text(250, y + 15, label, INK, 12, "end"))  # noqa: F405
        body.append(rect(260, y, max(value * scale, 2), 20, color, None, 3))
        body.append(text(260 + value * scale + 8, y + 15, f"{value:.0f}토큰/초", INK, 12, "start"))  # noqa: F405
        y += 28
    body.append(text(20, y + 14, "줄마다 가중치의 형식과 파일을 만든 방법이 다르다. 엔진의 순위가 아니다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "engines.svg", 880, y + 30, body)  # noqa: F405


if __name__ == "__main__":
    fig_budget()
    fig_hybrid()
    fig_engines()
