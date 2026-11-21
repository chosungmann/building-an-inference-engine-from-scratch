"""52장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch52.py
출력: book/src/images/ch52/*.svg
"""

import re

from common import *  # noqa: F403

OUT = IMAGES / "ch52"  # noqa: F405
BENCH = IMAGES.parents[2] / "bench/ch52"  # noqa: F405
LINES = (BENCH / "compile.txt").read_text(encoding="utf-8").splitlines()


def rect(x, y, w, h, fill, stroke=None, rx=6, width=1.5):
    edge = f" stroke='{stroke}' stroke-width='{width}'" if stroke else ""
    return f"<rect x='{x:.1f}' y='{y:.1f}' width='{w:.1f}' height='{h:.1f}' rx='{rx}' fill='{fill}'{edge}/>"


def mono(x, y, s, fill=INK, size=11):  # noqa: F405
    return (
        f"<text x='{x}' y='{y}' fill='{fill}' font-size='{size}' xml:space='preserve' "
        f"font-family='ui-monospace, SFMono-Regular, Menlo, monospace'>{s}</text>"
    )


def fig_layers():
    """세 층의 표현: 그래프, 반복문, 소스."""
    body = [text(20, 28, "그래프가 프로그램이 되기까지: 7장의 신경망", INK, 14, "start", "bold")]  # noqa: F405
    columns = [
        (20, 250, "1. 그래프", "무엇을 계산하는가", BLUE_BG, BLUE),  # noqa: F405
        (304, 282, "2. 반복문", "어떤 순서로 도는가", ORANGE_BG, ORANGE),  # noqa: F405
        (626, 300, "3. 소스", "기계가 읽을 글", GREEN_BG, GREEN),  # noqa: F405
    ]
    for x, w, title, sub, fill, stroke in columns:
        body.append(rect(x, 44, w, 40, fill, stroke))
        body.append(text(x + w / 2, 61, title, INK, 13, "middle", "bold"))  # noqa: F405
        body.append(text(x + w / 2, 77, sub, MUTED, 11))  # noqa: F405
    # 그래프
    nodes = [("FULLY_CONNECTED", "784 → 128"), ("RELU", "128"), ("FULLY_CONNECTED", "128 → 10")]
    for index, (name, shape) in enumerate(nodes):
        y = 104 + index * 58
        body.append(rect(60, y, 170, 38, "#ffffff", BLUE))  # noqa: F405
        body.append(text(145, y + 17, name, INK, 12, "middle", "bold"))  # noqa: F405
        body.append(text(145, y + 32, shape, MUTED, 11))  # noqa: F405
        if index:
            body.append(line(145, y - 20, 145, y - 2, BLUE, 1.5, True))  # noqa: F405
    # 반복문
    loops = [
        ("for r in 0..128", "out[r] = max(dot(x, W1[r]) + b1[r], 0)", 104, 96),
        ("for r in 0..10", "out[r] = dot(h, W2[r]) + b2[r]", 220, 38),
    ]
    for head, inner, y, h in loops:
        body.append(rect(310, y, 270, h, "#ffffff", ORANGE))  # noqa: F405
        body.append(mono(320, y + 17, head))
        body.append(mono(332, y + 32, inner, INK, 10))  # noqa: F405
    body.append(text(445, 162, "선형 층과 ReLU가", MUTED, 11))  # noqa: F405
    body.append(text(445, 178, "반복문 하나로 합쳐졌다", MUTED, 11))  # noqa: F405
    # 소스
    source = [
        "fn dot_784_16_1(x: &amp;[f32; 784], ...)",
        "    while i &lt; 784 {",
        "        sums[r][k] += x[i + k] * w[r][i + k];",
        "pub fn run(state, input, output) {",
        "    while r &lt; 128 {",
        "        state.v6[r] = (dots[0] + C2[r]).max(0.0);",
        "    while r &lt; 10 {",
        "        output[r] = dots[0] + C4[r];",
    ]
    body.append(rect(626, 104, 300, 154, "#ffffff", GREEN))  # noqa: F405
    for index, row in enumerate(source):
        body.append(mono(634, 122 + index * 17, row, INK, 9.5))  # noqa: F405
    for x, label in ((258, "낮추기"), (586, "적기")):
        body.append(line(x, 180, x + 36, 180, MUTED, 2, True))  # noqa: F405
        body.append(text(x + 18, 200, label, MUTED, 11))  # noqa: F405
    body.append(text(20, 290, "아래로 갈수록 형상이 숫자로 박히고, 기계에 가까워지고, 다른 모델에는 쓸 수 없게 된다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "layers.svg", 946, 304, body)  # noqa: F405


def shade(share):
    """0(빠름)에서 1(느림)까지를 옅은 파랑에서 짙은 파랑으로."""
    low, high = (239, 246, 255), (30, 64, 175)
    return "#" + "".join(f"{round(a + (b - a) * share):02x}" for a, b in zip(low, high))


def fig_kernels():
    """내적 커널의 생김새 스물네 가지."""
    import math

    body = [text(20, 28, "같은 선형 층, 내적을 적는 방법 스물네 가지 (스레드 하나, 한 번의 시간 µs)", INK, 14, "start", "bold")]  # noqa: F405
    for panel, shape in enumerate(("128x784", "1536x576")):
        left = 110 + panel * 430
        rows = [line for line in LINES if line.startswith(f"kernels {shape} 줄기")]
        hand = float(re.search(r"(\d+)ns", next(line for line in LINES if line.startswith(f"kernels {shape} 해석기"))).group(1))
        table = [[float(v) for v in re.findall(r"(\d+)ns", row)] for row in rows]
        flat = [v for row in table for v in row]
        fastest, slowest = min(flat), max(flat)
        rows_label, cols_label = shape.split("x")
        body.append(text(left, 56, f"{rows_label}행 × {cols_label}열", INK, 12, "start", "bold"))  # noqa: F405
        body.append(text(left + 160, 74, "가중치의 행을 몇 개씩 함께 읽는가", MUTED, 11))  # noqa: F405
        for c, together in enumerate((1, 2, 4, 8)):
            body.append(text(left + c * 80 + 40, 92, str(together), INK, 12))  # noqa: F405
        for r, accumulators in enumerate((1, 2, 4, 8, 16, 32)):
            y = 100 + r * 30
            body.append(text(left - 10, y + 19, str(accumulators), INK, 12, "end"))  # noqa: F405
            for c in range(4):
                value = table[r][c]
                share = math.log(value / fastest) / math.log(slowest / fastest)
                best = value == fastest
                body.append(rect(left + c * 80 + 1, y + 1, 78, 28, shade(share), ORANGE if best else None, 3, 2.5))  # noqa: F405
                ink = "#ffffff" if share > 0.55 else INK  # noqa: F405
                body.append(text(left + c * 80 + 40, y + 19, f"{value / 1000:.1f}", ink, 12, "middle", "bold" if best else "normal"))
        if panel == 0:
            body.append(text(46, 196, "더하는", MUTED, 11))  # noqa: F405
            body.append(text(46, 211, "줄기의 수", MUTED, 11))  # noqa: F405
        body.append(text(left, 302, f"손으로 짠 커널(17장): {hand / 1000:.1f}µs. 찾은 것 가운데 가장 빠른 것(주황 테두리)도 그 {fastest / hand:.2f}배.", MUTED, 11, "start"))  # noqa: F405
    svg(OUT / "kernels.svg", 960, 318, body)  # noqa: F405


def fig_glue():
    """해석기와 컴파일한 것: 세 가지 그래프."""
    groups = [
        ("glue_64", "작은 연산 64개, 값 64개"),
        ("glue_65536", "작은 연산 64개, 값 65,536개"),
        ("mlp", "7장의 신경망 (선형 층 둘)"),
    ]
    body = [text(20, 28, "해석기의 시간을 1로 놓았을 때 (스레드 하나, 짧을수록 빠르다)", INK, 14, "start", "bold")]  # noqa: F405
    y = 50
    for key, title in groups:
        rows = [line for line in LINES if line.startswith(key + " ")]
        base = float(re.search(r"한 번에 (\d+)ns", rows[0]).group(1))
        body.append(text(20, y + 14, title, INK, 12, "start", "bold"))  # noqa: F405
        unit = f"{base:.0f}ns" if base < 100_000 else f"{base / 1000:.0f}µs"
        body.append(text(20, y + 30, f"해석기 {unit}", MUTED, 11, "start"))  # noqa: F405
        for row in rows:
            value = float(re.search(r"한 번에 (\d+)ns", row).group(1))
            if " 해석기 마디 " in row:
                label, color = "해석기", MUTED  # noqa: F405
            else:
                label = re.search(r"컴파일\((.*?)\)", row).group(1)
                label = {"마디마다": "컴파일: 마디마다 반복문", "합쳐서": "컴파일: 반복문을 합쳐서", "합치고, 찾은 커널로": "컴파일: 합치고, 찾은 커널로"}[label]
                color = BLUE if "합쳐서" in label or "찾은" in label else ORANGE  # noqa: F405
            share = value / base
            body.append(text(420, y + 12, label, INK, 12, "end"))  # noqa: F405
            body.append(rect(430, y, max(1.0, share * 170), 16, color, None, 2))
            body.append(text(436 + share * 170, y + 12, f"{share:.2f}", INK, 11, "start"))  # noqa: F405
            y += 22
        y += 14
    body.append(line(600, 44, 600, y - 10, MUTED, 1, False, "3 3"))  # noqa: F405
    svg(OUT / "compare.svg", 900, y, body)  # noqa: F405


if __name__ == "__main__":
    fig_layers()
    fig_kernels()
    fig_glue()
