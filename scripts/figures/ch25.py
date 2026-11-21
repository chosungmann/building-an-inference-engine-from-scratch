"""25장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch25.py
출력: book/src/images/ch25/*.svg

장마다의 값은 bench/baseline/과 bench/ch13/, bench/ch25/에 적어 둔 측정에서
옮겨 적었다. 다시 측정하면 아래의 표를 고쳐야 한다.
"""

import math

from common import *  # noqa: F403

OUT = IMAGES / "ch25"  # noqa: F405
GRID = "#e5e7eb"
LIGHT_BLUE = "#93c5fd"
PURPLE = "#7c3aed"

# (이름, 프롬프트 처리, 토큰 생성). 토큰/초.
STEPS = [
    ("13장: 2부의 엔진", 2.13, 1.71),
    ("16장: 행렬 곱을 다시 짬", 27.8, 14.0),
    ("17장: 벡터 명령", 42.4, 18.1),
    ("18장: 스레드", 389, 50.5),
    ("20장: BF16 그대로", 395, 96.2),
    ("22장: 메모리", 424, 96.4),
    ("24장: 어텐션과 융합", 437, 97.4),
    ("20~24장: Q4_0", 653, 242),
    ("23~24장: GPU (f32)", 2751, 88),
]


def fig_ladder():
    x0, w, low, high = 250, 270, -0.3, 4.0
    xs = lambda value: (math.log10(value) - low) / (high - low) * w  # noqa: E731
    body = [text(20, 28, "3부에서 장마다 올라온 속도 (토큰/초, 스케일 하나가 열 배)", INK, 14, "start", "bold")]
    for panel, (title, column, llama) in enumerate([("프롬프트 처리", 1, 5273), ("토큰 생성", 2, 90)]):
        px = x0 + panel * 320
        body.append(text(px, 54, title, INK, 13, "start", "bold"))
        for power in range(0, 5):
            x = px + xs(10**power)
            body.append(line(x, 62, x, 62 + len(STEPS) * 26, GRID, 1))
            body.append(text(x, 62 + len(STEPS) * 26 + 14, f"{10 ** power:,}", MUTED, 10))
        x = px + xs(llama)
        body.append(line(x, 62, x, 62 + len(STEPS) * 26, ORANGE, 1.5, dash="5 4"))
        y = 64
        for index, step in enumerate(STEPS):
            if panel == 0:
                body.append(text(x0 - 12, y + 15, step[0], INK, 12, "end"))
            color = GREEN if index == 8 else (PURPLE if index == 7 else (MUTED if index == 0 else BLUE))
            width = xs(step[column])
            body.append(f"<rect x='{px}' y='{y + 3}' width='{width:.1f}' height='16' rx='3' fill='{color}'/>")
            label = f"{step[column]:,.2f}" if step[column] < 10 else f"{step[column]:,.0f}"
            body.append(text(px + width + 6, y + 16, label, MUTED, 11, "start"))
            y += 26
    y = 62 + len(STEPS) * 26 + 34
    body.append(line(20, y - 4, 48, y - 4, ORANGE, 1.5, dash="5 4"))
    body.append(text(54, y, "llama.cpp의 기본 설정 (f32, GPU). 13장에서 2,600배와 53배 떨어져 있던 선이다.", MUTED, 12, "start"))
    svg(OUT / "ladder.svg", 900, y + 16, body)


# (조건, badak 프롬프트/생성, llama.cpp 프롬프트/생성)
MATCHES = [
    ("16비트·f32 가중치, CPU", (437, 97.4), (200, 51.9)),
    ("Q8_0, CPU", (731, 170), (1226, 167)),
    ("Q4_0, CPU", (653, 242), (1348, 239)),
    ("f32 가중치, GPU", (2751, 88), (5273, 90)),
]


def fig_versus():
    body = [text(20, 28, "같은 조건에서 나란히 (토큰/초)", INK, 14, "start", "bold")]
    for index, (label, color) in enumerate([("badak", BLUE), ("llama.cpp", ORANGE)]):
        body.append(f"<rect x='{620 + index * 110}' y='18' width='14' height='11' rx='2' fill='{color}'/>")
        body.append(text(640 + index * 110, 28, label, MUTED, 11, "start"))
    for panel, (title, column, top) in enumerate([("프롬프트 처리", 0, 6000), ("토큰 생성", 1, 260)]):
        px, w = 230 + panel * 330, 240
        body.append(text(px, 56, title, INK, 13, "start", "bold"))
        y = 66
        for label, ours, theirs in MATCHES:
            if panel == 0:
                body.append(text(218, y + 22, label, INK, 12, "end"))
            for offset, (value, color) in enumerate(((ours[column], BLUE), (theirs[column], ORANGE))):
                width = value / top * w
                body.append(f"<rect x='{px}' y='{y + offset * 17}' width='{width:.1f}' height='14' rx='3' fill='{color}'/>")
                shown = f"{value:,.0f}" if value >= 100 else f"{value:.1f}"
                body.append(text(px + width + 6, y + offset * 17 + 12, shown, MUTED, 11, "start"))
            y += 48
    body.append(text(20, 66 + 4 * 48 + 12, "두 엔진 모두 스레드의 수를 바꿔 가며 잰 것 가운데 가장 빠른 값이다. GPU 줄의 badak은 선형 층만 GPU에 맡긴 것이다.", MUTED, 12, "start"))
    svg(OUT / "versus.svg", 900, 66 + 4 * 48 + 28, body)


CONTEXTS = [0, 256, 512, 1024, 2048, 4096]
LINES = [
    ("badak, BF16, CPU", [85.2, 96.2, 83.6, 82.1, 82.2, 76.2], BLUE, None),
    ("badak, Q4_0, CPU", [208.1, 238.0, 231.8, 219.7, 198.9, 161.2], BLUE, "5 4"),
    ("llama.cpp, F32, CPU", [40.1, 38.6, 33.7, 37.9, 32.9, 33.2], ORANGE, None),
    ("llama.cpp, Q4_0, CPU", [263.0, 235.9, 218.3, 194.4, 157.5, 119.4], ORANGE, "5 4"),
    ("llama.cpp, F32, GPU", [90.8, 90.0, 90.4, 90.7, 90.2, 88.6], GREEN, None),
    ("llama.cpp, Q4_0, GPU", [386.5, 386.1, 381.3, 378.6, 360.6, 338.8], GREEN, "5 4"),
]


def fig_context():
    x0, y0, w, h = 80, 60, 500, 260
    xs = lambda index: x0 + index / (len(CONTEXTS) - 1) * w  # noqa: E731
    ys = lambda value: y0 + h - value / 400 * h  # noqa: E731
    body = [text(20, 28, "문맥의 길이와 토큰 생성 속도: 두 엔진", INK, 14, "start", "bold")]
    for tick in range(0, 401, 100):
        body.append(line(x0, ys(tick), x0 + w, ys(tick), GRID, 1))
        body.append(text(x0 - 8, ys(tick) + 4, str(tick), MUTED, 11, "end"))
    for index, context in enumerate(CONTEXTS):
        body.append(text(xs(index), y0 + h + 16, f"{context:,}", MUTED, 11))
    body.append(text(x0 + w / 2, y0 + h + 36, "문맥에 이미 든 토큰의 수", MUTED, 12))
    body.append(text(20, y0 - 12, "토큰/초", MUTED, 12, "start"))
    for index, (label, values, color, dash) in enumerate(LINES):
        points = " ".join(f"{xs(i):.1f},{ys(v):.1f}" for i, v in enumerate(values))
        extra = f" stroke-dasharray='{dash}'" if dash else ""
        body.append(f"<polyline points='{points}' fill='none' stroke='{color}' stroke-width='2'{extra}/>")
        for i, v in enumerate(values):
            body.append(f"<circle cx='{xs(i):.1f}' cy='{ys(v):.1f}' r='3' fill='{color}'/>")
        y = y0 + 20 + index * 22
        body.append(line(x0 + w + 24, y, x0 + w + 52, y, color, 2, dash=dash))
        body.append(text(x0 + w + 58, y + 4, label, INK, 12, "start"))
    body.append(text(20, y0 + h + 62, "GPU는 문맥이 길어져도 거의 그대로다. CPU에서는 두 엔진 모두 떨어지고, llama.cpp 쪽이 더 가파르다.", MUTED, 12, "start"))
    svg(OUT / "context.svg", 900, y0 + h + 78, body)


if __name__ == "__main__":
    for figure in (fig_ladder, fig_versus, fig_context):
        figure()
    print(f"wrote {len(list(OUT.glob('*.svg')))} figures to {OUT}")
