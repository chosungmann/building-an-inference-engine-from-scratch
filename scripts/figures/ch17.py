"""17장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch17.py
출력: book/src/images/ch17/*.svg

측정값은 book/src/part3/output/ch17_shapes.txt와 bench/의 기록에서 읽는다.
다시 측정하면 그림도 다시 만들어야 한다.
"""

import re

from common import *  # noqa: F403

OUT = IMAGES / "ch17"  # noqa: F405
ROOT = IMAGES.parents[2]  # noqa: F405
OUTPUT = IMAGES.parent / "part3/output"  # noqa: F405
GRID = "#e5e7eb"


def lanes(x, y, labels, fill, stroke, w=58):
    out = []
    for i, label in enumerate(labels):
        out.append(f"<rect x='{x + i * w}' y='{y}' width='{w}' height='30' fill='{fill}' stroke='{stroke}'/>")
        out.append(text(x + i * w + w / 2, y + 20, label, INK, 12))
    return out


def fig_register():
    body = []
    body.append(text(20, 30, "명령 하나로 값 넷을: 벡터 레지스터", INK, 14, "start", "bold"))
    # 왼쪽: 값 하나씩
    body.append(text(20, 70, "값 하나씩 (명령 네 개)", INK, 13, "start", "bold"))
    for i in range(4):
        y = 86 + i * 38
        body += lanes(20, y, [f"s{i}"], BLUE_BG, BLUE)
        body.append(text(92, y + 20, "+=", MUTED, 13))
        body += lanes(110, y, [f"x{i}"], ORANGE_BG, ORANGE)
        body.append(text(180, y + 20, "×", MUTED, 13))
        body += lanes(196, y, [f"w{i}"], GREEN_BG, GREEN)
    # 오른쪽: 넷을 한꺼번에
    x0 = 370
    body.append(text(x0, 70, "넷을 한꺼번에 (명령 하나)", INK, 13, "start", "bold"))
    body += lanes(x0 + 60, 86, ["s0", "s1", "s2", "s3"], BLUE_BG, BLUE)
    body.append(text(x0 + 30, 106, "합", MUTED, 12))
    body.append(text(x0 + 176, 134, "+=", MUTED, 13))
    body += lanes(x0 + 60, 142, ["x0", "x1", "x2", "x3"], ORANGE_BG, ORANGE)
    body.append(text(x0 + 30, 162, "입력", MUTED, 12))
    body.append(text(x0 + 176, 190, "×", MUTED, 13))
    body += lanes(x0 + 60, 198, ["w0", "w1", "w2", "w3"], GREEN_BG, GREEN)
    body.append(text(x0 + 30, 218, "가중치", MUTED, 12))
    body.append(text(x0 + 310, 106, "레지스터 하나에", MUTED, 12, "start"))
    body.append(text(x0 + 310, 124, "f32 넷 (NEON)", MUTED, 12, "start"))
    body.append(text(x0 + 310, 162, "AVX2는 여덟", MUTED, 12, "start"))
    body.append(text(20, 262, "칸마다 따로 계산한다. 칸끼리는 서로 영향을 주지 않는다. 16장에서 합을 여러 개로 나눈 것이 바로 이 칸에 맞춘 것이다.", MUTED, 12, "start"))
    svg(OUT / "register.svg", 860, 282, body)


def fig_fma():
    body = []
    body.append(text(20, 30, "곱하고 더하기: 따로 할 때와 한 번에 할 때", INK, 14, "start", "bold"))

    def step(x, y, w, label, fill, stroke):
        return box(x, y, w, 40, label, None, fill, stroke)

    y = 60
    body.append(text(20, y + 25, "따로", INK, 13, "start", "bold"))
    body += step(90, y, 110, "x × w", GRAY_BG, MUTED)
    body.append(line(200, y + 20, 226, y + 20, INK, 1.5, arrow=True))
    body += step(230, y, 110, "반올림", ORANGE_BG, ORANGE)
    body.append(line(340, y + 20, 366, y + 20, INK, 1.5, arrow=True))
    body += step(370, y, 110, "+ 합", GRAY_BG, MUTED)
    body.append(line(480, y + 20, 506, y + 20, INK, 1.5, arrow=True))
    body += step(510, y, 110, "반올림", ORANGE_BG, ORANGE)
    body.append(text(640, y + 25, "연산 둘, 반올림 두 번", MUTED, 12, "start"))
    y = 130
    body.append(text(20, y + 25, "한 번에", INK, 13, "start", "bold"))
    body += step(90, y, 250, "x × w + 합  (정확한 값)", GRAY_BG, MUTED)
    body.append(line(340, y + 20, 506, y + 20, INK, 1.5, arrow=True))
    body += step(510, y, 110, "반올림", ORANGE_BG, ORANGE)
    body.append(text(640, y + 25, "연산 하나, 반올림 한 번", MUTED, 12, "start"))
    body.append(text(20, 206, "두 결과는 마지막 자리가 다를 수 있다. 한 번에 하는 쪽이 더 정확하고, 연산의 수가 절반이다.", MUTED, 12, "start"))
    svg(OUT / "fma.svg", 860, 226, body)


def read_shapes():
    rows = []
    for row in (OUTPUT / "ch17_shapes.txt").read_text(encoding="utf-8").splitlines():
        match = re.match(r"(\d × \d)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)$", row)
        if match:
            rows.append((match[1], int(match[2]), float(match[5])))
    return rows


def fig_shapes():
    body = []
    rows = read_shapes()
    x0, y0, w, h, top = 80, 60, 600, 230, 0.5
    body.append(text(x0, 30, "합을 담는 레지스터의 수와 속도 (곱셈당 사이클, 아래가 빠르다)", INK, 14, "start", "bold"))

    def pos(registers, cycles):
        return x0 + registers / 34 * w, y0 + h - cycles / top * h

    for value in (0, 0.1, 0.2, 0.3, 0.4, 0.5):
        _, yy = pos(0, value)
        body.append(line(x0, yy, x0 + w, yy, GRID, 1))
        body.append(text(x0 - 8, yy + 4, f"{value:g}", MUTED, 11, "end"))
    for registers in (2, 4, 8, 16, 24, 32):
        xx, _ = pos(registers, 0)
        body.append(text(xx, y0 + h + 18, str(registers), MUTED, 11))
    body.append(text(x0 + w / 2, y0 + h + 42, "합을 담는 레지스터의 수", MUTED, 12))
    # 레지스터 하나가 네 사이클에 한 번 일한다면: 1 / 레지스터 수.
    curve = " ".join("{:.1f},{:.1f}".format(*pos(r / 4, 4 / r)) for r in range(8, 65))
    body.append(f"<polyline points='{curve}' fill='none' stroke='{MUTED}' stroke-width='1.5' stroke-dasharray='6 4'/>")
    body.append(text(pos(5, 0.2)[0] + 6, pos(5, 0.2)[1] - 8, "1 ÷ 레지스터 수", MUTED, 11, "start"))
    seen = {}
    for shape, registers, cycles in rows:
        px, py = pos(registers, cycles)
        color = BLUE if registers <= 16 else ORANGE
        body.append(f"<circle cx='{px:.1f}' cy='{py:.1f}' r='5' fill='{color}' stroke='#ffffff' stroke-width='2'/>")
        # 같은 자리에 겹치는 모양은 이름을 이어 적는다.
        key = (registers, round(cycles, 1))
        seen.setdefault(key, []).append(shape)
    for (registers, cycles), shapes in seen.items():
        px, py = pos(registers, cycles)
        body.append(text(px + 9, py - 8, ", ".join(shapes), INK, 11, "start"))
    body.append(text(pos(24, 0.3)[0], pos(24, 0.3)[1], "레지스터가 모자라 메모리를 쓴다", ORANGE, 11))
    svg(OUT / "shapes.svg", 760, y0 + h + 58, body)


def fig_walks():
    body = []
    body.append(text(20, 30, "가중치 행렬을 걷는 세 가지 방법", INK, 14, "start", "bold"))

    def matrix(x, y, rows, picked, title, sub):
        out = [text(x, y - 26, title, INK, 13, "start", "bold"), text(x, y - 9, sub, MUTED, 11, "start")]
        for r in range(rows):
            hot = r in picked
            out.append(
                f"<rect x='{x}' y='{y + r * 14}' width='190' height='12' rx='2' "
                f"fill='{BLUE_BG if hot else GRAY_BG}' stroke='{BLUE if hot else GRID}'/>"
            )
        return out

    body += matrix(30, 90, 12, {1, 4, 7, 10}, "벡터 하나: 네 줄기", "행렬을 넷으로 나누어 한 행씩")
    for r in (1, 4, 7, 10):
        body.append(line(226, 96 + r * 14, 244, 96 + r * 14, BLUE, 1.5, arrow=True))
    body.append(text(30, 282, "줄기마다 메모리를 차례로 읽는다", MUTED, 11, "start"))
    body += matrix(300, 90, 12, {4, 5}, "벡터 몇 개: 두 행씩", "이웃한 두 행 × 벡터 둘")
    body.append(text(300, 282, "읽어 온 행을 벡터 둘이 함께 쓴다", MUTED, 11, "start"))
    body += matrix(570, 90, 12, {4, 5, 6, 7}, "벡터 많이: 네 행씩", "이웃한 네 행 × 벡터 둘")
    body.append(text(570, 282, "네 행이 캐시에 머무는 동안 모든 벡터가 지나간다", MUTED, 11, "start"))
    svg(OUT / "walks.svg", 880, 300, body)


def speed(path, test):
    text_ = (ROOT / path).read_text(encoding="utf-8")
    return float(re.search(rf"{test}\s+([\d.]+)", text_)[1])


def fig_result():
    body = []
    llama = (ROOT / "bench/ch13/llama_cpp_llama32.txt").read_text(encoding="utf-8").split("## ")[1]
    groups = [
        ("토큰 생성", "tg32", float(re.search(r"tg32 \|\s+([\d.]+)", llama)[1])),
        ("프롬프트 처리", "pp256", float(re.search(r"pp256 \|\s+([\d.]+)", llama)[1])),
    ]
    x0, w, top_value = 250, 480, 45.0
    body.append(text(20, 28, "Llama 3.2 1B, 스레드 1개: 1초에 처리하는 토큰", INK, 14, "start", "bold"))
    y = 60
    for title, test, theirs in groups:
        bars = [
            ("15장까지", speed("bench/baseline/badak_llama32_ch14.txt", test), MUTED),
            ("16장", speed("bench/baseline/badak_llama32_ch16.txt", test), "#93c5fd"),
            ("이 장", speed("bench/baseline/badak_llama32_ch17.txt", test), BLUE),
            ("llama.cpp (13.5절)", theirs, ORANGE),
        ]
        body.append(text(20, y + 16, title, INK, 13, "start", "bold"))
        for label, value, color in bars:
            width = value / top_value * w
            body.append(text(x0 - 12, y + 15, label, INK, 12, "end"))
            body.append(
                f"<path d='M{x0},{y + 2} h{max(width - 3, 1):.1f} a3,3 0 0 1 3,3 v12 a3,3 0 0 1 -3,3 h-{max(width - 3, 1):.1f} z' fill='{color}'/>"
            )
            body.append(text(x0 + width + 8, y + 16, f"{value:.2f}" if value < 10 else f"{value:.1f}", MUTED, 12, "start"))
            y += 28
        y += 18
    svg(OUT / "result.svg", 800, y + 4, body)


if __name__ == "__main__":
    for figure in (fig_register, fig_fma, fig_shapes, fig_walks, fig_result):
        figure()
    print(f"wrote {len(list(OUT.glob('*.svg')))} figures to {OUT}")
