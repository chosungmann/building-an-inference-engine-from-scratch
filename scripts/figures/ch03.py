"""3장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch03.py
출력: book/src/images/ch03/*.svg
"""

import math

from common import *  # noqa: F403

OUT = IMAGES / "ch03"  # noqa: F405

# 여러 계열을 구분하는 색. 색각 이상에서도 구분되는지 검증한 순서다.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
GRID = "#e5e7eb"


def circle(x, y, r, fill, stroke, label=None, size=14):
    out = [f"<circle cx='{x}' cy='{y}' r='{r}' fill='{fill}' stroke='{stroke}' stroke-width='1.5'/>"]
    if label:
        out.append(text(x, y + 5, label, size=size))
    return out


def fig_neuron():
    body = []
    xs, ws = ["x₁", "x₂", "x₃"], ["w₁", "w₂", "w₃"]
    for i in range(3):
        y = 60 + i * 70
        body += circle(60, y, 22, ORANGE_BG, ORANGE, xs[i])
        body.append(line(84, y, 248, 130 + (y - 130) * 0.25, MUTED, 1.5, arrow=True))
        body.append(text(160, y + (130 - y) * 0.4 - 8, ws[i], BLUE, 14, weight="bold"))
    body += box(252, 100, 130, 60, "곱해서 더한다", "w·x + b", BLUE_BG, BLUE)
    body.append(text(317, 200, "편향 b", BLUE, 13))
    body.append(line(317, 184, 317, 164, BLUE, 1.5, arrow=True))
    body.append(line(382, 130, 428, 130, MUTED, 1.5, arrow=True))
    body += box(432, 100, 120, 60, "활성화 함수", "꺾거나 누른다", GREEN_BG, GREEN)
    body.append(line(552, 130, 598, 130, MUTED, 1.5, arrow=True))
    body += circle(624, 130, 22, GRAY_BG, MUTED, "y")
    body.append(text(60, 250, "입력", MUTED, 13))
    body.append(text(317, 250, "내적과 편향", MUTED, 13))
    body.append(text(492, 250, "비선형", MUTED, 13))
    body.append(text(624, 250, "출력", MUTED, 13))
    svg(OUT / "neuron.svg", 680, 270, body)


def fig_network():
    body = []
    columns = [
        (90, 3, ORANGE_BG, ORANGE, "입력 (3)"),
        (330, 4, BLUE_BG, BLUE, "은닉층 (4)"),
        (570, 2, GREEN_BG, GREEN, "출력 (2)"),
    ]
    points = []
    for x, count, _, _, _ in columns:
        top = 170 - (count - 1) * 30
        points.append([(x, top + i * 60) for i in range(count)])
    for a, b in zip(points, points[1:]):
        for x1, y1 in a:
            for x2, y2 in b:
                body.append(line(x1 + 18, y1, x2 - 18, y2, "#cbd5e1", 1))
    for (x, _, fill, stroke, label), pts in zip(columns, points):
        for px, py in pts:
            body += circle(px, py, 18, fill, stroke)
        body.append(text(x, 40, label, INK, 14, weight="bold"))
    body.append(text(210, 320, "W₁: 4 × 3 행렬, b₁: 길이 4", BLUE, 13))
    body.append(text(210, 340, "선 12개 + 편향 4개 = 파라미터 16개", MUTED, 12))
    body.append(text(450, 320, "W₂: 2 × 4 행렬, b₂: 길이 2", GREEN, 13))
    body.append(text(450, 340, "선 8개 + 편향 2개 = 파라미터 10개", MUTED, 12))
    svg(OUT / "network.svg", 660, 360, body)


def fig_activations():
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    functions = [
        ("ReLU", lambda x: max(x, 0.0)),
        ("시그모이드", sigmoid),
        ("tanh", math.tanh),
        ("SiLU", lambda x: x * sigmoid(x)),
    ]
    left, top, width, height = 60, 30, 470, 300
    x_min, x_max, y_min, y_max = -5.0, 5.0, -1.5, 5.0

    def px(x):
        return left + (x - x_min) / (x_max - x_min) * width

    def py(y):
        return top + (y_max - y) / (y_max - y_min) * height

    body = []
    for gx in range(-5, 6):
        body.append(line(px(gx), top, px(gx), top + height, GRID, 1))
        body.append(text(px(gx), top + height + 18, str(gx).replace("-", "−"), MUTED, 12))
    for gy in range(-1, 6):
        body.append(line(left, py(gy), left + width, py(gy), GRID, 1))
        body.append(text(left - 10, py(gy) + 4, str(gy).replace("-", "−"), MUTED, 12, "end"))
    body.append(line(left, py(0), left + width, py(0), MUTED, 1))
    body.append(line(px(0), top, px(0), top + height, MUTED, 1))
    body.append(text(left + width / 2, top + height + 40, "입력 x", MUTED, 13))

    # 오른쪽 끝에서 각 선이 닿는 높이 근처에 이름을 적는다.
    label_y = {"ReLU": 4.7, "SiLU": 4.2, "시그모이드": 1.35, "tanh": 0.75}
    for (name, f), color in zip(functions, SERIES):
        xs = [x_min + i * (x_max - x_min) / 400 for i in range(401)]
        path = " ".join(
            f"{'M' if i == 0 else 'L'}{px(x):.1f},{py(f(x)):.1f}" for i, x in enumerate(xs)
        )
        body.append(
            f"<path d='{path}' fill='none' stroke='{color}' stroke-width='2' stroke-linejoin='round'/>"
        )
        # 글자는 본문 색으로 쓰고, 옆의 선 조각이 어느 계열인지 알린다.
        y = py(label_y[name])
        body.append(line(left + width + 10, y - 4, left + width + 34, y - 4, color, 2))
        body.append(text(left + width + 40, y, name, INK, 13, "start"))
    svg(OUT / "activations.svg", 680, 385, body)


def fig_xor():
    body = []

    def plane(ox, oy, unit, x_label, y_label, title, x_ticks, y_ticks):
        w, h = unit * (len(x_ticks) - 1), unit * (len(y_ticks) - 1)
        out = [text(ox + w / 2, oy - h - 42, title, INK, 14, weight="bold")]
        out.append(line(ox - 15, oy, ox + w + 30, oy, MUTED, 1.5, arrow=True))
        out.append(line(ox, oy + 15, ox, oy - h - 28, MUTED, 1.5, arrow=True))
        for i, tick in enumerate(x_ticks):
            out.append(text(ox + i * unit, oy + 22, tick, MUTED, 12))
        for i, tick in enumerate(y_ticks):
            if i:
                out.append(text(ox - 12, oy - i * unit + 4, tick, MUTED, 12, "end"))
        out.append(text(ox + w + 38, oy + 22, x_label, MUTED, 13))
        out.append(text(ox - 12, oy - h - 30, y_label, MUTED, 13, "end"))
        return out

    def point(x, y, positive, label, dx=14, dy=-12):
        fill, stroke = (BLUE, BLUE) if positive else ("#ffffff", ORANGE)
        return [
            f"<circle cx='{x}' cy='{y}' r='9' fill='{fill}' stroke='{stroke}' stroke-width='3'/>",
            text(x + dx, y + dy, label, INK, 12, "start"),
        ]

    # 왼쪽: 입력 평면
    ox, oy, u = 80, 270, 130
    body += plane(ox, oy, u, "x₁", "x₂", "입력 평면", ["0", "1"], ["0", "1"])
    body += point(ox, oy, False, "(0, 0)")
    body += point(ox + u, oy, True, "(1, 0)")
    body += point(ox, oy - u, True, "(0, 1)")
    body += point(ox + u, oy - u, False, "(1, 1)")
    body.append(text(ox + u / 2, oy + 52, "어떤 직선으로도 두 종류를 가를 수 없다", MUTED, 12))

    body.append(line(275, 190, 345, 190, MUTED, 2, arrow=True))
    body.append(text(310, 175, "은닉층", MUTED, 12))

    # 오른쪽: 은닉층이 만든 평면
    ox2, oy2, u2 = 410, 270, 100
    body += plane(ox2, oy2, u2, "h₁", "h₂", "은닉층을 지난 뒤", ["0", "1", "2"], ["0", "1"])
    # 경계선 h₁ − 2h₂ = 0.5 : (0.5, 0)과 (2.5, 1)을 지난다.
    body.append(line(ox2 + 0.3 * u2, oy2 + 0.1 * u2, ox2 + 2.7 * u2, oy2 - 1.1 * u2, GREEN, 2, dash="6 4"))
    body.append(text(ox2 + 2.2 * u2, oy2 - 0.45 * u2, "h₁ − 2h₂ = 0.5", GREEN, 12, "start"))
    body += point(ox2, oy2, False, "(0, 0)", -14, -14)
    body += point(ox2 + u2, oy2, True, "(1, 0) 두 점이 겹친다", -20, 44)
    body += point(ox2 + 2 * u2, oy2 - u2, False, "(2, 1)", -52, -14)
    body.append(text(ox2 + u2, oy2 + 72, "직선 하나로 갈린다", MUTED, 12))

    # 범례
    body.append(f"<circle cx='250' cy='30' r='7' fill='{BLUE}' stroke='{BLUE}' stroke-width='3'/>")
    body.append(text(264, 35, "XOR = 1", INK, 13, "start"))
    body.append(f"<circle cx='360' cy='30' r='7' fill='#ffffff' stroke='{ORANGE}' stroke-width='3'/>")
    body.append(text(374, 35, "XOR = 0", INK, 13, "start"))
    svg(OUT / "xor.svg", 760, 360, body)


def fig_training_vs_inference():
    body = []

    def row(y, title, backward):
        out = [text(30, y + 35, title, INK, 15, "start", "bold")]
        xs = [150, 300, 450]
        out.append(text(110, y + 35, "입력", MUTED, 13))
        for i, x in enumerate(xs):
            out += box(x, y + 8, 100, 46, f"층 {i + 1}", None, BLUE_BG, BLUE)
        out.append(line(128, y + 31, 146, y + 31, INK, 2, arrow=True))
        for x in xs[:-1]:
            out.append(line(x + 100, y + 31, x + 146, y + 31, INK, 2, arrow=True))
        out.append(line(550, y + 31, 596, y + 31, INK, 2, arrow=True))
        if backward:
            out += box(600, y + 8, 110, 46, "손실", "정답과 비교", ORANGE_BG, ORANGE)
            out.append(line(655, y + 54, 655, y + 78, ORANGE, 2))
            out.append(line(655, y + 78, 200, y + 78, ORANGE, 2, dash="5 4"))
            for x in xs:
                out.append(line(x + 50, y + 78, x + 50, y + 58, ORANGE, 2, arrow=True))
            out.append(
                text(425, y + 102, "거꾸로 되짚으며 각 층의 가중치를 얼마나 고칠지 구한다", ORANGE, 12)
            )
        else:
            out.append(text(628, y + 35, "출력", MUTED, 13))
        return out

    body += row(20, "추론", False)
    body.append(text(380, 100, "앞으로 한 번 계산하면 끝난다. 가중치는 읽기만 한다.", MUTED, 12))
    body.append(line(30, 125, 730, 125, GRID, 1))
    body += row(140, "학습", True)
    svg(OUT / "training_vs_inference.svg", 760, 265, body)


if __name__ == "__main__":
    for figure in (fig_neuron, fig_network, fig_activations, fig_xor, fig_training_vs_inference):
        figure()
    print(f"wrote {len(list(OUT.glob('*.svg')))} figures to {OUT}")
