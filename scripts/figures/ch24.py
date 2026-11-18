"""24장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch24.py
출력: book/src/images/ch24/*.svg

측정값은 book/src/part3/output/ch24_*.txt에서 읽는다. 다시 측정하면 그림도
다시 만들어야 한다.
"""

import math

from common import *  # noqa: F403

OUT = IMAGES / "ch24"  # noqa: F405
OUTPUT = IMAGES.parent / "part3/output"  # noqa: F405
GRID = "#e5e7eb"
LIGHT_BLUE = "#93c5fd"
PURPLE = "#7c3aed"


def read(name):
    return (OUTPUT / name).read_text(encoding="utf-8")


def read_context(name):
    rows = []
    for row in read(name).splitlines():
        parts = row.replace("%", "").split()
        if len(parts) == 6 and parts[0].isdigit():
            rows.append([float(v) for v in parts])
    return rows


def fig_layout():
    body = [text(20, 28, "KV 캐시에 키를 놓는 두 가지 방법 (헤드 둘, 토큰 넷으로 줄여 그렸다)", INK, 14, "start", "bold")]
    w, h = 84, 34
    body.append(text(20, 62, "22장까지: 토큰마다, 그 안에서 헤드마다", INK, 13, "start", "bold"))
    for index in range(8):
        token, head = index // 2, index % 2
        fill, stroke = (BLUE_BG, BLUE) if head == 0 else (ORANGE_BG, ORANGE)
        x = 30 + index * (w + 4)
        body.append(f"<rect x='{x}' y='74' width='{w}' height='{h}' rx='4' fill='{fill}' stroke='{stroke}'/>")
        body.append(text(x + w / 2, 96, f"토큰 {token}, 헤드 {head}", INK, 11))
    body.append(text(30, 128, "헤드 0의 키를 차례로 읽으려면 하나 걸러 하나씩 건너뛰어야 한다.", MUTED, 12, "start"))
    body.append(text(20, 168, "이 장: 헤드마다 따로, 그 안에서 토큰마다", INK, 13, "start", "bold"))
    for head in range(2):
        fill, stroke = (BLUE_BG, BLUE) if head == 0 else (ORANGE_BG, ORANGE)
        for token in range(4):
            x = 30 + head * (4 * (w + 4) + 24) + token * (w + 4)
            body.append(f"<rect x='{x}' y='180' width='{w}' height='{h}' rx='4' fill='{fill}' stroke='{stroke}'/>")
            body.append(text(x + w / 2, 202, f"토큰 {token}, 헤드 {head}", INK, 11))
    body.append(text(30, 234, "헤드 하나의 키들이 나란히 놓인다. ‘행이 키인 행렬’이 되어 17장의 커널에 그대로 넣을 수 있다.", MUTED, 12, "start"))
    svg(OUT / "layout.svg", 880, 250, body)


def fig_context():
    series = [
        ("23장까지, f32 캐시", read_context("ch24_context_before.txt"), MUTED, None),
        ("23장까지, 16비트 캐시", read_context("ch24_context_f16_before.txt"), MUTED, "5 4"),
        ("이 장, f32 캐시", read_context("ch24_context_after.txt"), BLUE, None),
        ("이 장, 16비트 캐시", read_context("ch24_context_f16_after.txt"), BLUE, "5 4"),
    ]
    x0, y0, w, h = 80, 60, 520, 240
    contexts = [r[0] for r in series[0][1]]
    xs = lambda index: x0 + index / (len(contexts) - 1) * w  # noqa: E731
    top = 100.0
    ys = lambda value: y0 + h - value / top * h  # noqa: E731
    body = [text(20, 28, "문맥의 길이와 토큰 생성 속도", INK, 14, "start", "bold")]
    for tick in range(0, 101, 20):
        body.append(line(x0, ys(tick), x0 + w, ys(tick), GRID, 1))
        body.append(text(x0 - 8, ys(tick) + 4, str(tick), MUTED, 11, "end"))
    for index, context in enumerate(contexts):
        body.append(text(xs(index), y0 + h + 16, f"{int(context):,}", MUTED, 11))
    body.append(text(x0 + w / 2, y0 + h + 36, "문맥에 이미 든 토큰의 수", MUTED, 12))
    body.append(text(20, y0 - 12, "토큰/초", MUTED, 12, "start"))
    for index, (label, rows, color, dash) in enumerate(series):
        points = " ".join(f"{xs(i):.1f},{ys(r[3]):.1f}" for i, r in enumerate(rows))
        extra = f" stroke-dasharray='{dash}'" if dash else ""
        body.append(f"<polyline points='{points}' fill='none' stroke='{color}' stroke-width='2'{extra}/>")
        for i, r in enumerate(rows):
            body.append(f"<circle cx='{xs(i):.1f}' cy='{ys(r[3]):.1f}' r='3' fill='{color}'/>")
        y = y0 + 20 + index * 22
        body.append(line(x0 + w + 24, y, x0 + w + 52, y, color, 2, dash=dash))
        body.append(text(x0 + w + 58, y + 4, label, INK, 12, "start"))
    body.append(text(20, y0 + h + 62, "23장까지는 문맥이 두 배가 될 때마다 속도가 크게 떨어졌다. 이 장의 엔진은 4,096토큰에서도 처음의 속도에 가깝다.", MUTED, 12, "start"))
    svg(OUT / "context.svg", 880, y0 + h + 78, body)


def fig_ladder():
    rows = []
    for row in read("ch24_attention.txt").splitlines()[2:7]:
        parts = row.split()
        rows.append((" ".join(parts[:-3]), float(parts[-1])))
    x0, w = 300, 420
    top = rows[0][1]
    body = [text(20, 28, "블록 하나에서 새 토큰 하나의 어텐션 (앞의 토큰 4,096개, 마이크로초)", INK, 14, "start", "bold")]
    y = 50
    colors = [MUTED, LIGHT_BLUE, LIGHT_BLUE, BLUE, GREEN]
    for (label, value), color in zip(rows, colors):
        width = value / top * w
        body.append(text(x0 - 12, y + 15, label, INK, 12, "end"))
        body.append(f"<rect x='{x0}' y='{y + 2}' width='{max(width, 2):.1f}' height='18' rx='3' fill='{color}'/>")
        body.append(text(x0 + max(width, 2) + 8, y + 16, f"{value:,.0f}", MUTED, 12, "start"))
        y += 28
    body.append(text(20, y + 12, f"처음과 끝이 {rows[0][1] / rows[-1][1]:.0f}배 차이다. 계산하는 것은 같고, 값을 놓는 자리와 곱하는 방법만 바꿨다.", MUTED, 12, "start"))
    svg(OUT / "ladder.svg", 880, y + 28, body)


def fig_variants():
    body = [text(20, 28, "새 토큰이 보는 키: 어텐션의 변형들", INK, 14, "start", "bold")]
    cell_size, count = 26, 12

    def row(y, title, note, visible, shared=None):
        out = [text(20, y + 18, title, INK, 13, "start", "bold"), text(20, y + 36, note, MUTED, 11, "start")]
        for index in range(count):
            x = 330 + index * (cell_size + 3)
            seen = visible(index)
            fill, stroke = (BLUE_BG, BLUE) if seen else (GRAY_BG, GRID)
            out.append(f"<rect x='{x}' y='{y}' width='{cell_size}' height='{cell_size}' rx='3' fill='{fill}' stroke='{stroke}'/>")
        x = 330 + count * (cell_size + 3) + 8
        out.append(f"<rect x='{x}' y='{y}' width='{cell_size}' height='{cell_size}' rx='3' fill='{ORANGE_BG}' stroke='{ORANGE}'/>")
        return out

    body.append(text(330 + (count * (cell_size + 3)) / 2, 54, "앞의 토큰들 (왼쪽이 오래된 것)", MUTED, 11))
    body.append(text(330 + count * (cell_size + 3) + 8 + cell_size / 2, 54, "새 토큰", ORANGE, 11))
    body += row(64, "전부 본다", "이 책의 모델. 일이 문맥의 길이만큼 는다", lambda i: True)
    body += row(118, "가까운 것만 본다 (창)", "일이 창의 크기에서 멈춘다. 그렇게 학습한 모델만", lambda i: i >= count - 5)
    body += row(172, "가까운 것과 몇몇 먼 것", "긴 문서용 모델들. 무엇을 볼지를 정해 둔다", lambda i: i >= count - 3 or i in (0, 4))
    body.append(text(20, 242, "파란 칸이 점수를 내고 값을 섞는 대상이다. 무엇을 보는지는 모델이 학습할 때 정해지고, 엔진이 마음대로 줄일 수 없다.", MUTED, 12, "start"))
    svg(OUT / "variants.svg", 880, 258, body)


if __name__ == "__main__":
    for figure in (fig_layout, fig_context, fig_ladder, fig_variants):
        figure()
    print(f"wrote {len(list(OUT.glob('*.svg')))} figures to {OUT}")
