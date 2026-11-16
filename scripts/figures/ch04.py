"""4장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch04.py
출력: book/src/images/ch04/*.svg
"""

from common import *  # noqa: F403

OUT = IMAGES / "ch04"  # noqa: F405


def strip(x, y, w, h, labels, fills, positions=True):
    """저장소를 한 줄의 칸으로 그린다."""
    out = grid(x, y, 1, len(labels), w, h, labels, fills)
    if positions:
        for i in range(len(labels)):
            out.append(text(x + i * w + w / 2, y + h + 16, str(i), MUTED, 11))
    return out


def meta(x, y, title, rows, stroke, fill):
    """텐서의 메타데이터를 적은 상자."""
    h = 34 + 22 * len(rows)
    out = [
        f"<rect x='{x}' y='{y}' width='210' height='{h}' rx='8' fill='{fill}' "
        f"stroke='{stroke}' stroke-width='1.5'/>",
        text(x + 105, y + 22, title, INK, 14, weight="bold"),
    ]
    for i, (key, value) in enumerate(rows):
        out.append(text(x + 16, y + 48 + i * 22, key, MUTED, 13, "start"))
        out.append(text(x + 194, y + 48 + i * 22, value, INK, 13, "end"))
    return out, h


def fig_anatomy():
    body = []
    labels = [str(i) for i in range(6)]
    sx, sy, w, h = 250, 150, 56, 44
    body.append(text(sx + 3 * w, sy - 14, "저장소 (숫자가 실제로 있는 곳)", INK, 14, weight="bold"))
    body += strip(sx, sy, w, h, labels, [GRAY_BG] * 6)
    body.append(text(sx - 10, sy + h + 16, "위치", MUTED, 11, "end"))

    a, ah = meta(20, 20, "텐서 t", [("형상", "(2, 3)"), ("스트라이드", "(3, 1)"), ("시작", "0")], BLUE, BLUE_BG)
    b, bh = meta(20, 230, "t.transpose(0, 1)", [("형상", "(3, 2)"), ("스트라이드", "(1, 3)"), ("시작", "0")], ORANGE, ORANGE_BG)
    body += a + b
    body.append(line(230, 20 + ah / 2, sx + 10, sy - 4, BLUE, 2, arrow=True))
    body.append(line(230, 230 + bh / 2, sx + 10, sy + h + 26, ORANGE, 2, arrow=True))

    # 오른쪽: 각 텐서가 읽는 모습
    gx, gw, gh = 640, 40, 34
    body.append(text(gx + 1.5 * gw, 34, "t가 읽는 모습", BLUE, 13, weight="bold"))
    body += grid(gx, 44, 2, 3, gw, gh, ["0", "1", "2", "3", "4", "5"], [BLUE_BG] * 6)
    body.append(text(gx + gw, 246, "전치 뷰가 읽는 모습", ORANGE, 13, weight="bold"))
    body += grid(gx + gw * 0.5, 256, 3, 2, gw, gh, ["0", "3", "1", "4", "2", "5"], [ORANGE_BG] * 6)
    body.append(text(400, 385, "두 텐서는 같은 저장소를 가리킨다. 다른 것은 읽는 방법뿐이다.", MUTED, 13))
    svg(OUT / "anatomy.svg", 800, 405, body)


def fig_strides():
    body = []
    x0, y0, w, h = 40, 110, 28, 36
    palette = [BLUE_BG, ORANGE_BG, GREEN_BG]
    fills = [palette[(i // 4) % 3] for i in range(24)]
    body.append(text(x0 + 12 * w, 30, "형상 (2, 3, 4)인 텐서의 저장소: 원소 24개", INK, 14, weight="bold"))
    body += strip(x0, y0, w, h, [str(i) for i in range(24)], fills)
    body.append(line(x0 + 12 * w, y0 - 8, x0 + 12 * w, y0 + h + 24, INK, 2, dash="4 3"))

    def bracket(start, count, y, label, color):
        x1, x2 = x0 + start * w + 2, x0 + (start + count) * w - 2
        return [
            line(x1, y, x2, y, color, 2),
            line(x1, y - 5, x1, y + 5, color, 2),
            line(x2, y - 5, x2, y + 5, color, 2),
            text((x1 + x2) / 2, y - 8, label, color, 13, weight="bold"),
        ]

    body += bracket(0, 12, 70, "축 0에서 한 칸: 12", INK)
    body += bracket(12, 4, 84, "축 1에서 한 칸: 4", ORANGE)
    body += bracket(18, 1, 84, "축 2: 1", BLUE)
    body.append(text(x0 + 12 * w, 210, "(i, j, k)의 위치 = i × 12 + j × 4 + k × 1", INK, 14))
    body.append(text(x0 + 12 * w, 234, "예: (1, 2, 3) → 12 + 8 + 3 = 23", MUTED, 13))
    svg(OUT / "strides.svg", 760, 255, body)


def fig_slice():
    body = []
    w, h = 44, 36
    labels = [str(i) for i in range(12)]

    def panel(x, title, selected, color, fill, note1, note2):
        out = [text(x + 2 * w, 30, title, INK, 14, weight="bold")]
        fills = [fill if i in selected else GRAY_BG for i in range(12)]
        out += grid(x, 44, 3, 4, w, h, labels, fills)
        out.append(text(x + 2 * w, 180, "저장소에서는", MUTED, 12))
        out += strip(x - 44, 190, 22, 26, labels, fills, positions=False)
        out.append(text(x + 2 * w, 244, note1, color, 13, weight="bold"))
        out.append(text(x + 2 * w, 264, note2, MUTED, 12))
        return out

    body += panel(70, "t.slice(0, 1..3)", set(range(4, 12)), BLUE, BLUE_BG,
                  "한 덩어리로 이어진다 → 연속", "시작 4, 형상 (2, 4), 스트라이드 (4, 1)")
    body += panel(430, "t.slice(1, 1..3)", {1, 2, 5, 6, 9, 10}, ORANGE, ORANGE_BG,
                  "중간에 건너뛴다 → 연속이 아님", "시작 1, 형상 (3, 2), 스트라이드 (4, 1)")
    svg(OUT / "slice.svg", 720, 285, body)


def fig_broadcast():
    body = []
    w, h = 44, 36
    body.append(text(62, 30, "형상 (3, 1)", INK, 14, weight="bold"))
    body += grid(40, 44, 3, 1, w, h, ["10", "20", "30"], [BLUE_BG] * 3)
    body.append(text(118, 104, "+", INK, 22))
    body.append(text(240, 30, "형상 (1, 4)", INK, 14, weight="bold"))
    body += grid(152, 80, 1, 4, w, h, ["1", "2", "3", "4"], [ORANGE_BG] * 4)
    body.append(text(362, 104, "=", INK, 22))

    # 늘어난 모습
    bx = 400
    body.append(text(bx + 2 * w, 30, "둘 다 (3, 4)로 늘려서", INK, 14, weight="bold"))
    values = [10, 20, 30]
    for r in range(3):
        for c in range(4):
            x, y = bx + c * w, 44 + r * h
            body.append(
                f"<rect x='{x}' y='{y}' width='{w}' height='{h}' fill='{GREEN_BG}' stroke='{MUTED}'/>"
            )
            body.append(text(x + w / 2, y + h / 2 + 5, str(values[r] + c + 1)))
    body.append(text(bx + 2 * w, 44 + 3 * h + 22, "결과: 형상 (3, 4)", GREEN, 13, weight="bold"))

    body.append(line(40, 200, 590, 200, "#e5e7eb", 1))
    body.append(text(40, 228, "늘린다고 숫자를 복사하지 않는다.", INK, 13, "start"))
    body.append(text(40, 250, "(3, 1)을 (3, 4)로: 스트라이드 (1, 1) → (1, 0). 가로로 움직여도 같은 칸을 다시 읽는다.", MUTED, 12, "start"))
    body.append(text(40, 270, "(1, 4)를 (3, 4)로: 스트라이드 (4, 1) → (0, 1). 세로로 움직여도 같은 줄을 다시 읽는다.", MUTED, 12, "start"))
    svg(OUT / "broadcast.svg", 640, 290, body)


if __name__ == "__main__":
    for figure in (fig_anatomy, fig_strides, fig_slice, fig_broadcast):
        figure()
    print(f"wrote {len(list(OUT.glob('*.svg')))} figures to {OUT}")
