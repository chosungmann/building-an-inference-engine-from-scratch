"""2장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch02.py
출력: book/src/images/ch02/*.svg
"""

from common import *  # noqa: F403

OUT = IMAGES / "ch02"  # noqa: F405
def fig_dot():
    # 원점 (70, 250), 한 칸 45픽셀. a = (4, 1), b = (2, 3).
    ox, oy, u = 70, 250, 45
    body = []
    for i in range(6):
        body.append(line(ox + i * u, oy, ox + i * u, oy - 4 * u, "#e5e7eb", 1))
    for j in range(5):
        body.append(line(ox, oy - j * u, ox + 5 * u, oy - j * u, "#e5e7eb", 1))
    body.append(line(ox, oy, ox + 5 * u + 10, oy, MUTED, 1.5, arrow=True))
    body.append(line(ox, oy, ox, oy - 4 * u - 10, MUTED, 1.5, arrow=True))
    for i in range(1, 6):
        body.append(text(ox + i * u, oy + 18, str(i), MUTED, 12))
    for j in range(1, 5):
        body.append(text(ox - 12, oy - j * u + 4, str(j), MUTED, 12))
    body.append(line(ox, oy, ox + 4 * u, oy - 1 * u, BLUE, 3, arrow=True))
    body.append(line(ox, oy, ox + 2 * u, oy - 3 * u, ORANGE, 3, arrow=True))
    body.append(text(ox + 4 * u + 12, oy - u + 20, "a = (4, 1)", BLUE, 14, "start", "bold"))
    body.append(text(ox + 2 * u + 10, oy - 3 * u - 8, "b = (2, 3)", ORANGE, 14, "start", "bold"))
    body.append(f"<path d='M {ox + 44} {oy - 11} A 45 45 0 0 0 {ox + 25} {oy - 37}' fill='none' stroke='{GREEN}' stroke-width='2'/>")
    body.append(text(ox + 52, oy - 30, "θ", GREEN, 16, "start", "bold"))
    x0 = 345
    body.append(text(x0, 70, "내적", INK, 15, "start", "bold"))
    body.append(text(x0, 96, "a · b = 4×2 + 1×3 = 11", INK, 14, "start"))
    body.append(text(x0, 136, "길이", INK, 15, "start", "bold"))
    body.append(text(x0, 162, "‖a‖ = √17 ≈ 4.12", BLUE, 14, "start"))
    body.append(text(x0, 184, "‖b‖ = √13 ≈ 3.61", ORANGE, 14, "start"))
    body.append(text(x0, 224, "코사인 유사도", INK, 15, "start", "bold"))
    body.append(text(x0, 250, "cos θ = 11 / (4.12 × 3.61) ≈ 0.74", GREEN, 14, "start"))
    svg(OUT / "dot_product.svg", 620, 290, body)


def fig_matvec():
    w, h = 52, 44
    body = [text(40 + 1.5 * w, 34, "W  (2 × 3)", INK, 14, weight="bold")]
    fills = [BLUE_BG] * 3 + [GRAY_BG] * 3
    body += grid(40, 50, 2, 3, w, h, ["1", "2", "3", "4", "5", "6"], fills)
    body.append(text(225, 100, "×", INK, 22))
    body.append(text(260 + w / 2, 14, "x  (3)", INK, 14, weight="bold"))
    body += grid(260, 28, 3, 1, w, h, ["1", "0", "−1"], [ORANGE_BG] * 3)
    body.append(text(345, 100, "=", INK, 22))
    wide = 3.4 * w
    body.append(text(380 + wide / 2, 34, "y  (2)", INK, 14, weight="bold"))
    body += grid(380, 50, 2, 1, wide, h, ["1×1 + 2×0 + 3×(−1)", "4×1 + 5×0 + 6×(−1)"], [GREEN_BG, GRAY_BG])
    for r in range(2):
        body += [cell(380 + wide + 30, 50 + r * h, w, h, "−2", GREEN_BG if r == 0 else GRAY_BG)]
        body.append(text(380 + wide + 15, 50 + r * h + h / 2 + 6, "=", INK, 18))
    body.append(text(340, 190, "y의 각 원소는 W의 한 행과 x의 내적이다. 색칠한 첫 행이 y의 첫 원소를 만든다.", MUTED, 13))
    svg(OUT / "matvec.svg", 680, 210, body)


def fig_matmul_shape():
    body = []
    def box(x, y, w, h, fill, stroke, name, rows, cols):
        return [
            f"<rect x='{x}' y='{y}' width='{w}' height='{h}' fill='{fill}' stroke='{stroke}' stroke-width='2'/>",
            text(x + w / 2, y + h / 2 + 6, name, INK, 20, weight="bold"),
            text(x - 12, y + h / 2 + 5, rows, stroke, 15, "end", "bold"),
            text(x + w / 2, y - 10, cols, stroke, 15, weight="bold"),
        ]
    body += box(60, 70, 110, 150, BLUE_BG, BLUE, "A", "m", "k")
    body.append(text(205, 152, "×", INK, 22))
    body += box(265, 90, 170, 110, ORANGE_BG, ORANGE, "B", "k", "n")
    body.append(text(470, 152, "=", INK, 22))
    body += box(525, 70, 170, 150, GREEN_BG, GREEN, "C", "m", "n")
    body.append(f"<path d='M 128 52 C 160 14, 253 30, 253 126' fill='none' stroke='{MUTED}' stroke-width='1.5' stroke-dasharray='5 4'/>")
    body.append(text(225, 26, "안쪽 두 수가 같아야 한다", MUTED, 13, "start"))
    body.append(text(250, 262, "(m × k) · (k × n) = (m × n)", INK, 14))
    body.append(text(520, 262, "곱셈 횟수 = m × k × n", INK, 14))
    svg(OUT / "matmul_shape.svg", 760, 285, body)


def fig_row_major():
    w, h = 56, 44
    body = [text(40 + 1.5 * w, 30, "2 × 3 행렬", INK, 14, weight="bold")]
    labels = ["a", "b", "c", "d", "e", "f"]
    fills = [BLUE_BG] * 3 + [ORANGE_BG] * 3
    body += grid(40, 44, 2, 3, w, h, labels, fills)
    body.append(text(20, 44 + h / 2 + 5, "0행", BLUE, 12))
    body.append(text(20, 44 + h + h / 2 + 5, "1행", ORANGE, 12))
    body.append(line(225, 88, 285, 88, MUTED, 2, arrow=True))
    x0 = 300
    body.append(text(x0 + 3 * w, 62, "메모리에 놓인 모습 (행 우선)", INK, 14, weight="bold"))
    body += grid(x0, 76, 1, 6, w, h, labels, fills)
    for i in range(6):
        body.append(text(x0 + i * w + w / 2, 76 + h + 18, str(i), MUTED, 12))
    body.append(text(x0 - 8, 76 + h + 18, "위치", MUTED, 12, "end"))
    body.append(text(200, 180, "(i, j) 원소의 위치 = i × 열 수 + j", INK, 14))
    body.append(text(500, 180, "예: (1, 2) → 1 × 3 + 2 = 5", INK, 14))
    svg(OUT / "row_major.svg", 680, 200, body)


def fig_f32_bits():
    x0, y0, w, h = 30, 50, 19, 40
    body = []
    def span(start, count, fill, stroke, label, sub):
        out = []
        for i in range(count):
            out.append(f"<rect x='{x0 + (start + i) * w}' y='{y0}' width='{w}' height='{h}' fill='{fill}' stroke='{stroke}'/>")
        cx = x0 + (start + count / 2) * w
        out.append(text(cx, y0 - 12, label, stroke, 14, weight="bold"))
        out.append(text(cx, y0 + h + 20, sub, MUTED, 12))
        return out
    body += span(0, 1, GREEN_BG, GREEN, "부호", "1비트")
    body += span(1, 8, ORANGE_BG, ORANGE, "지수", "8비트")
    body += span(9, 23, BLUE_BG, BLUE, "가수", "23비트")
    body.append(text(x0 + 16 * w, 150, "값 = (−1)^부호 × 1.가수 × 2^(지수 − 127)", INK, 14))
    body.append(text(x0 + 16 * w, 176, "가수 23비트 + 숨은 1비트 = 유효 숫자 24비트 ≈ 십진수 7자리", MUTED, 13))
    svg(OUT / "f32_bits.svg", 670, 195, body)


if __name__ == "__main__":
    for figure in (fig_dot, fig_matvec, fig_matmul_shape, fig_row_major, fig_f32_bits):
        figure()
    print(f"wrote {len(list(OUT.glob('*.svg')))} figures to {OUT}")
