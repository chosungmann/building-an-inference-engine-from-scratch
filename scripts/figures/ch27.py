"""27장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch27.py
출력: book/src/images/ch27/*.svg
"""

from common import *  # noqa: F403

OUT = IMAGES / "ch27"  # noqa: F405
GRID = "#e5e7eb"
PURPLE, PURPLE_BG = "#7c3aed", "#ede9fe"


def rect(x, y, w, h, fill, stroke=None, rx=0, width=1):
    edge = f" stroke='{stroke}' stroke-width='{width}'" if stroke else ""
    return f"<rect x='{x:.1f}' y='{y:.1f}' width='{w:.1f}' height='{h:.1f}' rx='{rx}' fill='{fill}'{edge}/>"


def path(d, stroke=INK, width=1.5, arrow=False, dash=None):  # noqa: F405
    extra = f" marker-end='url(#arrow-{stroke[1:]})'" if arrow else ""
    extra += f" stroke-dasharray='{dash}'" if dash else ""
    return f"<path d='{d}' fill='none' stroke='{stroke}' stroke-width='{width}'{extra}/>"


def board(x, y, rows, columns, size, fill=GRAY_BG, stroke=MUTED):  # noqa: F405
    """칸으로 나뉜 판 하나."""
    out = [rect(x, y, columns * size, rows * size, fill, stroke)]
    for row in range(1, rows):
        out.append(line(x, y + row * size, x + columns * size, y + row * size, stroke, 0.6))  # noqa: F405
    for column in range(1, columns):
        out.append(line(x + column * size, y, x + column * size, y + rows * size, stroke, 0.6))  # noqa: F405
    return out


def fig_convolution():
    """창이 입력 위를 움직이며 값 하나씩을 낸다."""
    body = [text(20, 28, "합성곱: 창을 옮겨 가며, 창 안의 값과 필터를 내적한다", INK, 14, "start", "bold")]  # noqa: F405
    size, x0, y0 = 30, 60, 110
    # 입력 6 × 6. 덧댄 테두리 한 칸을 점선으로.
    body.append(rect(x0 - size, y0 - size, 8 * size, 8 * size, "#ffffff", MUTED, 0, 1))  # noqa: F405
    body[-1] = body[-1].replace("/>", " stroke-dasharray='4 3'/>")
    body += board(x0, y0, 6, 6, size)
    body.append(text(x0 + 3 * size, y0 + 6 * size + size + 18, "입력 6 × 6 (채널 하나만 그렸다)", MUTED, 12))  # noqa: F405
    body.append(text(x0 + 3 * size, y0 - size - 8, "점선: 덧댄 자리 (값은 0)", MUTED, 11))  # noqa: F405
    # 창 두 개: 왼쪽 위(덧댄 자리에 걸친 것)와 가운데.
    body.append(rect(x0 - size, y0 - size, 3 * size, 3 * size, "none", ORANGE, 0, 2.5))  # noqa: F405
    body.append(rect(x0 + 2 * size, y0 + 2 * size, 3 * size, 3 * size, BLUE_BG, BLUE, 0, 2.5))  # noqa: F405
    for row in range(3):
        for column in range(3):
            body.append(text(x0 + (2.5 + column) * size, y0 + (2.65 + row) * size, "x", BLUE, 11))  # noqa: F405
    # 필터.
    fx, fy = 350, 170
    body += board(fx, fy, 3, 3, size, PURPLE_BG, PURPLE)
    for row in range(3):
        for column in range(3):
            body.append(text(fx + (0.5 + column) * size, fy + (0.65 + row) * size, "w", PURPLE, 11))
    body.append(text(fx + 1.5 * size, fy - 12, "필터 3 × 3", PURPLE, 12, "middle", "bold"))
    body.append(text(fx + 1.5 * size, fy + 3 * size + 20, "가중치 아홉 개", MUTED, 11))  # noqa: F405
    body.append(text(fx - 28, fy + 1.5 * size + 5, "·", INK, 26))  # noqa: F405
    body.append(text(fx + 3 * size + 30, fy + 1.5 * size + 5, "=", INK, 20))  # noqa: F405
    # 출력.
    ox, oy = 520, 110
    body += board(ox, oy, 6, 6, size)
    body.append(rect(ox, oy, size, size, ORANGE_BG, ORANGE, 0, 2.5))  # noqa: F405
    body.append(rect(ox + 3 * size, oy + 3 * size, size, size, BLUE_BG, BLUE, 0, 2.5))  # noqa: F405
    body.append(text(ox + 3.5 * size, oy + 3.65 * size, "y", BLUE, 11))  # noqa: F405
    body.append(text(ox + 3 * size, oy + 6 * size + size + 18, "출력 6 × 6 (보폭 1, 덧대기 있음)", MUTED, 12))  # noqa: F405
    body.append(path(f"M{x0 + 5 * size + 4},{y0 + 3.5 * size} C{fx - 60},{y0 + 3.5 * size} {fx - 60},{fy + 1.5 * size} {fx - 40},{fy + 1.5 * size}", BLUE, 1.5))  # noqa: F405
    body.append(path(f"M{fx + 3 * size + 44},{fy + 1.5 * size} L{ox + 3 * size - 4},{oy + 3.5 * size}", BLUE, 1.5, arrow=True))  # noqa: F405
    y = y0 + 8 * size + 20
    body.append(text(20, y, "창이 한 칸 움직일 때마다 출력의 값 하나가 나온다. 채널이 여럿이면 창 안의 값은 3 × 3 × (입력 채널)개이고,", MUTED, 12, "start"))  # noqa: F405
    body.append(text(20, y + 18, "출력 채널마다 필터가 하나씩 있다. 주황색 창처럼 가장자리에서는 창의 일부가 덧댄 자리에 걸친다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "convolution.svg", 780, y + 34, body)  # noqa: F405


def fig_im2col():
    """창들을 행렬의 행으로 펼치면 합성곱이 선형 층이 된다."""
    body = [text(20, 28, "im2col: 창을 행으로 펼치면 합성곱이 선형 층이 된다", INK, 14, "start", "bold")]  # noqa: F405
    size, x0, y0 = 24, 30, 76
    body += board(x0, y0, 5, 5, size)
    colors = [(BLUE_BG, BLUE), (ORANGE_BG, ORANGE), (GREEN_BG, GREEN)]  # noqa: F405
    windows = [(0, 0), (0, 1), (2, 2)]
    for (row, column), (fill, stroke) in zip(windows, colors):
        body.append(rect(x0 + column * size + 1.5, y0 + row * size + 1.5, 3 * size - 3, 3 * size - 3, "none", stroke, 0, 2.2))
    body.append(text(x0 + 2.5 * size, y0 + 5 * size + 18, "입력 (높이 × 너비 × 채널)", MUTED, 11))  # noqa: F405
    # 펼친 행렬.
    mx, my, cell_w, cell_h = 250, 62, 20, 20
    body.append(text(mx + 4.5 * cell_w, my - 10, "펼친 행렬: 행 하나가 창 하나", INK, 12, "middle", "bold"))  # noqa: F405
    rows = [0, 1, None, 2, None]
    for index, which in enumerate(rows):
        y = my + index * (cell_h + 6)
        if which is None:
            body.append(text(mx + 4.5 * cell_w, y + 15, "⋮", MUTED, 14))  # noqa: F405
            continue
        fill, stroke = colors[which]
        for column in range(9):
            body.append(rect(mx + column * cell_w, y, cell_w, cell_h, fill, stroke, 0, 1))
        source = windows[which]
        body.append(path(f"M{x0 + (source[1] + 3) * size + 2},{y0 + (source[0] + 1.5) * size} L{mx - 6},{y + cell_h / 2}", stroke, 1.2, arrow=True))
    body.append(text(mx + 4.5 * cell_w, my + 5 * (cell_h + 6) + 10, "(출력 자리의 수) × (창의 높이 × 너비 × 입력 채널)", MUTED, 11))  # noqa: F405
    # 필터 행렬과 출력.
    wx = mx + 9 * cell_w + 60
    body.append(text(wx - 30, my + 2.5 * (cell_h + 6), "×", INK, 18))  # noqa: F405
    for row in range(4):
        for column in range(9):
            body.append(rect(wx + column * cell_w, my + 20 + row * cell_h, cell_w, cell_h, PURPLE_BG, PURPLE, 0, 1))
    body.append(text(wx + 4.5 * cell_w, my + 8, "필터: 행 하나가 출력 채널 하나", PURPLE, 12, "middle", "bold"))
    body.append(text(wx + 4.5 * cell_w, my + 20 + 4 * cell_h + 18, "(출력 채널) × (창 하나의 값 수)", MUTED, 11))  # noqa: F405
    ox = wx + 9 * cell_w + 60
    body.append(text(ox - 30, my + 2.5 * (cell_h + 6), "=", INK, 18))  # noqa: F405
    for index, which in enumerate(rows):
        y = my + index * (cell_h + 6)
        if which is None:
            body.append(text(ox + 2 * cell_w, y + 15, "⋮", MUTED, 14))  # noqa: F405
            continue
        fill, stroke = colors[which]
        for column in range(4):
            body.append(rect(ox + column * cell_w, y, cell_w, cell_h, fill, stroke, 0, 1))
    body.append(text(ox + 2 * cell_w, my - 10, "출력", INK, 12, "middle", "bold"))  # noqa: F405
    body.append(text(ox + 2 * cell_w, my + 5 * (cell_h + 6) + 10, "(자리 수) × (출력 채널)", MUTED, 11))  # noqa: F405
    y = my + 5 * (cell_h + 6) + 44
    body.append(text(20, y, "가운데의 곱셈은 5장의 선형 층이다. 16~18장에서 다듬은 커널을 그대로 쓴다. 대가는 펼친 행렬이다.", MUTED, 12, "start"))  # noqa: F405
    body.append(text(20, y + 18, "입력의 값 하나가 창 아홉 개에 들어가므로 3 × 3 창이면 입력의 아홉 배쯤 되는 메모리를 쓴다.", MUTED, 12, "start"))  # noqa: F405
    body.append(text(20, y + 36, "창이 1 × 1이고 보폭이 1이면 창 하나가 픽셀 하나다. 입력이 이미 펼친 행렬이어서 펼칠 것이 없다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "im2col.svg", 900, y + 52, body)  # noqa: F405


def fig_depthwise():
    """보통의 합성곱과, 깊이별 합성곱에 1 × 1을 이은 것."""
    body = [text(20, 28, "보통의 합성곱과 깊이별 분리 합성곱: 같은 크기의 입력과 출력", INK, 14, "start", "bold")]  # noqa: F405

    def stack(x, y, channels, fill, stroke, label):
        out = []
        for index in range(channels):
            out.append(rect(x + index * 9, y - index * 7, 54, 54, fill, stroke, 2, 1.2))
        out.append(text(x + 27 + (channels - 1) * 4.5, y + 74, label, MUTED, 11))  # noqa: F405
        return out

    # 위: 보통의 합성곱.
    y = 110
    body.append(text(20, 64, "보통의 합성곱 3 × 3", INK, 12, "start", "bold"))  # noqa: F405
    body += stack(40, y, 4, BLUE_BG, BLUE, "입력 C채널")  # noqa: F405
    body += stack(300, y, 4, ORANGE_BG, ORANGE, "출력 C채널")  # noqa: F405
    for index in range(4):
        body.append(line(125, y + 27 - index * 6, 296, y + 27, MUTED, 1))  # noqa: F405
    body.append(text(210, y - 22, "출력 채널 하나가", INK, 11))  # noqa: F405
    body.append(text(210, y - 8, "입력 채널을 모두 본다", INK, 11))  # noqa: F405
    body.append(text(470, y + 10, "곱셈 (자리 하나에)", MUTED, 11, "start"))  # noqa: F405
    body.append(text(470, y + 30, "9 × C × C", INK, 14, "start", "bold"))  # noqa: F405
    # 아래: 깊이별 + 1 × 1.
    y = 260
    body.append(text(20, y - 46, "깊이별 합성곱 3 × 3, 이어서 1 × 1", INK, 12, "start", "bold"))  # noqa: F405
    body += stack(40, y, 4, BLUE_BG, BLUE, "입력 C채널")  # noqa: F405
    body += stack(220, y, 4, GREEN_BG, GREEN, "채널마다 따로 3 × 3")  # noqa: F405
    body += stack(400, y, 4, ORANGE_BG, ORANGE, "1 × 1로 채널을 섞음")  # noqa: F405
    for index in range(4):
        body.append(line(98 + index * 9, y + 27 - index * 7, 216 + index * 9, y + 27 - index * 7, MUTED, 1, arrow=True))  # noqa: F405
    for index in range(4):
        body.append(line(305, y + 27 - index * 6, 396, y + 27, MUTED, 1))  # noqa: F405
    body.append(text(570, y - 2, "곱셈 (자리 하나에)", MUTED, 11, "start"))  # noqa: F405
    body.append(text(570, y + 18, "9 × C + C × C", INK, 14, "start", "bold"))  # noqa: F405
    body.append(text(570, y + 38, "C = 256이면 8.7분의 1", GREEN, 12, "start"))  # noqa: F405
    y += 104
    body.append(text(20, y, "깊이별 합성곱은 채널 사이를 잇지 않는다. 공간을 보는 일(3 × 3)과 채널을 섞는 일(1 × 1)을 나눈 것이다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "depthwise.svg", 800, y + 18, body)  # noqa: F405


def fig_bottleneck():
    """ResNet-50의 블록 하나."""
    body = [text(20, 28, "ResNet-50의 블록: 줄이고, 보고, 늘리고, 더한다", INK, 14, "start", "bold")]  # noqa: F405
    y, h = 86, 54
    nodes = [
        (150, "CONV 1 × 1", "256 → 64", BLUE_BG, BLUE),  # noqa: F405
        (330, "CONV 3 × 3", "64 → 64", PURPLE_BG, PURPLE),
        (510, "CONV 1 × 1", "64 → 256", BLUE_BG, BLUE),  # noqa: F405
        (690, "ADD", "+ ReLU", ORANGE_BG, ORANGE),  # noqa: F405
    ]
    for x, title, sub, fill, stroke in nodes:
        body += box(x, y, 130, h, title, sub, fill, stroke)  # noqa: F405
    body.append(line(60, y + h / 2, 148, y + h / 2, INK, 1.5, arrow=True))  # noqa: F405
    for index in range(3):
        left = nodes[index][0] + 130
        body.append(line(left, y + h / 2, nodes[index + 1][0] - 2, y + h / 2, INK, 1.5, arrow=True))  # noqa: F405
    body.append(line(820, y + h / 2, 880, y + h / 2, INK, 1.5, arrow=True))  # noqa: F405
    body.append(path(f"M100,{y + h / 2} L100,{y - 28} L755,{y - 28} L755,{y - 2}", ORANGE, 1.5, arrow=True))  # noqa: F405
    body.append(text(430, y - 36, "건너뛰는 길 (56 × 56 × 256을 그대로)", ORANGE, 11))  # noqa: F405
    body.append(text(60, y + h / 2 + 22, "56×56×256", MUTED, 10, "start"))  # noqa: F405
    for x, label in [(305, "56×56×64"), (485, "56×56×64"), (665, "56×56×256")]:
        body.append(text(x, y + h + 16, label, MUTED, 10))  # noqa: F405
    y2 = y + h + 48
    body.append(text(20, y2, "곱셈의 수 (자리 3,136개)", INK, 12, "start", "bold"))  # noqa: F405
    for x, value in [(215, "5천만"), (395, "1억 1,600만"), (575, "5천만")]:
        body.append(text(x, y2, value, INK, 12))  # noqa: F405
    body.append(text(20, y2 + 26, "3 × 3을 256채널에서 그대로 하면 18억 5천만 번이다. 앞뒤의 1 × 1이 채널을 4분의 1로 줄였다가 되돌려서", MUTED, 12, "start"))  # noqa: F405
    body.append(text(20, y2 + 44, "블록 전체가 2억 1,800만 번이 된다. 이런 블록이 열여섯 개이고, 마디 53개가 합성곱이다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "bottleneck.svg", 900, y2 + 60, body)  # noqa: F405


def fig_resize():
    """줄일 때 새 픽셀 하나가 옛 픽셀 몇 개를 보는가."""
    body = [text(20, 28, "여덟 칸을 한 칸으로 줄일 때, 새 픽셀 하나가 보는 옛 픽셀", INK, 14, "start", "bold")]  # noqa: F405
    x0, size, scale = 60, 26, 8
    count = 24
    for panel, (title, kind) in enumerate([("넓이를 보는 방법 (Pillow)", "area"), ("둘만 보는 방법", "taps")]):
        y = 74 + panel * 150
        body.append(text(20, y - 14, title, INK, 12, "start", "bold"))  # noqa: F405
        center = 1.5 * scale  # 둘째 새 픽셀의 중심 (옛 픽셀 단위)
        for index in range(count):
            if kind == "area":
                weight = max(0.0, 1.0 - abs(index + 0.5 - center) / scale)
            else:
                position = center - 0.5
                weight = max(0.0, 1.0 - abs(index - position))
            fill = "#ffffff"
            if weight > 0:
                level = int(255 - 150 * weight)
                fill = f"rgb({level - 30},{level},{255})" if kind == "area" else f"rgb(255,{level + 20},{level - 40})"
            body.append(rect(x0 + index * size, y + 56, size, size, fill, MUTED, 0, 0.8))  # noqa: F405
            if weight > 0:
                height = 44 * weight
                body.append(rect(x0 + index * size + 6, y + 50 - height, size - 12, height, BLUE if kind == "area" else ORANGE))  # noqa: F405
        # 새 픽셀의 경계.
        for index in range(count // scale + 1):
            x = x0 + index * scale * size
            body.append(line(x, y + 50, x, y + 56 + size + 8, INK, 1.5))  # noqa: F405
        cx = x0 + center * size
        body.append(line(cx, y + 2, cx, y + 56 + size + 8, GREEN, 1.2, dash="4 3"))  # noqa: F405
        body.append(text(cx, y + 56 + size + 22, "새 픽셀의 중심", GREEN, 10))  # noqa: F405
        note = "옛 픽셀 열여섯 개를 삼각형 모양의 비중으로 섞는다" if kind == "area" else "중심 양옆의 둘만 섞는다. 나머지 여섯 개는 버려진다"
        body.append(text(x0 + count * size + 16, y + 30, note, MUTED, 11, "start"))  # noqa: F405
    y = 74 + 150 + 56 + size + 44
    body.append(text(20, y, "막대의 높이가 비중이다. 굵은 세로선 사이가 새 픽셀 하나가 차지하는 넓이다. 둘 다 \"쌍선형\"이라고 불린다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "resize.svg", 1000, y + 18, body)  # noqa: F405


if __name__ == "__main__":
    fig_convolution()
    fig_im2col()
    fig_depthwise()
    fig_bottleneck()
    fig_resize()
    print("그림을", OUT, "에 썼습니다")
