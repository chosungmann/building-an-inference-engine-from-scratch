"""28장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch28.py
출력: book/src/images/ch28/*.svg

층마다의 비교 그림은 bench/ch28/ 의 측정(둘째 차례)을 읽어 그린다.
"""

import re

from common import *  # noqa: F403

OUT = IMAGES / "ch28"  # noqa: F405
BENCH = IMAGES.parents[2] / "bench/ch28"  # noqa: F405
GRID = "#e5e7eb"
PURPLE, PURPLE_BG = "#7c3aed", "#ede9fe"


def rect(x, y, w, h, fill, stroke=None, rx=0, width=1):
    edge = f" stroke='{stroke}' stroke-width='{width}'" if stroke else ""
    return f"<rect x='{x:.1f}' y='{y:.1f}' width='{w:.1f}' height='{h:.1f}' rx='{rx}' fill='{fill}'{edge}/>"


def path(d, stroke=INK, width=1.5, arrow=False, dash=None):  # noqa: F405
    extra = f" marker-end='url(#arrow-{stroke[1:]})'" if arrow else ""
    extra += f" stroke-dasharray='{dash}'" if dash else ""
    return f"<path d='{d}' fill='none' stroke='{stroke}' stroke-width='{width}'{extra}/>"


def fig_tiles():
    """사진 한 장을 통째로 펼치는 것과 조각마다 펼치는 것."""
    body = [text(20, 28, "일을 나누는 두 가지 방법", INK, 14, "start", "bold")]  # noqa: F405
    colors = [(BLUE_BG, BLUE), (ORANGE_BG, ORANGE), (GREEN_BG, GREEN), (PURPLE_BG, PURPLE)]  # noqa: F405
    # 왼쪽: 27장. 펼친 행렬 하나, 가중치의 행을 스레드에 나눈다.
    x0, y0 = 30, 86
    body.append(text(x0, y0 - 26, "27장: 통째로 펼치고, 가중치의 행을 나눈다", INK, 12, "start", "bold"))  # noqa: F405
    body.append(rect(x0, y0, 150, 150, GRAY_BG, MUTED, 2))  # noqa: F405
    body.append(text(x0 + 75, y0 + 70, "펼친 행렬", INK, 12))  # noqa: F405
    body.append(text(x0 + 75, y0 + 88, "(스레드 하나가 만든다)", MUTED, 10))  # noqa: F405
    body.append(text(x0 + 75, y0 + 168, "7.4MB까지", MUTED, 10))  # noqa: F405
    body.append(text(x0 + 172, y0 + 80, "×", INK, 16))  # noqa: F405
    for index, (fill, stroke) in enumerate(colors):
        body.append(rect(x0 + 196, y0 + 30 + index * 22, 120, 22, fill, stroke, 0, 1.2))
        body.append(text(x0 + 256, y0 + 45 + index * 22, f"스레드 {index + 1}", INK, 10))  # noqa: F405
    body.append(text(x0 + 256, y0 + 136, "가중치 (행이 출력 채널)", MUTED, 10))  # noqa: F405
    body.append(text(x0, y0 + 196, "스레드마다 펼친 행렬 전체를 읽는다.", MUTED, 11, "start"))  # noqa: F405
    body.append(text(x0, y0 + 212, "행렬이 캐시보다 크면 매번 메모리에서 다시 가져온다.", MUTED, 11, "start"))  # noqa: F405
    # 오른쪽: 28장. 출력을 조각내고 조각마다 펼친다.
    x1 = 480
    body.append(text(x1, y0 - 26, "28장: 출력을 조각내고, 조각마다 펼친다", INK, 12, "start", "bold"))  # noqa: F405
    for index, (fill, stroke) in enumerate(colors):
        y = y0 + index * 38
        body.append(rect(x1, y, 110, 30, fill, stroke, 2, 1.2))
        body.append(text(x1 + 55, y + 19, f"조각 (스레드 {index + 1})", INK, 10))  # noqa: F405
        body.append(text(x1 + 126, y + 20, "×", INK, 13))  # noqa: F405
        body.append(rect(x1 + 142, y, 90, 30, GRAY_BG, MUTED, 0, 1))  # noqa: F405
        body.append(text(x1 + 187, y + 19, "가중치 전체", INK, 10))  # noqa: F405
        body.append(text(x1 + 248, y + 20, "=", INK, 13))  # noqa: F405
        body.append(rect(x1 + 264, y, 70, 30, fill, stroke, 2, 1.2))
        body.append(text(x1 + 299, y + 19, "출력의 조각", INK, 10))  # noqa: F405
    body.append(text(x1 + 55, y0 + 168, "64KB까지", MUTED, 10))  # noqa: F405
    body.append(text(x1, y0 + 196, "스레드마다 자기 조각을 펼치고, 곱하고, 편향과 활성화 함수까지 끝낸다.", MUTED, 11, "start"))  # noqa: F405
    body.append(text(x1, y0 + 212, "대신 스레드마다 가중치 전체를 읽는다. 가중치가 크고 조각이 잘면 손해다.", MUTED, 11, "start"))  # noqa: F405
    svg(OUT / "tiles.svg", 930, y0 + 230, body)  # noqa: F405


def read_layers(method):
    """`ch28_layers`의 출력에서 (이름, 개수, 시간 ms)를 읽는다."""
    rows = []
    for row in (BENCH / f"layers_resnet50_12_{method}_2.txt").read_text(encoding="utf-8").splitlines()[2:]:
        match = re.match(r"\s*(\d+)×(\d+)\s+(\d+) → (\d+)\s+(\d)×\d 스트라이드 (\d)\s+(\d+)\s+\d+\s+([\d.]+)", row)
        if match:
            side, _, cin, cout, window, stride, count, time = match.groups()
            out = int(side) // int(stride) if window == "1" or stride == "1" else (int(side) - int(window)) // 2 + 1
            rows.append((f"{out}×{out}", f"{cin}→{cout}", f"{window}×{window}", float(time)))
    return rows


def fig_choose():
    """층마다, 통째로 펼치는 것과 조각내는 것의 시간."""
    whole, tiled = read_layers("whole"), read_layers("tiled")
    assert len(whole) == len(tiled) == 23, (len(whole), len(tiled))
    body = [text(20, 28, "ResNet-50의 층마다: 통째로 펼치기와 조각내기 (스레드 12개, 낮을수록 빠름)", INK, 14, "start", "bold")]  # noqa: F405
    x0, y0, row_h, scale = 190, 64, 17, 150
    top = max(max(row[3] for row in whole), max(row[3] for row in tiled))
    scale = 560 / top
    for tick in range(0, int(top * 2) + 1):
        value = tick / 2
        x = x0 + value * scale
        body.append(line(x, y0 - 6, x, y0 + len(whole) * row_h, GRID, 1))  # noqa: F405
        body.append(text(x, y0 - 10, f"{value:g}ms", MUTED, 10))  # noqa: F405
    previous = None
    for index, (a, b) in enumerate(zip(whole, tiled)):
        y = y0 + index * row_h
        if previous is not None and previous != a[0]:
            body.append(line(20, y - 1, x0 + 600, y - 1, MUTED, 0.6, dash="3 3"))  # noqa: F405
        previous = a[0]
        body.append(text(x0 - 10, y + 12, f"{a[0]:>7}  {a[1]:>9}  {a[2]}", INK, 10, "end"))  # noqa: F405
        body.append(rect(x0, y + 2, a[3] * scale, 6, MUTED))  # noqa: F405
        better = b[3] < a[3]
        body.append(rect(x0, y + 8, b[3] * scale, 6, GREEN if better else ORANGE))  # noqa: F405
    y = y0 + len(whole) * row_h + 22
    body.append(rect(20, y - 9, 22, 7, MUTED))  # noqa: F405
    body.append(text(48, y, "통째로 펼치기 (27장)", MUTED, 11, "start"))  # noqa: F405
    body.append(rect(190, y - 9, 22, 7, GREEN))  # noqa: F405
    body.append(text(218, y, "조각내기가 빠른 층", MUTED, 11, "start"))  # noqa: F405
    body.append(rect(350, y - 9, 22, 7, ORANGE))  # noqa: F405
    body.append(text(378, y, "조각내기가 느린 층", MUTED, 11, "start"))  # noqa: F405
    body.append(text(20, y + 22, "왼쪽은 출력의 크기, 채널(입력→출력), 창이다. 출력이 28 × 28 이상이면 조각내는 쪽이, 14 × 14 이하이면 통째로 펼치는 쪽이 빠르다.", MUTED, 11, "start"))  # noqa: F405
    svg(OUT / "choose.svg", 900, y + 38, body)  # noqa: F405


def fig_winograd():
    """4 × 4 입력에서 2 × 2 출력을 곱셈 16번으로."""
    body = [text(20, 28, "위노그라드 변환: 출력 2 × 2를 곱셈 16번으로", INK, 14, "start", "bold")]  # noqa: F405
    size, y0 = 26, 70

    def board(x, y, n, fill, stroke, label):
        out = [rect(x, y, n * size, n * size, fill, stroke, 0, 1.2)]
        for k in range(1, n):
            out.append(line(x, y + k * size, x + n * size, y + k * size, stroke, 0.6))  # noqa: F405
            out.append(line(x + k * size, y, x + k * size, y + n * size, stroke, 0.6))  # noqa: F405
        out.append(text(x + n * size / 2, y + n * size + 16, label, MUTED, 11))  # noqa: F405
        return out

    body += board(40, y0, 4, BLUE_BG, BLUE, "입력 4 × 4")  # noqa: F405
    body += board(40, y0 + 150, 3, PURPLE_BG, PURPLE, "필터 3 × 3")
    body.append(path(f"M{40 + 4 * size + 6},{y0 + 2 * size} L{250},{y0 + 2 * size}", BLUE, 1.5, arrow=True))  # noqa: F405
    body.append(text(205, y0 + 2 * size - 8, "더하고 빼기 (Bᵀ d B)", BLUE, 10))  # noqa: F405
    body.append(path(f"M{40 + 3 * size + 6},{y0 + 150 + 1.5 * size} L{250},{y0 + 150 + 1.5 * size}", PURPLE, 1.5, arrow=True))
    body.append(text(196, y0 + 150 + 1.5 * size - 8, "미리 한 번 (G g Gᵀ)", PURPLE, 10))
    body += board(256, y0, 4, BLUE_BG, BLUE, "바꾼 입력 4 × 4")  # noqa: F405
    body += board(256, y0 + 150, 4, PURPLE_BG, PURPLE, "바꾼 필터 4 × 4")
    body.append(text(256 + 2 * size, y0 + 4 * size + 44, "⊙", INK, 18))  # noqa: F405
    body.append(text(256 + 2 * size + 60, y0 + 4 * size + 44, "칸끼리 곱한다: 16번", INK, 11, "start"))  # noqa: F405
    body.append(path(f"M{256 + 4 * size + 8},{y0 + 110 + 2 * size} L{480},{y0 + 110 + 2 * size}", INK, 1.5, arrow=True))  # noqa: F405
    body += board(486, y0 + 110, 4, ORANGE_BG, ORANGE, "곱 4 × 4")  # noqa: F405
    body.append(path(f"M{486 + 4 * size + 6},{y0 + 110 + 2 * size} L{690},{y0 + 110 + 2 * size}", ORANGE, 1.5, arrow=True))  # noqa: F405
    body.append(text(642, y0 + 110 + 2 * size - 8, "더하고 빼기 (Aᵀ M A)", ORANGE, 10))  # noqa: F405
    body += board(696, y0 + 110 + size, 2, GREEN_BG, GREEN, "출력 2 × 2")  # noqa: F405
    y = y0 + 150 + 4 * size + 40
    body.append(text(20, y, "정의대로 하면 출력 넷에 곱셈이 4 × 9 = 36번이다(채널 쌍마다). 16번으로 줄어 2.25분의 1이 된다.", MUTED, 12, "start"))  # noqa: F405
    body.append(text(20, y + 18, "칸마다의 곱셈은 채널 전체로 보면 선형 층 하나다. 그래서 선형 층의 커널을 16번 부른다.", MUTED, 12, "start"))  # noqa: F405
    body.append(text(20, y + 36, "대가는 앞뒤의 더하고 빼기다. 계산은 싸지만 값 전체를 메모리에서 두 번 더 훑는다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "winograd.svg", 820, y + 52, body)  # noqa: F405


def fig_indirect():
    """펼친 행렬과, 가리키기만 하는 목록."""
    body = [text(20, 28, "펼치기와 가리키기", INK, 14, "start", "bold")]  # noqa: F405
    size, y0 = 26, 76
    for panel, title in enumerate(["펼치기: 창 안의 값을 복사한다", "가리키기: 값이 있는 곳만 적는다"]):
        x0 = 30 + panel * 450
        body.append(text(x0, y0 - 24, title, INK, 12, "start", "bold"))  # noqa: F405
        body.append(rect(x0, y0, 5 * size, 5 * size, GRAY_BG, MUTED, 0, 1))  # noqa: F405
        for k in range(1, 5):
            body.append(line(x0, y0 + k * size, x0 + 5 * size, y0 + k * size, MUTED, 0.6))  # noqa: F405
            body.append(line(x0 + k * size, y0, x0 + k * size, y0 + 5 * size, MUTED, 0.6))  # noqa: F405
        body.append(rect(x0 + size, y0 + size, 3 * size, 3 * size, "none", BLUE, 0, 2.2))  # noqa: F405
        body.append(text(x0 + 2.5 * size, y0 + 5 * size + 16, "입력", MUTED, 11))  # noqa: F405
        tx = x0 + 5 * size + 70
        for index in range(9):
            y = y0 + index * 15
            if panel == 0:
                body.append(rect(tx, y, 130, 13, BLUE_BG, BLUE, 0, 0.8))  # noqa: F405
            else:
                body.append(rect(tx, y, 36, 13, ORANGE_BG, ORANGE, 0, 0.8))  # noqa: F405
                row, column = divmod(index, 3)
                px, py = x0 + (column + 1.5) * size, y0 + (row + 1.5) * size
                body.append(path(f"M{tx - 2},{y + 6.5} L{px + 4},{py}", ORANGE, 0.9, arrow=True))  # noqa: F405
        if panel == 0:
            body.append(path(f"M{x0 + 4 * size + 4},{y0 + 2.5 * size} L{tx - 4},{y0 + 67}", BLUE, 1.5, arrow=True))  # noqa: F405
            body.append(text(tx + 65, y0 + 9 * 15 + 16, "칸 9개 × 채널 C개의 값", MUTED, 11))  # noqa: F405
        else:
            body.append(text(tx + 18, y0 + 9 * 15 + 16, "주소 9개", MUTED, 11))  # noqa: F405
            body.append(text(tx + 60, y0 + 62, "채널이 몇 개든", MUTED, 11, "start"))  # noqa: F405
            body.append(text(tx + 60, y0 + 78, "목록의 크기는 같다", MUTED, 11, "start"))  # noqa: F405
    y = y0 + 5 * size + 44
    body.append(text(20, y, "커널이 하는 내적은 같다. 왼쪽은 이어진 메모리 하나를 읽고, 오른쪽은 칸이 바뀔 때마다 읽는 곳을 옮긴다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "indirect.svg", 930, y + 18, body)  # noqa: F405


def fig_layout():
    """채널을 두는 세 가지 방법."""
    body = [text(20, 28, "값 2 × 2 × 채널 16개를 메모리에 늘어놓는 세 가지 순서", INK, 14, "start", "bold")]  # noqa: F405
    shades = ["#bfdbfe", "#fde68a", "#a7f3d0", "#ddd6fe"]
    edge = [BLUE, ORANGE, GREEN, PURPLE]  # noqa: F405
    x0, cell_w, cell_h = 150, 11.5, 22

    def strip(y, order, label, note):
        out = [text(20, y + 15, label, INK, 12, "start", "bold"), text(x0, y + cell_h + 14, note, MUTED, 11, "start")]  # noqa: F405
        for index, pixel in enumerate(order):
            out.append(rect(x0 + index * cell_w, y, cell_w, cell_h, shades[pixel], edge[pixel], 0, 0.6))
        return out

    # 값마다 어느 픽셀의 것인지(0~3)를 색으로 나타낸다.
    nchw = [pixel for _ in range(16) for pixel in range(4)]
    nhwc = [pixel for pixel in range(4) for _ in range(16)]
    blocked = [pixel for _ in range(2) for pixel in range(4) for _ in range(8)]
    body += strip(60, nchw, "NCHW", "채널 하나의 네 픽셀, 다음 채널의 네 픽셀, … 같은 픽셀의 채널들이 흩어져 있다")
    body += strip(124, nhwc, "NHWC", "픽셀 하나의 채널 열여섯 개, 다음 픽셀의 열여섯 개, … 같은 픽셀의 채널들이 모두 붙어 있다")
    body += strip(188, blocked, "채널 묶음 (8)", "채널 0~7에 대해 픽셀마다 여덟 개씩, 그다음 채널 8~15에 대해 다시. 레지스터 하나만큼만 붙어 있다")
    y = 258
    for pixel in range(4):
        body.append(rect(20 + pixel * 110, y - 11, 16, 12, shades[pixel], edge[pixel], 0, 0.8))
        body.append(text(42 + pixel * 110, y, f"픽셀 {pixel}의 값", MUTED, 11, "start"))  # noqa: F405
    svg(OUT / "layout.svg", 920, y + 16, body)  # noqa: F405


if __name__ == "__main__":
    fig_tiles()
    fig_winograd()
    fig_indirect()
    fig_layout()
    fig_choose()
    print("그림을", OUT, "에 썼습니다")
