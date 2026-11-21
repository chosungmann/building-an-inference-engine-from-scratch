"""6장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch06.py
출력: book/src/images/ch06/*.svg

파일 레이아웃 그림의 숫자는 crates/badak_formats/tests/data의 실제 파일에서
가져온 것이다.
"""

from common import *  # noqa: F403

OUT = IMAGES / "ch06"  # noqa: F405
GRID = "#e5e7eb"


def segment(x, y, w, h, title, sub, fill, stroke, size=13):
    out = [
        f"<rect x='{x}' y='{y}' width='{w}' height='{h}' fill='{fill}' stroke='{stroke}' stroke-width='1.5'/>",
        text(x + w / 2, y + h / 2 - (4 if sub else -5), title, INK, size, weight="bold"),
    ]
    if sub:
        out.append(text(x + w / 2, y + h / 2 + 14, sub, MUTED, 11))
    return out


def tick(x, y, label):
    return [line(x, y, x, y + 8, MUTED, 1), text(x, y + 22, label, MUTED, 11)]


def fig_safetensors():
    body = []
    y, h = 60, 58
    body.append(text(380, 30, "tiny.safetensors (420바이트)", INK, 14, weight="bold"))
    body += segment(30, y, 90, h, "N = 352", "8바이트", GRAY_BG, MUTED)
    body += segment(120, y, 330, h, "헤더 (JSON)", "352바이트", ORANGE_BG, ORANGE)
    parts = [("bias", 12), ("weight", 24), ("ids", 16), ("half", 4), ("pixels", 4)]
    x = 450
    starts = []
    for name, size in parts:
        w = 34 + size * 2.4
        body += segment(x, y, w, h, name, f"{size}", BLUE_BG, BLUE, 11 if size < 12 else 12)
        starts.append(x)
        x += w
    end = x
    for px, label in [(30, "0"), (120, "8"), (450, "360"), (end, "420")]:
        body += tick(px, y + h, label)
    body.append(text((450 + end) / 2, y - 10, "바이트 덩어리", BLUE, 13, weight="bold"))

    # 헤더의 한 항목이 덩어리의 구간을 가리킨다.
    by = 190
    body.append(
        f"<rect x='120' y='{by}' width='470' height='56' rx='6' fill='#ffffff' stroke='{ORANGE}' stroke-dasharray='5 4'/>"
    )
    body.append(text(134, by + 22, '"weight": {"dtype": "F32", "shape": [2, 3],', INK, 13, "start"))
    body.append(text(134, by + 42, '           "data_offsets": [12, 36]}', INK, 13, "start"))
    body.append(line(285, y + h + 30, 285, by - 4, ORANGE, 1.5, dash="5 4"))
    wx = starts[1]
    body.append(line(590, by + 28, wx + 45, y + h + 34, BLUE, 1.5, arrow=True))
    body.append(text(600, by + 60, "덩어리의 12번째 바이트부터 36번째 앞까지", MUTED, 12, "start"))
    body.append(text(600, by + 78, "= 파일의 372번째부터 396번째 앞까지", MUTED, 12, "start"))
    svg(OUT / "safetensors_layout.svg", 860, 290, body)


def fig_gguf():
    body = []
    y, h = 60, 58
    body.append(text(400, 30, "tiny.gguf (608바이트)", INK, 14, weight="bold"))
    body += segment(30, y, 120, h, "머리", "24바이트", GRAY_BG, MUTED)
    body += segment(150, y, 230, h, "메타데이터", "키와 값 9개", ORANGE_BG, ORANGE)
    body += segment(380, y, 140, h, "텐서 정보", "3개", GREEN_BG, GREEN)
    body.append(
        f"<rect x='520' y='{y}' width='40' height='{h}' fill='#ffffff' stroke='{MUTED}' stroke-dasharray='4 3'/>"
    )
    body.append(text(540, y + h / 2 + 4, "0", MUTED, 11))
    x = 560
    for name, used in [("weight", "24"), ("ids", "16"), ("half", "4")]:
        body += segment(x, y, 76, h, name, f"{used} + 채움", BLUE_BG, BLUE, 12)
        x += 76
    for px, label in [(30, "0"), (150, "24"), (560, "512"), (636, "544"), (712, "576"), (788, "608")]:
        body += tick(px, y + h, label)
    body.append(text(674, y - 10, "텐서의 값 (32바이트 단위로 정렬)", BLUE, 13, weight="bold"))
    body.append(text(540, y + h + 44, "채움", MUTED, 11))

    # 머리의 내용
    hy = 180
    body.append(
        f"<rect x='30' y='{hy}' width='330' height='98' rx='6' fill='#ffffff' stroke='{MUTED}' stroke-dasharray='5 4'/>"
    )
    rows = [("식별 문자", '"GGUF"', "4바이트"), ("버전", "3", "4바이트"), ("텐서 수", "3", "8바이트"), ("메타데이터 수", "9", "8바이트")]
    for i, (key, value, size) in enumerate(rows):
        yy = hy + 24 + i * 20
        body.append(text(46, yy, key, MUTED, 12, "start"))
        body.append(text(190, yy, value, INK, 12, "start"))
        body.append(text(344, yy, size, MUTED, 12, "end"))
    body.append(line(90, y + h + 30, 90, hy - 4, MUTED, 1.5, dash="5 4"))

    # 텐서 정보 한 항목
    body.append(
        f"<rect x='400' y='{hy}' width='400' height='98' rx='6' fill='#ffffff' stroke='{GREEN}' stroke-dasharray='5 4'/>"
    )
    rows = [("이름", '"weight"'), ("차원", "[3, 2]  ← 빠른 축이 먼저. 뒤집으면 (2, 3)"), ("종류", "0 (F32)"), ("위치", "0  ← 값 영역의 시작에서 센 위치")]
    for i, (key, value) in enumerate(rows):
        yy = hy + 24 + i * 20
        body.append(text(416, yy, key, MUTED, 12, "start"))
        body.append(text(470, yy, value, INK, 12, "start"))
    body.append(line(450, y + h + 30, 450, hy - 4, GREEN, 1.5, dash="5 4"))
    svg(OUT / "gguf_layout.svg", 830, 300, body)


def fig_mmap():
    body = []

    def pages(x, y, count, filled, color, fill, w=26):
        out = []
        for i in range(count):
            f = fill if i in filled else "#ffffff"
            out.append(f"<rect x='{x + i * w}' y='{y}' width='{w - 3}' height='30' fill='{f}' stroke='{color}'/>")
        return out

    # 왼쪽: 통째로 읽기
    body.append(text(190, 28, "통째로 읽기", INK, 14, weight="bold"))
    body.append(text(30, 62, "디스크의 파일", MUTED, 12, "start"))
    body += pages(30, 70, 12, set(range(12)), MUTED, GRAY_BG)
    for i in range(12):
        body.append(line(30 + i * 26 + 11, 102, 30 + i * 26 + 11, 138, ORANGE, 1.2, arrow=True))
    body.append(text(30, 162, "프로그램의 메모리", MUTED, 12, "start"))
    body += pages(30, 170, 12, set(range(12)), ORANGE, ORANGE_BG)
    body.append(text(190, 228, "여는 순간 전부 복사한다.", INK, 13))
    body.append(text(190, 248, "쓰지 않을 부분까지 읽고, 메모리도 그만큼 쓴다.", MUTED, 12))

    body.append(line(385, 20, 385, 260, GRID, 1))

    # 오른쪽: 메모리에 매핑
    ox = 420
    body.append(text(ox + 160, 28, "메모리에 매핑", INK, 14, weight="bold"))
    body.append(text(ox, 62, "디스크의 파일", MUTED, 12, "start"))
    body += pages(ox, 70, 12, set(range(12)), MUTED, GRAY_BG)
    touched = {3, 4, 5}
    for i in touched:
        body.append(line(ox + i * 26 + 11, 102, ox + i * 26 + 11, 138, BLUE, 1.5, arrow=True))
    body.append(text(ox, 162, "프로그램이 보는 주소 공간", MUTED, 12, "start"))
    body += pages(ox, 170, 12, touched, BLUE, BLUE_BG)
    body.append(text(ox + 160, 228, "여는 순간에는 주소만 잡아 둔다.", INK, 13))
    body.append(text(ox + 160, 248, "실제로 읽은 부분만 운영체제가 그때 가져온다.", MUTED, 12))
    svg(OUT / "mmap.svg", 770, 270, body)


def fig_alignment():
    body = []
    w, h = 34, 34

    def ruler(x, y, start_byte, label, color, fill, ok, note):
        out = [text(x, y - 14, label, INK, 13, "start", "bold")]
        for i in range(16):
            in_value = start_byte <= i < start_byte + 8
            f = fill if in_value else "#ffffff"
            out.append(f"<rect x='{x + i * w}' y='{y}' width='{w}' height='{h}' fill='{f}' stroke='{MUTED}'/>")
            out.append(text(x + i * w + w / 2, y + h + 15, str(i), MUTED, 11))
        for i in range(0, 17, 4):
            out.append(line(x + i * w, y - 6, x + i * w, y + h + 4, INK, 2))
        for k in range(2):
            sx = x + (start_byte + 4 * k) * w
            out.append(
                f"<rect x='{sx + 2}' y='{y + 3}' width='{4 * w - 4}' height='{h - 6}' rx='4' fill='none' stroke='{color}' stroke-width='2'/>"
            )
            out.append(text(sx + 2 * w, y + h / 2 + 5, f"f32 #{k}", INK, 12))
        out.append(text(x + 16 * w + 16, y + h / 2 + 5, note, color if not ok else GREEN, 13, "start", "bold"))
        return out

    body.append(text(30, 26, "굵은 세로선: 4의 배수인 주소. f32는 여기서 시작해야 한다.", MUTED, 12, "start"))
    body += ruler(30, 70, 4, "4번째 바이트에서 시작", BLUE, BLUE_BG, True, "그대로 빌려 쓴다")
    body += ruler(30, 170, 5, "5번째 바이트에서 시작", ORANGE, ORANGE_BG, False, "복사해서 맞춘다")
    svg(OUT / "alignment.svg", 760, 240, body)


if __name__ == "__main__":
    for figure in (fig_safetensors, fig_gguf, fig_mmap, fig_alignment):
        figure()
    print(f"wrote {len(list(OUT.glob('*.svg')))} figures to {OUT}")
