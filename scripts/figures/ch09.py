"""9장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch09.py
출력: book/src/images/ch09/*.svg

파라미터 비중 그림은 예제의 실행 결과 파일에서 그린다.
"""

import math
import re

from common import *  # noqa: F403

OUT = IMAGES / "ch09"  # noqa: F405
OUTPUT = IMAGES.parent / "part2/output"  # noqa: F405
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
GRID = "#e5e7eb"


def fig_architecture():
    body = []
    # 왼쪽: 모델 전체
    x, w = 60, 200
    body.append(text(x + w / 2, 28, "모델 전체", INK, 14, weight="bold"))
    rows = [
        (50, "토큰 번호", "(토큰 수)", GRAY_BG, MUTED, 40),
        (112, "임베딩", "번호 → 벡터", ORANGE_BG, ORANGE, 46),
        (182, "블록 × 16", "어텐션 + MLP", BLUE_BG, BLUE, 96),
        (302, "정규화", "RMSNorm", GREEN_BG, GREEN, 46),
        (372, "출력 층", "벡터 → 토큰마다의 점수", ORANGE_BG, ORANGE, 46),
        (442, "점수", "(토큰 수, 단어 사전 크기)", GRAY_BG, MUTED, 40),
    ]
    for i, (y, title, sub, fill, stroke, h) in enumerate(rows):
        body += box(x, y, w, h, title, sub, fill, stroke)
        if i < len(rows) - 1:
            body.append(line(x + w / 2, y + h, x + w / 2, rows[i + 1][0] - 4, MUTED, 2, arrow=True))
    body.append(text(x + w + 12, 140, "(토큰 수, 2048)", MUTED, 11, "start"))
    body.append(text(x + w + 12, 290, "(토큰 수, 2048)", MUTED, 11, "start"))

    # 오른쪽: 블록 하나
    bx, bw = 470, 190
    body.append(text(bx + bw / 2, 28, "블록 하나", INK, 14, weight="bold"))
    body.append(f"<rect x='{bx - 90}' y='44' width='{bw + 150}' height='430' rx='10' fill='none' stroke='{BLUE}' stroke-dasharray='6 4'/>")
    parts = [
        (86, "정규화", "RMSNorm", GREEN_BG, GREEN),
        (150, "어텐션", "앞의 토큰들에서 정보를 가져온다", BLUE_BG, BLUE),
        (286, "정규화", "RMSNorm", GREEN_BG, GREEN),
        (350, "MLP", "토큰마다 따로 계산한다", BLUE_BG, BLUE),
    ]
    for y, title, sub, fill, stroke in parts:
        body += box(bx, y, bw, 46, title, sub, fill, stroke)
    # 잔차 흐름: 왼쪽의 굵은 세로선
    rx = bx - 50
    body.append(line(rx, 52, rx, 466, INK, 3, arrow=True))
    body.append(text(rx - 8, 66, "입력", MUTED, 11, "end"))
    body.append(text(rx - 8, 462, "출력", MUTED, 11, "end"))
    for top, mid, plus in [(109, 173, 232), (309, 373, 432)]:
        body.append(line(rx, top, bx - 4, top, MUTED, 1.5, arrow=True))
        body.append(line(bx + bw / 2, top + 23, bx + bw / 2, mid - 27, MUTED, 1.5, arrow=True))
        body.append(line(bx + bw / 2, mid + 23, bx + bw / 2, plus, MUTED, 1.5))
        body.append(line(bx + bw / 2, plus, rx + 14, plus, MUTED, 1.5, arrow=True))
        body.append(f"<circle cx='{rx}' cy='{plus}' r='11' fill='#ffffff' stroke='{INK}' stroke-width='2'/>")
        body.append(text(rx, plus + 5, "+", INK, 16, weight="bold"))
    body.append(text(bx + bw + 14, 236, "더한다", MUTED, 11, "start"))
    body.append(text(bx + bw + 14, 436, "더한다", MUTED, 11, "start"))
    body.append(line(x + w, 230, bx - 96, 230, BLUE, 1.5, dash="6 4"))
    svg(OUT / "architecture.svg", 760, 500, body)


def fig_attention():
    body = [text(430, 26, "마지막 토큰 \"is\"가 앞의 토큰들에서 정보를 가져오는 과정 (헤드 하나)", INK, 14, weight="bold")]
    tokens = ["The", "capital", "of", "France", "is"]
    weights = [0.05, 0.30, 0.05, 0.50, 0.10]
    x0, w, gap = 150, 110, 18
    for i, (token, weight) in enumerate(zip(tokens, weights)):
        x = x0 + i * (w + gap)
        last = i == len(tokens) - 1
        body += box(x, 50, w, 34, token, None, ORANGE_BG if last else GRAY_BG, ORANGE if last else MUTED)
        body += box(x, 116, w, 34, "키", None, BLUE_BG, BLUE)
        body += box(x, 250, w, 34, "값", None, GREEN_BG, GREEN)
        body.append(line(x + w / 2, 84, x + w / 2, 112, MUTED, 1.2, arrow=True))
        # 비중을 막대의 높이로
        bar = 60 * weight / 0.5
        body.append(f"<rect x='{x + w / 2 - 16}' y='{232 - bar}' width='32' height='{bar}' rx='3' fill='{ORANGE}'/>")
        body.append(text(x + w / 2, 226 - bar, f"{weight:.2f}", INK, 12))
        body.append(line(x + w / 2, 284, x + w / 2, 318, MUTED, 1.2, arrow=True))
    qx = x0 + 4 * (w + gap)
    body.append(text(64, 72, "토큰", MUTED, 12, "end"))
    body.append(text(64, 138, "① 키", MUTED, 12, "end"))
    body.append(text(64, 200, "② 비중", MUTED, 12, "end"))
    body.append(text(64, 272, "③ 값", MUTED, 12, "end"))

    body += box(x0, 322, 5 * w + 4 * gap, 38, "④ 값들을 비중대로 섞은 벡터 = 0.05 × 값(The) + 0.30 × 값(capital) + …", None, ORANGE_BG, ORANGE)
    body.append(text(430, 384, "② \"is\"의 질의와 각 키를 내적해 점수를 얻고, 소프트맥스로 비중을 만든다.", MUTED, 12))
    body.append(text(430, 404, "비중의 숫자는 설명을 위해 지어낸 것이다. 뒤의 토큰은 볼 수 없으므로 \"is\"는 자기까지만 본다.", MUTED, 12))
    svg(OUT / "attention.svg", 860, 420, body)


def fig_rope():
    body = [text(430, 26, "위치를 회전으로 나타낸다: 같은 벡터가 위치에 따라 다른 각도로 돈다", INK, 14, weight="bold")]

    def dial(cx, cy, r, speed, title, sub):
        out = [f"<circle cx='{cx}' cy='{cy}' r='{r}' fill='#ffffff' stroke='{GRID}' stroke-width='2'/>"]
        out.append(line(cx - r - 8, cy, cx + r + 8, cy, GRID, 1))
        out.append(line(cx, cy - r - 8, cx, cy + r + 8, GRID, 1))
        for position, color in zip(range(4), [INK, BLUE, ORANGE, GREEN]):
            angle = position * speed
            ex, ey = cx + r * math.cos(angle), cy - r * math.sin(angle)
            out.append(line(cx, cy, ex, ey, color, 2.5, arrow=True))
            lx, ly = cx + (r + 16) * math.cos(angle), cy - (r + 16) * math.sin(angle)
            out.append(text(lx, ly + 4, str(position), INK, 12))
        out.append(text(cx, cy + r + 34, title, INK, 13, weight="bold"))
        out.append(text(cx, cy + r + 52, sub, MUTED, 12))
        return out

    body += dial(190, 150, 80, 0.6, "앞쪽 짝: 빠르게 돈다", "가까운 토큰 사이의 순서를 구분")
    body += dial(470, 150, 80, 0.12, "뒤쪽 짝: 느리게 돈다", "멀리 떨어진 토큰 사이의 거리를 구분")
    body.append(text(700, 110, "화살표 옆의 숫자는", MUTED, 12))
    body.append(text(700, 128, "토큰의 위치다.", MUTED, 12))
    body.append(text(700, 164, "두 벡터의 내적은", MUTED, 12))
    body.append(text(700, 182, "각도의 차이, 즉", MUTED, 12))
    body.append(text(700, 200, "위치의 차이에만", MUTED, 12))
    body.append(text(700, 218, "달려 있다.", MUTED, 12))
    svg(OUT / "rope.svg", 820, 290, body)


def fig_gqa():
    body = [text(400, 26, "질의 헤드 32개가 키·값 헤드 8개를 넷씩 함께 쓴다", INK, 14, weight="bold")]
    x0, w = 40, 22
    for g in range(8):
        gx = x0 + g * (4 * w + 8)
        for q in range(4):
            body.append(f"<rect x='{gx + q * w}' y='56' width='{w - 3}' height='30' rx='3' fill='{ORANGE_BG}' stroke='{ORANGE}'/>")
            body.append(line(gx + q * w + (w - 3) / 2, 86, gx + 2 * w - 2, 126, MUTED, 1))
        body.append(f"<rect x='{gx}' y='128' width='{4 * w - 3}' height='30' rx='3' fill='{BLUE_BG}' stroke='{BLUE}'/>")
        body.append(text(gx + 2 * w - 2, 148, str(g), INK, 12))
    body.append(text(x0 + 8 * (4 * w + 8) + 6, 76, "질의 헤드 (32)", ORANGE, 12, "start", "bold"))
    body.append(text(x0 + 8 * (4 * w + 8) + 6, 148, "키·값 헤드 (8)", BLUE, 12, "start", "bold"))
    body.append(text(400, 192, "키와 값을 만드는 선형 층이 4분의 1 크기가 되고, 11장에서 저장해 둘 키와 값도 4분의 1이 된다.", MUTED, 12))
    svg(OUT / "gqa.svg", 880, 210, body)


def fig_parameters():
    def shares(name):
        result = []
        for l in (OUTPUT / name).read_text(encoding="utf-8").splitlines():
            m = re.match(r"^(토큰 임베딩|어텐션 \(모든 블록\)|MLP \(모든 블록\))\s+(\d+)\s+([\d.]+)%", l)
            if m:
                result.append((m[1].split(" (")[0], int(m[2]), float(m[3])))
        return result

    models = [("Llama 3.2 1B", shares("ch09_anatomy_llama.txt")), ("SmolLM2 135M", shares("ch09_anatomy_smollm2.txt"))]
    body = [text(430, 26, "파라미터는 어디에 있는가", INK, 14, weight="bold")]
    x0, width, h = 150, 620, 34
    for row, (label, parts) in enumerate(models):
        y = 62 + row * 76
        total = sum(p[1] for p in parts)
        body.append(text(x0 - 12, y + 22, label, INK, 13, "end", "bold"))
        body.append(text(x0 - 12, y + 40, f"{total / 1e6:,.0f}M", MUTED, 11, "end"))
        x = x0
        for (name, count, percent), color in zip(parts, SERIES):
            w = width * count / total
            body.append(f"<rect x='{x + 1}' y='{y}' width='{w - 2}' height='{h}' rx='4' fill='{color}'/>")
            body.append(text(x + w / 2, y + h + 16, f"{name} {percent:.1f}%", INK, 12))
            x += w
    ly = 62 + 2 * 76 + 6
    for i, (name, color) in enumerate(zip(["토큰 임베딩", "어텐션", "MLP"], SERIES)):
        lx = x0 + i * 150
        body.append(f"<rect x='{lx}' y='{ly}' width='14' height='14' rx='3' fill='{color}'/>")
        body.append(text(lx + 20, ly + 12, name, INK, 12, "start"))
    svg(OUT / "parameters.svg", 860, ly + 28, body)


if __name__ == "__main__":
    for figure in (fig_architecture, fig_attention, fig_rope, fig_gqa, fig_parameters):
        figure()
    print(f"wrote {len(list(OUT.glob('*.svg')))} figures to {OUT}")
