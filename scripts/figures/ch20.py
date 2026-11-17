"""20장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch20.py
출력: book/src/images/ch20/*.svg

측정값은 book/src/part3/output/ch20_*.txt와 bench/의 기록에서 읽는다.
다시 측정하면 그림도 다시 만들어야 한다.
"""

import re

from common import *  # noqa: F403

OUT = IMAGES / "ch20"  # noqa: F405
ROOT = IMAGES.parents[2]  # noqa: F405
OUTPUT = IMAGES.parent / "part3/output"  # noqa: F405
GRID = "#e5e7eb"
LIGHT_BLUE = "#93c5fd"


def read(name):
    return (OUTPUT / name).read_text(encoding="utf-8")


def fig_blocks():
    body = [text(20, 30, "값 32개를 담는 묶음 하나", INK, 14, "start", "bold")]
    unit = 18

    def layout(y, name, note, parts):
        out = [text(20, y + 22, name, INK, 13, "start", "bold")]
        x = 90
        for width, label, fill, stroke in parts:
            out.append(f"<rect x='{x}' y='{y}' width='{width * unit - 2}' height='34' rx='3' fill='{fill}' stroke='{stroke}'/>")
            out.append(text(x + width * unit / 2 - 1, y + 22, label, INK, 12))
            x += width * unit
        out.append(text(x + 10, y + 22, note, MUTED, 12, "start"))
        return out

    body += layout(56, "Q8_0", "34바이트, 값 하나에 8.5비트", [(2, "d", ORANGE_BG, ORANGE), (32, "8비트 정수 32개 (-127 ~ 127)", BLUE_BG, BLUE)])
    body += layout(112, "Q4_0", "18바이트, 값 하나에 4.5비트", [(2, "d", ORANGE_BG, ORANGE), (16, "4비트 정수 32개 (-8 ~ 7)", GREEN_BG, GREEN)])
    body.append(text(90, 172, "칸 하나가 1바이트다. d는 눈금의 크기이고 16비트 실수다. 되돌린 값은 d × 정수.", MUTED, 12, "start"))
    # Q4_0의 바이트 하나
    body.append(text(20, 214, "Q4_0의 바이트 하나", INK, 13, "start", "bold"))
    body.append(f"<rect x='200' y='194' width='118' height='32' rx='3' fill='{GREEN_BG}' stroke='{GREEN}'/>")
    body.append(text(259, 215, "위 4비트", INK, 12))
    body.append(f"<rect x='320' y='194' width='118' height='32' rx='3' fill='{GREEN_BG}' stroke='{GREEN}'/>")
    body.append(text(379, 215, "아래 4비트", INK, 12))
    body.append(text(259, 246, "(i + 16)번째 값", MUTED, 11))
    body.append(text(379, 246, "i번째 값", MUTED, 11))
    body.append(text(460, 215, "저장된 0 ~ 15에서 8을 빼면 -8 ~ 7이 된다.", MUTED, 12, "start"))
    svg(OUT / "blocks.svg", 900, 262, body)


def fig_integer_dot():
    body = [text(20, 30, "묶음 하나의 내적: 정수로 곱하고, 마지막에 한 번 실수로", INK, 14, "start", "bold")]
    # 왼쪽: 가중치 묶음과 입력 묶음
    body += box(20, 60, 250, 54, "가중치 묶음", "눈금 d_w, 정수 32개", BLUE_BG, BLUE)
    body += box(20, 134, 250, 54, "입력 묶음", "눈금 d_x, 정수 32개 (방금 만듦)", ORANGE_BG, ORANGE)
    body += box(340, 96, 200, 56, "정수 32쌍의 곱의 합", "정수 넷에 나누어 담는다", GRAY_BG, MUTED)
    body += box(610, 96, 230, 56, "× (d_w × d_x)", "f32로 바꿔 합에 더한다", GREEN_BG, GREEN)
    body.append(line(270, 87, 336, 116, INK, 1.5, arrow=True))
    body.append(line(270, 161, 336, 132, INK, 1.5, arrow=True))
    body.append(line(540, 124, 606, 124, INK, 1.5, arrow=True))
    body.append(text(440, 176, "정확하다. 순서를 바꿔도 같다.", MUTED, 11))
    body.append(text(725, 176, "반올림이 있다. 순서를 정해 둔다.", MUTED, 11))
    body.append(text(20, 222, "행 하나는 묶음 64개(열 2,048개)다. 곱셈 2,048번 가운데 실수 연산은 묶음마다 몇 번뿐이다.", MUTED, 12, "start"))
    svg(OUT / "integer_dot.svg", 880, 240, body)


def read_scaling(name):
    rows = []
    for row in read(name).splitlines():
        match = re.match(r"\s+(\d+)\s+([\d.]+) 토큰/초\s+([\d.]+)배\s+([\d.]+) 토큰/초\s+([\d.]+)배", row)
        if match:
            rows.append((int(match[1]), float(match[2]), float(match[4])))
    return rows


FORMATS = [("f32", "f32", MUTED), ("stored", "BF16", LIGHT_BLUE), ("q8_0", "Q8_0", BLUE), ("q4_0", "Q4_0", ORANGE)]


def fig_scaling():
    x0, y0, w, h, top = 70, 56, 540, 300, 250
    body = [text(20, 30, "Llama 3.2 1B: 가중치를 줄일수록 토큰 생성의 벽이 물러난다", INK, 14, "start", "bold")]

    def pos(threads, speed):
        return x0 + threads / 16 * w, y0 + h - speed / top * h

    for value in range(0, top + 1, 50):
        _, yy = pos(0, value)
        body.append(line(x0, yy, x0 + w, yy, GRID, 1))
        body.append(text(x0 - 8, yy + 4, str(value), MUTED, 11, "end"))
    for threads in (1, 2, 4, 6, 8, 10, 12, 14, 16):
        xx, _ = pos(threads, 0)
        body.append(text(xx, y0 + h + 18, str(threads), MUTED, 11))
    body.append(text(x0 + w / 2, y0 + h + 42, "스레드의 수", MUTED, 12))
    body.append(text(x0 - 8, y0 - 12, "토큰/초", MUTED, 11, "start"))
    xx, _ = pos(12, 0)
    body.append(line(xx, y0, xx, y0 + h, MUTED, 1, dash="3 4"))
    body.append(text(xx + 6, y0 + h - 8, "여기부터 효율 코어", MUTED, 11, "start"))
    for key, label, color in FORMATS:
        rows = read_scaling(f"ch20_scaling_{key}.txt")
        path = " ".join("{:.1f},{:.1f}".format(*pos(t, tg)) for t, _, tg in rows)
        body.append(f"<polyline points='{path}' fill='none' stroke='{color}' stroke-width='2'/>")
        for t, _, tg in rows:
            px, py = pos(t, tg)
            body.append(f"<circle cx='{px:.1f}' cy='{py:.1f}' r='4' fill='{color}' stroke='#ffffff' stroke-width='2'/>")
        best = max(tg for _, _, tg in rows)
        px, py = pos(rows[-1][0], rows[-1][2])
        body.append(text(px + 12, py + 4, f"{label} (가장 빠를 때 {best:.0f})", INK, 12, "start"))
    svg(OUT / "scaling.svg", 880, y0 + h + 58, body)


def llama(section, threads, test):
    text_ = (ROOT / "bench/ch20/llama_cpp_llama32.txt").read_text(encoding="utf-8")
    part = text_.split(f"## {section}")[1].split("## ")[0]
    for row in part.splitlines():
        cells = [cell.strip() for cell in row.split("|")]
        if len(cells) > 8 and cells[5] == str(threads) and cells[7] == test:
            return float(cells[8].split("±")[0])
    raise ValueError((section, threads, test))


def fig_result():
    groups = []
    for title, test, column, top_value in [("토큰 생성", "tg32", 2, 260.0), ("프롬프트 처리", "pp256", 1, 1400.0)]:
        bars = []
        for key, label, file in [("f32", "f32", "model_f32.gguf"), ("stored", "BF16", None), ("q8_0", "Q8_0", "model_q8_0.gguf"), ("q4_0", "Q4_0", "model_q4_0.gguf")]:
            ours = next(row for row in read_scaling(f"ch20_scaling_{key}.txt") if row[0] == 12)[column]
            theirs = llama(file, 12, test) if file else None
            bars.append((label, ours, theirs))
        groups.append((title, bars, top_value))
    x0, w = 150, 520
    body = [text(20, 28, "Llama 3.2 1B, 스레드 12개: 1초에 처리하는 토큰", INK, 14, "start", "bold")]
    body.append(f"<rect x='{x0 + 300}' y='16' width='14' height='11' rx='2' fill='{BLUE}'/>")
    body.append(text(x0 + 320, 26, "badak", MUTED, 11, "start"))
    body.append(f"<rect x='{x0 + 380}' y='16' width='14' height='11' rx='2' fill='{ORANGE}'/>")
    body.append(text(x0 + 400, 26, "llama.cpp (직접 짠 CPU 코드만)", MUTED, 11, "start"))
    y = 54
    for title, bars, top_value in groups:
        body.append(text(20, y + 16, title, INK, 13, "start", "bold"))
        for label, ours, theirs in bars:
            body.append(text(x0 - 12, y + 15, label, INK, 12, "end"))
            for value, color in ((ours, BLUE), (theirs, ORANGE)):
                if value is None:
                    y += 2
                    continue
                width = value / top_value * w
                body.append(
                    f"<path d='M{x0},{y + 2} h{max(width - 3, 1):.1f} a3,3 0 0 1 3,3 v8 a3,3 0 0 1 -3,3 h-{max(width - 3, 1):.1f} z' fill='{color}'/>"
                )
                body.append(text(x0 + width + 8, y + 14, f"{value:.0f}", MUTED, 11, "start"))
                y += 17
            y += 8
        y += 14
    body.append(text(20, y + 4, "두 막대그래프의 눈금은 서로 다르다. llama.cpp의 BF16은 재지 않았다(20.7절).", MUTED, 12, "start"))
    svg(OUT / "result.svg", 860, y + 20, body)


if __name__ == "__main__":
    for figure in (fig_blocks, fig_integer_dot, fig_scaling, fig_result):
        figure()
    print(f"wrote {len(list(OUT.glob('*.svg')))} figures to {OUT}")
