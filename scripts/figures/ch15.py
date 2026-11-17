"""15장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch15.py
출력: book/src/images/ch15/*.svg

측정값은 book/src/part3/output/ch15_*.txt와 bench/의 기록에서 읽는다.
다시 측정하면 그림도 다시 만들어야 한다.
"""

import math
import re

from common import *  # noqa: F403

OUT = IMAGES / "ch15"  # noqa: F405
ROOT = IMAGES.parents[2]  # noqa: F405
OUTPUT = IMAGES.parent / "part3/output"  # noqa: F405
GRID = "#e5e7eb"


def read_table(name, title):
    """출력 파일에서 `title`로 시작하는 표를 [[칸, …]]으로 읽는다."""
    rows, inside = [], False
    for row in (OUTPUT / name).read_text(encoding="utf-8").splitlines():
        if row.startswith(title):
            inside = True
        elif inside and not row.strip():
            break
        elif inside and re.match(r"\s*\d", row):
            rows.append(row.split())
    return rows


def size_bytes(label):
    return int(label[:-2]) * (1 << 20 if label.endswith("MB") else 1 << 10)


def fig_sizes():
    body = []
    rows = [(size_bytes(r[0]), float(r[1])) for r in read_table("ch15_memory.txt", "배열의 크기")]
    x0, y0, w, h, top = 80, 60, 640, 220, 160.0
    lo, hi = math.log2(rows[0][0]), math.log2(rows[-1][0])
    body.append(text(x0, 30, "배열의 크기에 따른 읽기 속도 (스레드 1개)", INK, 14, "start", "bold"))

    def xpos(size):
        return x0 + (math.log2(size) - lo) / (hi - lo) * w

    for value in range(0, int(top) + 1, 40):
        yy = y0 + h - value / top * h
        body.append(line(x0, yy, x0 + w, yy, GRID, 1))
        body.append(text(x0 - 8, yy + 4, f"{value}", MUTED, 11, "end"))
    body.append(text(x0 - 8, y0 - 14, "GB/초", MUTED, 11, "end"))
    # 캐시의 크기: sysctl hw.perflevel0.l1dcachesize, l2cachesize.
    for size, label in ((128 << 10, "L1 캐시 128KB"), (16 << 20, "L2 캐시 16MB")):
        xx = xpos(size)
        body.append(line(xx, y0 - 4, xx, y0 + h, ORANGE, 1.5, dash="6 4"))
        body.append(text(xx + 6, y0 + 8, label, ORANGE, 11, "start"))
    for size, label in ((16 << 10, "16KB"), (256 << 10, "256KB"), (4 << 20, "4MB"), (64 << 20, "64MB"), (1 << 30, "1GB")):
        body.append(text(xpos(size), y0 + h + 18, label, MUTED, 11))
    points = [(xpos(size), y0 + h - speed / top * h) for size, speed in rows]
    path = " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
    body.append(f"<polyline points='{path}' fill='none' stroke='{BLUE}' stroke-width='2'/>")
    body += [f"<circle cx='{px:.1f}' cy='{py:.1f}' r='3.5' fill='{BLUE}' stroke='#ffffff'/>" for px, py in points]
    body.append(text(x0 + w / 2, y0 + h + 42, "배열의 크기", MUTED, 12))
    svg(OUT / "sizes.svg", 780, y0 + h + 58, body)


def fig_threads():
    body = []
    rows = [(int(r[0]), float(r[1])) for r in read_table("ch15_memory.txt", "스레드의 수")]
    x0, y0, w, h, top = 80, 60, 600, 230, 1100.0
    body.append(text(x0, 30, "스레드의 수에 따른 읽기 속도 (모두 합쳐서)", INK, 14, "start", "bold"))

    def pos(count, speed):
        return x0 + (count - 1) / 15 * w, y0 + h - min(speed, top) / top * h

    for value in range(0, int(top) + 1, 200):
        yy = y0 + h - value / top * h
        body.append(line(x0, yy, x0 + w, yy, GRID, 1))
        body.append(text(x0 - 8, yy + 4, f"{value}", MUTED, 11, "end"))
    body.append(text(x0 - 8, y0 - 14, "GB/초", MUTED, 11, "end"))
    for count, _ in rows:
        body.append(text(pos(count, 0)[0], y0 + h + 18, str(count), MUTED, 11))
    body.append(text(x0 + w / 2, y0 + h + 42, "스레드의 수", MUTED, 12))
    # 스레드마다 첫 스레드만큼 읽는다면.
    one = rows[0][1]
    ex, ey = pos(12, one * 12)
    body.append(line(*pos(1, one), ex, ey, MUTED, 1.5, dash="6 4"))
    body.append(text(ex + 8, ey + 4, "스레드마다 따로 읽을 수 있다면", MUTED, 11, "start"))
    body.append(line(pos(12, 0)[0], y0, pos(12, 0)[0], y0 + h, GRID, 1, dash="3 3"))
    points = [pos(count, speed) for count, speed in rows]
    path = " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
    body.append(f"<polyline points='{path}' fill='none' stroke='{BLUE}' stroke-width='2'/>")
    for (px, py), (count, speed) in zip(points, rows):
        body.append(f"<circle cx='{px:.1f}' cy='{py:.1f}' r='4' fill='{BLUE}' stroke='#ffffff'/>")
        if count in (1, 4, 12, 16):
            body.append(text(px, py - 10, f"{speed:.0f}", BLUE, 11, weight="bold"))
    body.append(text(pos(13, 0)[0] + 6, y0 + h - 8, "성능 코어 12개를 넘음", MUTED, 10, "start"))
    svg(OUT / "threads.svg", 820, y0 + h + 58, body)


def fig_budget():
    """토큰 생성과 프롬프트 처리가 가중치를 읽는 방식."""
    body = []
    body.append(text(20, 30, "가중치를 한 번 읽어서 하는 일", INK, 14, "start", "bold"))

    def weights(x, y, label):
        out = [f"<rect x='{x}' y='{y}' width='300' height='50' rx='6' fill='{GRAY_BG}' stroke='{MUTED}'/>"]
        out.append(text(x + 150, y + 30, label, INK, 13))
        return out

    # 토큰 생성
    y = 70
    body.append(text(20, y + 30, "토큰 생성", INK, 13, "start", "bold"))
    body += weights(130, y, "가중치 4.9GB (출력 층 포함)")
    body.append(line(432, y + 25, 478, y + 25, INK, 1.5, arrow=True))
    body.append(f"<rect x='484' y='{y + 8}' width='34' height='34' rx='5' fill='{BLUE_BG}' stroke='{BLUE}'/>")
    body.append(text(530, y + 22, "토큰 하나", INK, 12, "start"))
    body.append(text(530, y + 40, "읽은 바이트마다 곱셈 0.25번", MUTED, 12, "start"))
    # 프롬프트 처리
    y = 160
    body.append(text(20, y + 30, "프롬프트 처리", INK, 13, "start", "bold"))
    body += weights(130, y, "가중치 3.9GB (블록만)")
    body.append(line(432, y + 25, 478, y + 25, INK, 1.5, arrow=True))
    for i in range(6):
        body.append(f"<rect x='{484 + i * 14}' y='{y + 8 + i * 0}' width='34' height='34' rx='5' fill='{ORANGE_BG}' stroke='{ORANGE}' opacity='{1 - i * 0.12:.2f}'/>")
    body.append(text(606, y + 22, "토큰 256개", INK, 12, "start"))
    body.append(text(606, y + 40, "읽은 바이트마다 곱셈 64번", MUTED, 12, "start"))
    body.append(text(20, y + 88, "토큰을 하나씩 만들 때는 읽어 온 가중치를 한 번 쓰고 버린다. 묶어서 처리하면 같은 가중치를 토큰 수만큼 다시 쓴다.", MUTED, 12, "start"))
    svg(OUT / "budget.svg", 820, y + 106, body)


def speeds():
    """13장과 14장의 측정에서 (프롬프트 처리, 토큰 생성) 토큰/초를 읽는다."""
    speed = lambda test, text_: float(re.search(rf"{test}\s+([\d.]+)", text_)[1])  # noqa: E731
    badak = (ROOT / "bench/baseline/badak_llama32_ch14.txt").read_text(encoding="utf-8")
    result = {"badak": (speed("pp256", badak), speed("tg32", badak))}
    sections = (ROOT / "bench/ch13/llama_cpp_llama32.txt").read_text(encoding="utf-8").split("## ")[1:]
    for key, section in zip(("llama1", "llama12"), sections):
        result[key] = (speed(r"pp256 \|", section), speed(r"tg32 \|", section))
    return result


def fig_roofline():
    body = []
    # ch15_budget의 기본값과 같은 숫자다.
    peak, band1, band12, threads = 19.7, 83.0, 275.0, 12
    prompt_mul, gen_mul = 0.973, 1.236  # 토큰 하나의 곱셈 (십억 번)
    x0, y0, w, h = 90, 60, 620, 320
    xlo, xhi, ylo, yhi = -1.0, 2.5, 0.0, 3.0
    body.append(text(x0, 30, "Llama 3.2 1B와 이 기계의 두 한계", INK, 14, "start", "bold"))

    def pos(intensity, rate):
        return (
            x0 + (math.log10(intensity) - xlo) / (xhi - xlo) * w,
            y0 + h - (math.log10(rate) - ylo) / (yhi - ylo) * h,
        )

    for exponent in range(0, 4):
        _, yy = pos(1, 10**exponent)
        body.append(line(x0, yy, x0 + w, yy, GRID, 1))
        body.append(text(x0 - 8, yy + 4, f"{10**exponent:,}", MUTED, 11, "end"))
    for value in (0.1, 1, 10, 100):
        xx, _ = pos(value, 1)
        body.append(line(xx, y0, xx, y0 + h, GRID, 1))
        body.append(text(xx, y0 + h + 18, f"{value:g}", MUTED, 11))
    body.append(text(x0 - 8, y0 - 14, "십억 곱셈/초", MUTED, 11, "end"))
    body.append(text(x0 + w / 2, y0 + h + 42, "산술 강도: 읽은 바이트 하나로 하는 곱셈의 수", MUTED, 12))

    # 지붕: 메모리의 한계(기울어진 부분)와 계산의 한계(평평한 부분).
    for count, band, color, label in ((1, band1, BLUE, "스레드 1개"), (threads, band12, ORANGE, f"스레드 {threads}개")):
        compute = peak * count
        knee = compute / band
        a, b, c = pos(10**xlo, band * 10**xlo), pos(knee, compute), pos(10**xhi, compute)
        body.append(
            f"<polyline points='{a[0]:.1f},{a[1]:.1f} {b[0]:.1f},{b[1]:.1f} {c[0]:.1f},{c[1]:.1f}' fill='none' stroke='{color}' stroke-width='2.5'/>"
        )
        body.append(text(c[0] + 8, c[1] + 4, label, color, 12, "start", "bold"))

    # 두 가지 일의 자리.
    for intensity, name in ((0.25, "토큰 생성"), (0.25 * 256, "프롬프트 처리 (256토큰)")):
        xx, _ = pos(intensity, 1)
        body.append(line(xx, y0, xx, y0 + h, MUTED, 1, dash="5 4"))
        body.append(text(xx, y0 - 8, name, MUTED, 11))

    measured = speeds()
    marks = [
        ("badak", measured["badak"], GREEN),
        ("llama.cpp, 1개", measured["llama1"], BLUE),
        ("llama.cpp, 12개", measured["llama12"], ORANGE),
    ]
    for label, (prompt, generation), color in marks:
        for intensity, rate, anchor in ((0.25, generation * gen_mul, "end"), (64, prompt * prompt_mul, "start")):
            px, py = pos(intensity, rate)
            body.append(f"<circle cx='{px:.1f}' cy='{py:.1f}' r='6' fill='{color}' stroke='#ffffff' stroke-width='2'/>")
            dx = -12 if anchor == "end" else 12
            # 선과 겹치지 않게 프롬프트 쪽 글자는 점의 아래에 둔다.
            dy = 4 if anchor == "end" else 20
            body.append(text(px + dx, py + dy, f"{label}: {rate:.1f}", color, 11, anchor))
    body.append(text(x0, y0 + h + 68, "선: 이 장에서 잰 한계. 왼쪽 기울어진 부분은 메모리가, 오른쪽 평평한 부분은 계산이 막는다. 점: 13장과 14장에서 잰 속도.", MUTED, 12, "start"))
    svg(OUT / "roofline.svg", 880, y0 + h + 86, body)


if __name__ == "__main__":
    for figure in (fig_sizes, fig_threads, fig_budget, fig_roofline):
        figure()
    print(f"wrote {len(list(OUT.glob('*.svg')))} figures to {OUT}")
