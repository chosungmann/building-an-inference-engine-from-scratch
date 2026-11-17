"""19장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch19.py
출력: book/src/images/ch19/*.svg

측정값은 book/src/part3/output/ch19_*.txt에서 읽는다. 다시 측정하면 그림도
다시 만들어야 한다.
"""

import math
import re

from common import *  # noqa: F403

OUT = IMAGES / "ch19"  # noqa: F405
OUTPUT = IMAGES.parent / "part3/output"  # noqa: F405
GRID = "#e5e7eb"
LIGHT_BLUE = "#93c5fd"


def read(name):
    return (OUTPUT / name).read_text(encoding="utf-8")


def fig_formats():
    body = [text(20, 30, "실수 하나를 담는 세 형식", INK, 14, "start", "bold")]
    unit = 22
    rows = [("f32", 8, 23, "32비트"), ("f16", 5, 10, "16비트. 지수와 가수를 모두 줄였다"), ("BF16", 8, 7, "16비트. f32의 앞 절반")]
    y = 62
    for name, exponent, mantissa, note in rows:
        body.append(text(20, y + 21, name, INK, 13, "start", "bold"))
        x = 80
        for width, fill, stroke, label in [
            (1, ORANGE_BG, ORANGE, ""),
            (exponent, GREEN_BG, GREEN, f"지수 {exponent}"),
            (mantissa, BLUE_BG, BLUE, f"가수 {mantissa}"),
        ]:
            body.append(f"<rect x='{x}' y='{y}' width='{width * unit - 2}' height='32' rx='3' fill='{fill}' stroke='{stroke}'/>")
            if label:
                body.append(text(x + width * unit / 2 - 1, y + 21, label, INK, 12))
            x += width * unit
        body.append(text(x + 10, y + 21, note, MUTED, 12, "start"))
        y += 52
    body.append(f"<rect x='80' y='{y}' width='20' height='14' rx='3' fill='{ORANGE_BG}' stroke='{ORANGE}'/>")
    body.append(text(108, y + 12, "부호 1비트", MUTED, 12, "start"))
    body.append(text(200, y + 12, "칸 하나가 1비트다. 지수는 값의 자릿수를, 가수는 유효 숫자를 담는다(2.7절).", MUTED, 12, "start"))
    svg(OUT / "formats.svg", 860, y + 30, body)


def fig_grid():
    body = [text(20, 30, "실수를 정수 눈금에 맞추기 (3비트)", INK, 14, "start", "bold")]

    def axis(y, title, sub, lo, hi, codes, zero_code, scale_label, values):
        x0, x1 = 90, 770
        out = [text(20, y - 44, title, INK, 13, "start", "bold"), text(20, y - 26, sub, MUTED, 12, "start")]
        out.append(line(x0 - 30, y, x1 + 30, y, MUTED, 1.5))

        def px(v):
            return x0 + (v - lo) / (hi - lo) * (x1 - x0)

        step = (hi - lo) / (len(codes) - 1)
        for i, code in enumerate(codes):
            xx = px(lo + i * step)
            hot = code == zero_code
            out.append(line(xx, y - 9, xx, y + 9, ORANGE if hot else INK, 2))
            out.append(text(xx, y + 28, str(code), ORANGE if hot else INK, 12, "middle", "bold" if hot else "normal"))
        out.append(text(x0 - 40, y + 28, "정수", MUTED, 11, "end"))
        # 실수 0의 자리
        zx = px(0.0)
        out.append(text(zx, y + 46, "실수 0", ORANGE, 11))
        # 눈금 한 칸
        a, b = px(lo + step * (len(codes) - 2)), px(hi)
        out.append(line(a, y + 40, b, y + 40, MUTED, 1))
        out.append(text((a + b) / 2, y + 56, scale_label, MUTED, 11))
        for v in values:
            snapped = lo + round((min(max(v, lo), hi) - lo) / step) * step
            out.append(f"<circle cx='{px(v):.1f}' cy='{y - 20}' r='4' fill='{BLUE}'/>")
            out.append(line(px(v), y - 16, px(snapped), y - 3, BLUE, 1, arrow=True))
        return out

    values = [-0.83, -0.35, 0.12, 0.3, 0.71, 0.95]
    body += axis(120, "대칭", "0을 가운데 둔다. 가장 큰 절댓값이 눈금의 끝이다. 정수 하나(-4)는 쓰지 않는다.", -1.0, 1.0, [-3, -2, -1, 0, 1, 2, 3], 0, "눈금 한 칸 (scale)", values)
    one_sided = [0.05, 0.2, 0.33, 0.52, 0.71, 0.95]
    body += axis(260, "비대칭", "최솟값부터 최댓값까지를 다 쓴다. 실수 0에 대응하는 정수(zero_point)를 따로 적어 둔다.", -0.25, 1.5, [0, 1, 2, 3, 4, 5, 6, 7], 1, "눈금 한 칸 (scale)", one_sided)
    body.append(text(20, 344, "파란 점이 원래 값이고 화살표가 가리키는 눈금이 저장되는 값이다. 둘의 차이가 양자화 오차다.", MUTED, 12, "start"))
    svg(OUT / "grid.svg", 860, 360, body)


def read_uniform():
    text_ = read("ch19_uniform.txt")
    first = [
        tuple(float(v) for v in m.groups())
        for m in re.finditer(r"^\s+(\d+)\s+(\d+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)$", text_, re.M)
    ]
    second = []
    for row in text_.split("2. 눈금의 끝")[1].split("3. 유난히")[0].splitlines()[2:]:
        match = re.match(r"\s*(표준편차 × ([\d.]+)|가장 큰 값)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", row)
        if match:
            second.append((match[2], [float(match[i]) for i in (3, 4, 5, 6)]))
    largest = float(re.search(r"가장 큰 절댓값은 ([\d.]+)", text_)[1])
    return first, second, largest


def read_weights():
    row = [r for r in read("ch19_weights_llama32.txt").split("3. 정수 눈금")[1].splitlines() if r.startswith("전체")][0]
    return [float(v) for v in row.split()[1:]]


def fig_bits():
    first, _, _ = read_uniform()
    weights = read_weights()
    x0, y0, w, h, top = 80, 76, 520, 300, 100
    body = [text(20, 30, "비트의 수와 오차: 비트 하나에 6dB", INK, 14, "start", "bold")]

    def pos(bits, value):
        return x0 + (bits - 2) / 14 * w, y0 + h - value / top * h

    for value in range(0, 101, 20):
        _, yy = pos(2, value)
        body.append(line(x0, yy, x0 + w, yy, GRID, 1))
        body.append(text(x0 - 8, yy + 4, f"{value}dB", MUTED, 11, "end"))
    for bits in (2, 4, 6, 8, 10, 12, 16):
        xx, _ = pos(bits, 0)
        body.append(text(xx, y0 + h + 18, str(bits), MUTED, 11))
    body.append(text(x0 + w / 2, y0 + h + 42, "값 하나에 쓰는 비트", MUTED, 12))
    body.append(text(x0 - 8, y0 - 12, "SQNR (위가 정확하다)", MUTED, 11, "start"))

    def series(points, color, label, dy):
        path = " ".join("{:.1f},{:.1f}".format(*pos(b, v)) for b, v in points)
        out = [f"<polyline points='{path}' fill='none' stroke='{color}' stroke-width='2'/>"]
        for b, v in points:
            px, py = pos(b, v)
            out.append(f"<circle cx='{px:.1f}' cy='{py:.1f}' r='4' fill='{color}' stroke='#ffffff' stroke-width='2'/>")
        px, py = pos(*points[-1])
        out.append(text(px + 12, py + dy, label, INK, 12, "start"))
        return out

    body += series([(r[0], r[2]) for r in first], BLUE, "고르게 퍼진 값", 0)
    body += series([(r[0], r[4]) for r in first], LIGHT_BLUE, "종 모양으로 퍼진 값", 8)
    # 진짜 가중치
    for bits, value, label, dx, anchor in [
        (8, weights[0], "텐서마다", 10, "start"),
        (8, weights[2], "32개마다", 10, "start"),
        (4, weights[3], "텐서마다", 10, "start"),
        (4, weights[5], "32개마다", 10, "start"),
    ]:
        px, py = pos(bits, value)
        body.append(f"<rect x='{px - 5:.1f}' y='{py - 5:.1f}' width='10' height='10' fill='{ORANGE}' stroke='#ffffff' stroke-width='2'/>")
        body.append(text(px + dx, py + (16 if label == "32개마다" else 4), label, ORANGE, 11, anchor))
    body.append(f"<rect x='{x0 + w + 20}' y='{y0 + h - 40}' width='10' height='10' fill='{ORANGE}'/>")
    body.append(text(x0 + w + 36, y0 + h - 31, "Llama 3.2 1B의 가중치", INK, 12, "start"))
    svg(OUT / "bits.svg", 860, y0 + h + 58, body)


def fig_clipping():
    _, second, largest = read_uniform()
    x0, y0, w, h, top = 80, 56, 520, 280, 45
    body = [text(20, 30, "눈금의 끝을 어디에 둘까: 자르는 오차와 칸의 오차", INK, 14, "start", "bold")]

    def pos(limit, value):
        return x0 + (limit - 1) / 4 * w, y0 + h - value / top * h

    for value in range(0, 46, 10):
        _, yy = pos(1, value)
        body.append(line(x0, yy, x0 + w, yy, GRID, 1))
        body.append(text(x0 - 8, yy + 4, f"{value}dB", MUTED, 11, "end"))
    for limit in (1, 2, 3, 4):
        xx, _ = pos(limit, 0)
        body.append(text(xx, y0 + h + 18, f"× {limit}", MUTED, 11))
    xx, _ = pos(largest, 0)
    body.append(line(xx, y0, xx, y0 + h, MUTED, 1, dash="3 4"))
    body.append(text(xx, y0 + h + 18, "가장 큰 값", MUTED, 11))
    body.append(text(x0 + w / 2, y0 + h + 42, "눈금의 끝 (표준편차의 몇 배인가)", MUTED, 12))
    colors = [MUTED, ORANGE, LIGHT_BLUE, BLUE]
    for index, (bits, color) in enumerate(zip((3, 4, 6, 8), colors)):
        points = [(float(limit) if limit else largest, values[index]) for limit, values in second]
        path = " ".join("{:.1f},{:.1f}".format(*pos(*p)) for p in points)
        body.append(f"<polyline points='{path}' fill='none' stroke='{color}' stroke-width='2'/>")
        best = max(points, key=lambda p: p[1])
        px, py = pos(*best)
        body.append(f"<circle cx='{px:.1f}' cy='{py:.1f}' r='5' fill='{color}' stroke='#ffffff' stroke-width='2'/>")
        ex, ey = pos(*points[-1])
        body.append(text(ex + 12, ey + 4, f"{bits}비트 (가장 좋은 끝: × {best[0]:.1f})", INK, 12, "start"))
    body.append(text(x0 + 6, y0 + 14, "← 끝이 가까우면 큰 값이 잘린다", MUTED, 11, "start"))
    body.append(text(x0 + w - 6, y0 + h - 8, "끝이 멀면 칸이 굵어진다 →", MUTED, 11, "end"))
    svg(OUT / "clipping.svg", 880, y0 + h + 58, body)


def fig_granularity():
    weights = read_weights()
    labels = ["텐서 전체에 하나", "행마다 하나", "값 32개마다 하나"]
    x0, w, top = 250, 420, 50.0
    body = [text(20, 28, "눈금을 어느 범위마다 정할까: Llama 3.2 1B의 가중치 전체 (SQNR, 길수록 정확하다)", INK, 14, "start", "bold")]
    y = 54
    for title, values, color in [("8비트", weights[:3], BLUE), ("4비트", weights[3:], ORANGE)]:
        body.append(text(20, y + 16, title, INK, 13, "start", "bold"))
        for label, value in zip(labels, values):
            width = value / top * w
            body.append(text(x0 - 12, y + 15, label, INK, 12, "end"))
            body.append(
                f"<path d='M{x0},{y + 2} h{max(width - 3, 1):.1f} a3,3 0 0 1 3,3 v12 a3,3 0 0 1 -3,3 h-{max(width - 3, 1):.1f} z' fill='{color}'/>"
            )
            body.append(text(x0 + width + 8, y + 16, f"{value:.1f}dB", MUTED, 12, "start"))
            y += 28
        y += 16
    svg(OUT / "granularity.svg", 860, y, body)


def fig_effect():
    rows = []
    for row in read("ch19_effect_llama32.txt").splitlines()[2:]:
        match = re.match(r"(.+?)\s+([\d.]+)\s+([\d.]+)\s+([\d.e-]+)\s+(\".*?\")\s+([\d.]+)%\s+(\d+) / (\d+)$", row)
        if not match:
            break
        rows.append((match[1].strip(), float(match[2]), float(match[4]), float(match[6])))
    x0, w = 200, 420
    body = [text(20, 28, "가중치를 줄이면 출력이 얼마나 달라지는가 (Llama 3.2 1B)", INK, 14, "start", "bold")]
    body.append(text(x0, 52, "점수의 상대 오차 (로그 눈금, 왼쪽이 정확하다)", MUTED, 12, "start"))
    body.append(text(x0 + w + 110, 52, "값당 비트", MUTED, 12, "end"))
    body.append(text(x0 + w + 220, 52, "\" Paris\"의 확률", MUTED, 12, "end"))

    def px(error):
        return x0 + (math.log10(error) + 7) / 7 * w

    top = 66
    bottom = top + len(rows) * 26
    for exponent in range(-7, 1):
        xx = px(10.0**exponent)
        body.append(line(xx, top, xx, bottom, GRID, 1))
        body.append(text(xx, bottom + 16, f"1e{exponent}" if exponent else "1", MUTED, 11))
    y = top
    for label, bits, error, probability in rows:
        broken = probability < 20
        color = ORANGE if broken else BLUE
        body.append(text(x0 - 14, y + 17, label, INK, 12, "end"))
        body.append(line(x0, y + 13, px(error), y + 13, color, 2))
        body.append(f"<circle cx='{px(error):.1f}' cy='{y + 13}' r='5' fill='{color}' stroke='#ffffff' stroke-width='2'/>")
        body.append(text(x0 + w + 110, y + 17, f"{bits:.2f}", INK, 12, "end"))
        body.append(text(x0 + w + 220, y + 17, f"{probability:.1f}%", ORANGE if broken else INK, 12, "end"))
        y += 26
    body.append(text(20, bottom + 42, "주황색은 1위 토큰이 바뀐 경우다. 같은 4비트라도 눈금을 어느 범위마다 정하느냐에 따라 결과가 갈린다.", MUTED, 12, "start"))
    svg(OUT / "effect.svg", 880, bottom + 58, body)


if __name__ == "__main__":
    for figure in (fig_formats, fig_grid, fig_bits, fig_clipping, fig_granularity, fig_effect):
        figure()
    print(f"wrote {len(list(OUT.glob('*.svg')))} figures to {OUT}")
