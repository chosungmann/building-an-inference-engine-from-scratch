"""21장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch21.py
출력: book/src/images/ch21/*.svg

측정값은 book/src/part3/output/ch21_*.txt에서 읽는다. 다시 측정하면 그림도
다시 만들어야 한다.
"""

import math
import re

from common import *  # noqa: F403

OUT = IMAGES / "ch21"  # noqa: F405
OUTPUT = IMAGES.parent / "part3/output"  # noqa: F405
GRID = "#e5e7eb"
LIGHT_BLUE = "#93c5fd"


def read(name):
    return (OUTPUT / name).read_text(encoding="utf-8")


def fig_tokens():
    rows = []
    for row in read("ch21_tokens.txt").splitlines():
        match = re.match(r"\s+(\d+) (\".*?\")\s+([\d.]+)%\s+([\d.]+)$", row)
        if match:
            rows.append((match[2].strip('"'), float(match[3]), float(match[4])))
    mean = float(re.search(r"평균 손실\s+([\d.]+)", read("ch21_tokens.txt"))[1])
    x0, y0, h, step, top = 60, 60, 200, 52, 9.0
    body = [text(20, 30, "토큰마다의 손실: 확률이 낮을수록 길다", INK, 14, "start", "bold")]
    for value in (0, 3, 6, 9):
        yy = y0 + h - value / top * h
        body.append(line(x0, yy, x0 + step * len(rows), yy, GRID, 1))
        body.append(text(x0 - 8, yy + 4, str(value), MUTED, 11, "end"))
    for index, (token, probability, loss) in enumerate(rows):
        x = x0 + index * step + 8
        height = loss / top * h
        color = ORANGE if loss > 3 else BLUE
        body.append(f"<rect x='{x}' y='{y0 + h - height:.1f}' width='{step - 16}' height='{max(height, 1):.1f}' rx='3' fill='{color}'/>")
        body.append(text(x + (step - 16) / 2, y0 + h + 18, token.strip() or "␣", INK, 11))
        label = f"{probability:.0f}%" if probability >= 1 else f"{probability:.2f}%"
        body.append(text(x + (step - 16) / 2, y0 + h - height - 6, label, MUTED, 10))
    yy = y0 + h - mean / top * h
    body.append(line(x0, yy, x0 + step * len(rows), yy, INK, 1.5, dash="5 4"))
    body.append(text(x0 + step * len(rows) + 6, yy + 4, f"평균 {mean:.2f}", INK, 12, "start"))
    body.append(text(20, y0 + h + 46, "막대 위의 숫자는 모델이 그 토큰에 준 확률이다. 손실은 -ln(확률)이고, 평균 손실에 지수 함수를 씌운 것이 퍼플렉서티다.", MUTED, 12, "start"))
    svg(OUT / "tokens.svg", 60 + step * len(rows) + 110, y0 + h + 62, body)


def fig_protocol():
    body = [text(20, 30, "글을 끊어 채점하는 방법", INK, 14, "start", "bold")]
    x0, w = 90, 240
    for index in range(3):
        x = x0 + index * (w + 14)
        body.append(f"<rect x='{x}' y='60' width='{w / 2 - 1}' height='34' rx='3' fill='{GRAY_BG}' stroke='{MUTED}'/>")
        body.append(f"<rect x='{x + w / 2 + 1}' y='60' width='{w / 2 - 1}' height='34' rx='3' fill='{BLUE_BG}' stroke='{BLUE}'/>")
        body.append(f"<rect x='{x}' y='60' width='16' height='34' rx='3' fill='{ORANGE_BG}' stroke='{ORANGE}'/>")
        body.append(text(x + w / 4 + 6, 82, "문맥 256", MUTED, 12))
        body.append(text(x + 3 * w / 4, 82, "채점 255", INK, 12))
        body.append(text(x + w / 2, 114, f"조각 {index + 1}: 토큰 512개", MUTED, 11))
    body.append(text(x0 + 3 * (w + 14) + 4, 82, "…", INK, 16, "start"))
    body.append(f"<rect x='{x0}' y='134' width='16' height='14' rx='3' fill='{ORANGE_BG}' stroke='{ORANGE}'/>")
    body.append(text(x0 + 24, 146, "조각의 첫 토큰은 글머리 토큰으로 바꾼다", MUTED, 12, "start"))
    body.append(text(20, 176, "조각마다 모델에 따로 넣는다. 앞 절반은 흐름을 잡으라고 주는 문맥이고, 뒤 절반의 자리에서 다음 토큰을 맞힌 정도만 센다.", MUTED, 12, "start"))
    svg(OUT / "protocol.svg", 900, 194, body)


def read_table(name):
    rows = []
    for row in read(name).splitlines():
        match = re.match(
            r"(.+?)\s{2,}([\d.]+)\s+([\d.]+) ± ([\d.]+)\s+([\d.]+)\s+([+-][\d.]+) ± ([\d.]+)\s+(\d+)$", row
        )
        if match:
            rows.append(
                {
                    "label": match[1].strip(),
                    "bits": float(match[2]),
                    "ppl": float(match[3]),
                    "loss": float(match[5]),
                    "delta": float(match[6]),
                    "error": float(match[7]),
                }
            )
    return rows


def fig_formats():
    rows = read_table("ch21_formats_llama32.txt")
    base = rows[0]["ppl"]
    x0, w = 250, 380
    low, high = -4.0, 1.0  # 로그 눈금의 범위: 1e-4 ~ 10
    body = [text(20, 28, "가중치를 줄이면 손실이 얼마나 느는가 (Llama 3.2 1B, WikiText-2)", INK, 14, "start", "bold")]
    body.append(text(x0, 52, "원래 모델과의 평균 손실 차이 (로그 눈금, 왼쪽이 좋다)", MUTED, 12, "start"))
    body.append(text(x0 + w + 96, 52, "값당 비트", MUTED, 12, "end"))
    body.append(text(x0 + w + 200, 52, "퍼플렉서티", MUTED, 12, "end"))

    def px(value):
        value = min(max(value, 10**low), 10**high)
        return x0 + (math.log10(value) - low) / (high - low) * w

    shown = rows[1:]
    top = 66
    bottom = top + len(shown) * 26
    for exponent in range(int(low), int(high) + 1):
        xx = px(10.0**exponent)
        body.append(line(xx, top, xx, bottom, GRID, 1))
        label = {0: "1", 1: "10"}.get(exponent, f"1e{exponent}")
        body.append(text(xx, bottom + 16, label, MUTED, 11))
    y = top
    for row in shown:
        delta, error = row["delta"], row["error"]
        zero = delta <= 2 * error or delta <= 0
        broken = delta > 1
        color = ORANGE if broken else (MUTED if zero else BLUE)
        body.append(text(x0 - 14, y + 17, row["label"], INK, 12, "end"))
        if zero:
            body.append(text(x0 + 6, y + 17, "차이가 오차 안에 있다", MUTED, 11, "start"))
        else:
            body.append(line(x0, y + 13, px(delta), y + 13, color, 2))
            lo, hi = px(max(delta - error, 10**low)), px(delta + error)
            body.append(line(lo, y + 13, hi, y + 13, INK, 1))
            body.append(f"<circle cx='{px(delta):.1f}' cy='{y + 13}' r='5' fill='{color}' stroke='#ffffff' stroke-width='2'/>")
        body.append(text(x0 + w + 96, y + 17, f"{row['bits']:.2f}", INK, 12, "end"))
        ppl = f"{row['ppl']:.2f}" if row["ppl"] < 1000 else f"{row['ppl']:,.0f}"
        body.append(text(x0 + w + 200, y + 17, ppl, ORANGE if broken else INK, 12, "end"))
        y += 26
    body.append(text(20, bottom + 42, f"원래 모델의 퍼플렉서티는 {base:.2f}이다. 가는 선은 차이의 표준 오차다. 주황색은 모델이 무너진 경우다.", MUTED, 12, "start"))
    svg(OUT / "formats.svg", 900, bottom + 58, body)


def fig_sensitivity():
    kinds = read_table("ch21_kinds_llama32.txt")[1:-1]
    blocks = read_table("ch21_blocks_llama32.txt")[1:]
    body = [text(20, 28, "하나씩만 4비트로 줄였을 때 늘어나는 손실", INK, 14, "start", "bold")]
    # 왼쪽: 종류별
    x0, w = 130, 250
    top_value = max(row["delta"] + row["error"] for row in kinds) * 1.15
    body.append(text(20, 56, "가중치의 종류마다", INK, 13, "start", "bold"))
    y = 70
    for row in kinds:
        width = max(row["delta"], 0) / top_value * w
        body.append(text(x0 - 10, y + 15, row["label"], INK, 12, "end"))
        body.append(f"<rect x='{x0}' y='{y + 3}' width='{max(width, 1):.1f}' height='16' rx='3' fill='{BLUE}'/>")
        lo = x0 + max(row["delta"] - row["error"], 0) / top_value * w
        hi = x0 + (row["delta"] + row["error"]) / top_value * w
        body.append(line(lo, y + 11, hi, y + 11, INK, 1))
        body.append(text(hi + 8, y + 15, f"{row['delta']:+.3f}", MUTED, 11, "start"))
        y += 26
    left_bottom = y
    # 오른쪽: 블록별
    bx, bw, bh, by = 520, 340, 170, 80
    top_block = max(row["delta"] + row["error"] for row in blocks) * 1.15
    body.append(text(bx - 40, 56, "블록마다", INK, 13, "start", "bold"))
    for fraction in (0, 0.5, 1.0):
        yy = by + bh - fraction * bh
        body.append(line(bx, yy, bx + bw, yy, GRID, 1))
        body.append(text(bx - 6, yy + 4, f"{fraction * top_block:.3f}", MUTED, 10, "end"))
    step = bw / len(blocks)
    for index, row in enumerate(blocks):
        height = max(row["delta"], 0) / top_block * bh
        x = bx + index * step + 3
        body.append(f"<rect x='{x:.1f}' y='{by + bh - height:.1f}' width='{step - 6:.1f}' height='{max(height, 1):.1f}' rx='2' fill='{BLUE}'/>")
        cx = x + (step - 6) / 2
        lo = by + bh - max(row["delta"] - row["error"], 0) / top_block * bh
        hi = by + bh - (row["delta"] + row["error"]) / top_block * bh
        body.append(line(cx, lo, cx, hi, INK, 1))
        body.append(text(cx, by + bh + 16, str(index), MUTED, 10))
    body.append(text(bx + bw / 2, by + bh + 36, "블록의 번호", MUTED, 11))
    bottom = max(left_bottom, by + bh + 46)
    body.append(text(20, bottom + 14, "가는 선은 표준 오차다. 나머지 가중치는 원래 값 그대로 두었다.", MUTED, 12, "start"))
    svg(OUT / "sensitivity.svg", 900, bottom + 30, body)


if __name__ == "__main__":
    done = []
    for figure in (fig_tokens, fig_protocol, fig_formats, fig_sensitivity):
        try:
            figure()
            done.append(figure.__name__)
        except FileNotFoundError as error:
            print(f"건너뜀: {figure.__name__} ({error.filename}이 아직 없다)")
    print(f"wrote {len(done)} figures to {OUT}")
