"""46장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch46.py
출력: book/src/images/ch46/*.svg

ste.svg 를 뺀 그림들은 bench/ch46/ 에 남긴 측정값을 그린다.
"""

import math
import statistics

from common import *  # noqa: F403

OUT = IMAGES / "ch46"  # noqa: F405
BENCH = IMAGES.parents[2] / "bench/ch46"  # noqa: F405
PURPLE, PURPLE_BG, RED = "#7c3aed", "#ede9fe", "#dc2626"


def rect(x, y, w, h, fill, stroke=None, rx=0, width=1, dash=None):
    edge = f" stroke='{stroke}' stroke-width='{width}'" if stroke else ""
    edge += f" stroke-dasharray='{dash}'" if dash else ""
    return f"<rect x='{x:.1f}' y='{y:.1f}' width='{w:.1f}' height='{h:.1f}' rx='{rx}' fill='{fill}'{edge}/>"


def path(points, stroke, width=2, dash=None):
    d = " ".join(f"{'M' if index == 0 else 'L'}{x:.1f},{y:.1f}" for index, (x, y) in enumerate(points))
    extra = f" stroke-dasharray='{dash}'" if dash else ""
    return f"<path d='{d}' fill='none' stroke='{stroke}' stroke-width='{width}'{extra}/>"


def lines(name):
    file = BENCH / name
    return [row.split() for row in file.read_text(encoding="utf-8").splitlines() if row.strip()] if file.exists() else []


def rounds(prefix):
    out = []
    for file in sorted(BENCH.glob(f"{prefix}_*.txt")):
        out += [row.split() for row in file.read_text(encoding="utf-8").splitlines() if row.strip()]
    return out


def number(parts, key):
    return float(parts[parts.index(key) + 1])


def fig_ste():
    """순전파는 계단으로, 기울기는 곧은 선으로."""
    body = [text(20, 28, "흉내 양자화: 앞으로 갈 때는 계단, 뒤로 올 때는 곧은 선", INK, 14, "start", "bold")]  # noqa: F405
    for panel, (title, note) in enumerate(
        (
            ("순전파: 줄였다 되돌린 값을 쓴다", "가중치가 조금 움직여도 결과는 그대로다"),
            ("역전파: 계단을 곧은 선으로 친다", "기울기를 줄이지 않은 가중치에 그대로 준다"),
        )
    ):
        x0, y0, w, h = 60 + panel * 440, 60, 340, 220
        body.append(text(x0 + w / 2, y0 - 10, title, INK, 13, "middle", "bold"))  # noqa: F405
        body.append(line(x0, y0 + h / 2, x0 + w, y0 + h / 2, MUTED, 1))  # noqa: F405
        body.append(line(x0 + w / 2, y0, x0 + w / 2, y0 + h, MUTED, 1))  # noqa: F405
        body.append(text(x0 + w, y0 + h / 2 + 18, "가중치", MUTED, 11, "end"))  # noqa: F405
        body.append(text(x0 + w / 2 + 6, y0 + 12, "쓰는 값", MUTED, 11, "start"))  # noqa: F405
        steps = 8
        # 곧은 선.
        straight = [(x0, y0 + h), (x0 + w, y0)]
        # 계단.
        stairs = []
        for step in range(steps):
            left = x0 + w * step / steps
            right = x0 + w * (step + 1) / steps
            level = y0 + h - h * (step + 0.5) / steps
            stairs += [(left, level), (right, level)]
        if panel == 0:
            body.append(path(straight, MUTED, 1, "4,3"))  # noqa: F405
            body.append(path(stairs, ORANGE, 2.5))  # noqa: F405
        else:
            body.append(path(stairs, MUTED, 1, "4,3"))  # noqa: F405
            body.append(path(straight, GREEN, 2.5))  # noqa: F405
        body.append(text(x0 + w / 2, y0 + h + 24, note, MUTED, 12))  # noqa: F405
    body.append(text(20, 334, "계단의 기울기는 어디서나 0이다. 그대로 미분하면 아무것도 배우지 못한다. 그래서 뒤로 올 때만 계단이 없는 셈 친다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "ste.svg", 900, 350, body)  # noqa: F405


def quality():
    """모델마다 f32 손실, Q4_0 손실, 블록의 수."""
    data = {}
    for parts in lines("quality.txt"):
        if parts[0].startswith("smollm2_"):
            data[parts[0]] = {"blocks": int(parts[1]), "f32": float(parts[2]), "q4": float(parts[6])}
    return data


def fig_qat(data):
    """4비트로 줄였을 때의 손실."""
    body = [text(20, 28, "SmolLM2-135M: 4비트(Q4_0)로 줄였을 때 잃는 것", INK, 14, "start", "bold")]  # noqa: F405
    body.append(rect(20, 44, 14, 10, BLUE))  # noqa: F405
    body.append(text(40, 54, "파일에 적힌 그대로 (f32)", INK, 12, "start"))  # noqa: F405
    body.append(rect(230, 44, 14, 10, ORANGE))  # noqa: F405
    body.append(text(250, 54, "엔진이 Q4_0으로 줄인 것", INK, 12, "start"))  # noqa: F405
    marks = [
        ("smollm2_135m", "받은 그대로"),
        ("smollm2_135m_ft", "그냥 더 학습"),
        ("smollm2_135m_qat", "양자화를 흉내 내며 학습"),
        ("smollm2_135m_qat_kd", "흉내 내며, 원래 모델을 따라"),
    ]
    low, high = 2.5, 3.4
    scale = 520 / (high - low)
    y = 76
    for name, label in marks:
        if name not in data:
            continue
        entry = data[name]
        body.append(text(230, y + 22, label, INK, 12, "end"))  # noqa: F405
        for offset, key, color in ((0, "f32", BLUE), (18, "q4", ORANGE)):  # noqa: F405
            value = entry[key]
            body.append(rect(240, y + offset, (value - low) * scale, 15, color))
            body.append(text(240 + (value - low) * scale + 6, y + offset + 12, f"{value:.3f}", INK, 11, "start"))  # noqa: F405
        gap = entry["q4"] - entry["f32"]
        body.append(text(850, y + 22, f"{gap:+.3f}", RED if gap > 0.1 else GREEN, 13, "end", "bold"))  # noqa: F405
        y += 46
    body.append(text(850, 68, "줄여서 잃은 것", MUTED, 11, "end"))  # noqa: F405
    base = data["smollm2_135m"]["f32"]
    x = 240 + (base - low) * scale
    body.append(line(x, 70, x, y - 6, MUTED, 1, False, "4,3"))  # noqa: F405
    body.append(text(x + 4, y + 8, f"원래 모델 {base:.3f}", MUTED, 11, "start"))  # noqa: F405
    body.append(text(20, y + 30, "막대는 WikiText-2 시험 글의 평균 손실이다(작을수록 좋다). 가로축은 2.5에서 시작한다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "qat.svg", 880, y + 46, body)  # noqa: F405


def fig_prune():
    """깎은 비율에 따른 손실과 크기."""
    loss = {number(p, "깎은비율"): number(p, "차이") for p in lines("prune_loss.txt")}
    size = {number(p, "깎은비율"): (number(p, "그대로"), number(p, "gzip"), number(p, "희소형식")) for p in lines("prune_size.txt")}
    if not loss or not size:
        return
    body = [text(20, 28, "가중치를 0으로 만들면: 품질은 바로 떨어지고, 크기는 형식을 바꿔야 준다", INK, 14, "start", "bold")]  # noqa: F405
    fractions = sorted(loss)
    # 왼쪽: 손실.
    x0, y0, w, h = 70, 70, 340, 230
    body.append(text(x0 + w / 2, y0 - 14, "늘어난 손실 (다시 학습하지 않았을 때)", INK, 12, "middle", "bold"))  # noqa: F405
    top = 5.0
    for tick in range(0, 6):
        y = y0 + h - h * tick / top
        body.append(line(x0, y, x0 + w, y, GRAY_BG, 1))  # noqa: F405
        body.append(text(x0 - 6, y + 4, f"+{tick}", MUTED, 11, "end"))  # noqa: F405
    points = [(x0 + w * f / 90, y0 + h - h * min(loss[f], top) / top) for f in fractions if loss[f] <= top + 0.5]
    body.append(path(points, ORANGE, 2.5))  # noqa: F405
    for f in fractions:
        if loss[f] <= top:
            body.append(f"<circle cx='{x0 + w * f / 90:.1f}' cy='{y0 + h - h * loss[f] / top:.1f}' r='3.5' fill='{ORANGE}'/>")  # noqa: F405
    for f in (0, 30, 60, 90):
        body.append(text(x0 + w * f / 90, y0 + h + 16, f"{f:.0f}%", MUTED, 11))  # noqa: F405
    q4 = 0.3296
    yq = y0 + h - h * q4 / top
    body.append(line(x0, yq, x0 + w, yq, BLUE, 1.5, False, "5,4"))  # noqa: F405
    body.append(text(x0 + w - 4, yq - 5, "Q4_0으로 줄였을 때 (+0.33)", BLUE, 11, "end"))  # noqa: F405
    body.append(text(x0 + w * 60 / 90 + 8, y0 + 14, "60%: +4.9", ORANGE, 11, "start"))  # noqa: F405
    # 오른쪽: 크기.
    x1 = 500
    body.append(text(x1 + w / 2, y0 - 14, "블록의 행렬들이 차지하는 크기 (MB)", INK, 12, "middle", "bold"))  # noqa: F405
    top_mb = 650
    for tick in range(0, 7):
        y = y0 + h - h * tick * 100 / top_mb
        body.append(line(x1, y, x1 + w, y, GRAY_BG, 1))  # noqa: F405
        body.append(text(x1 - 6, y + 4, f"{tick * 100}", MUTED, 11, "end"))  # noqa: F405
    series = [(0, "16비트로 그대로", MUTED), (1, "그것을 gzip으로", GREEN), (2, "희소 형식 (값 4바이트 + 열 2바이트)", PURPLE)]  # noqa: F405
    for index, label, color in series:
        points = [(x1 + w * f / 90, y0 + h - h * size[f][index] / 1e6 / top_mb) for f in fractions]
        body.append(path(points, color, 2.5))
        lx, ly = points[0]
        body.append(text(lx + 8, ly - 7 if index != 1 else ly + 16, label, color, 11, "start"))
    q4_mb = 106168320 * 4.5 / 8 / 1e6
    yq = y0 + h - h * q4_mb / top_mb
    body.append(line(x1, yq, x1 + w, yq, BLUE, 1.5, False, "5,4"))  # noqa: F405
    body.append(text(x1 + 4, yq + 14, f"Q4_0 ({q4_mb:.0f}MB)", BLUE, 11, "start"))  # noqa: F405
    for f in (0, 30, 60, 90):
        body.append(text(x1 + w * f / 90, y0 + h + 16, f"{f:.0f}%", MUTED, 11))  # noqa: F405
    body.append(text(20, 340, "가로축은 0으로 만든 가중치의 비율이다. 희소 형식은 67%를 넘게 깎아야 16비트로 그대로 적은 것보다 작아진다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "prune.svg", 900, 356, body)  # noqa: F405


def fig_kernel():
    """희소 커널이 보통의 커널을 이기는 자리."""
    data = rounds("kernel")
    if not data:
        return
    body = [text(20, 28, "희소 커널은 얼마나 깎아야 보통의 커널을 이기는가 (1536 × 576 행렬)", INK, 14, "start", "bold")]  # noqa: F405
    x0, y0, w, h = 80, 60, 560, 220
    fractions = [0, 50, 70, 80, 90, 95, 98]
    top = 3.0
    for tick in (0.5, 1.0, 2.0, 3.0):
        y = y0 + h - h * tick / top
        body.append(line(x0, y, x0 + w, y, INK if tick == 1.0 else GRAY_BG, 1.5 if tick == 1.0 else 1))  # noqa: F405
        body.append(text(x0 - 6, y + 4, f"{tick:g}배", MUTED, 11, "end"))  # noqa: F405
    for index, fraction in enumerate(fractions):
        body.append(text(x0 + w * index / (len(fractions) - 1), y0 + h + 16, f"{fraction}%", MUTED, 11))  # noqa: F405
    for count, label, color in ((1, "벡터 하나 (토큰을 쓸 때)", ORANGE), (64, "벡터 64개 (프롬프트를 읽을 때)", BLUE)):  # noqa: F405
        points = []
        for index, fraction in enumerate(fractions):
            found = [p for p in data if p[1] == "1536x576" and int(number(p, "벡터")) == count and int(number(p, "깎은비율")) == fraction]
            ratio = statistics.median(number(p, "보통") for p in found) / statistics.median(number(p, "희소") for p in found)
            points.append((x0 + w * index / (len(fractions) - 1), y0 + h - h * min(ratio, top) / top, ratio))
        body.append(path([(x, y) for x, y, _ in points], color, 2.5))
        for x, y, ratio in points:
            body.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='3.5' fill='{color}'/>")
            body.append(text(x, y - 8, f"{ratio:.2f}", color, 10))
        body.append(rect(680, 96 + (0 if count == 1 else 24), 14, 10, color))
        body.append(text(700, 106 + (0 if count == 1 else 24), label, INK, 12, "start"))  # noqa: F405
    body.append(text(680, 160, "1배의 선 위쪽이", MUTED, 12, "start"))  # noqa: F405
    body.append(text(680, 178, "희소 커널이 빠른 곳", MUTED, 12, "start"))  # noqa: F405
    body.append(text(20, 318, "세로축은 보통의 커널이 걸린 시간을 희소 커널이 걸린 시간으로 나눈 것이다. 가로축은 0으로 만든 가중치의 비율이다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "kernel.svg", 920, 334, body)  # noqa: F405


def fig_blocks():
    """블록을 하나씩 뺐을 때 늘어나는 손실."""
    single = {int(number(p, "하나빼기")): number(p, "차이") for p in lines("prune_blocks.txt") if "하나빼기" in p}
    if not single:
        return
    body = [text(20, 28, "SmolLM2-135M의 블록을 하나씩 뺐을 때 늘어나는 손실", INK, 14, "start", "bold")]  # noqa: F405
    picked = {3, 4, 5, 7, 12, 13, 14, 15, 17, 25}
    x0, y0, w, h = 60, 50, 780, 200
    top = 3.0
    for tick in range(0, 4):
        y = y0 + h - h * tick / top
        body.append(line(x0, y, x0 + w, y, GRAY_BG, 1))  # noqa: F405
        body.append(text(x0 - 6, y + 4, f"+{tick}", MUTED, 11, "end"))  # noqa: F405
    bar = w / len(single)
    for index in sorted(single):
        value = single[index]
        height = h * min(value, top) / top
        color = GREEN if index in picked else MUTED  # noqa: F405
        body.append(rect(x0 + index * bar + 3, y0 + h - height, bar - 6, height, color))
        body.append(text(x0 + index * bar + bar / 2, y0 + h + 14, str(index), MUTED, 10))  # noqa: F405
        if value > top:
            body.append(text(x0 + index * bar + bar / 2 + 12, y0 + 12, f"+{value:.1f}", INK, 11, "start"))  # noqa: F405
    body.append(rect(60, 286, 14, 10, GREEN))  # noqa: F405
    body.append(text(80, 296, "빼도 덜 아픈 열 개. 46.6절에서 이것들을 덜어 낸다", INK, 12, "start"))  # noqa: F405
    body.append(text(20, 320, "맨 앞의 둘과 맨 뒤의 둘은 뺄 수 없다. 가운데의 블록들은 하나쯤 없어도 손실이 0.1에서 0.2 는다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "blocks.svg", 880, 336, body)  # noqa: F405


if __name__ == "__main__":
    fig_ste()
    fig_prune()
    fig_blocks()
    fig_kernel()
    found = quality()
    if "smollm2_135m" in found:
        fig_qat(found)
