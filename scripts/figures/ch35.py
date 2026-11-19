"""35장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch35.py
출력: book/src/images/ch35/*.svg

breakdown.svg 는 bench/ch35/ 에 남긴 측정값의 중앙값을 그린다. 그 파일이
없으면 그 그림은 건너뛴다. 같은 디렉터리의 JPEG 그림들(progress, steps,
guidance)은 이 스크립트가 아니라 ch35_generate 예제가 만든 그림을 줄여
이어 붙인 것이다(본문 35.9절).
"""

import re
import statistics

from common import *  # noqa: F403

OUT = IMAGES / "ch35"  # noqa: F405
BENCH = IMAGES.parents[2] / "bench/ch35"  # noqa: F405
PURPLE, PURPLE_BG = "#7c3aed", "#ede9fe"


def rect(x, y, w, h, fill, stroke=None, rx=0, width=1):
    edge = f" stroke='{stroke}' stroke-width='{width}'" if stroke else ""
    return f"<rect x='{x:.1f}' y='{y:.1f}' width='{w:.1f}' height='{h:.1f}' rx='{rx}' fill='{fill}'{edge}/>"


def fig_pipeline():
    """부품 셋과 반복문."""
    body = [text(20, 28, "글에서 그림까지: 부품 셋과 반복문 하나", INK, 14, "start", "bold")]  # noqa: F405
    # 글의 길.
    body += box(20, 60, 120, 50, "글", "a watercolor…", GRAY_BG, MUTED)  # noqa: F405
    body += box(180, 60, 150, 50, "텍스트 인코더", "CLIP, 한 번", BLUE_BG, BLUE)  # noqa: F405
    body += box(370, 60, 130, 50, "벡터 77개", "(77, 768)", GRAY_BG, MUTED)  # noqa: F405
    body.append(line(142, 85, 177, 85, INK, 1.5, True))  # noqa: F405
    body.append(line(332, 85, 367, 85, INK, 1.5, True))  # noqa: F405
    # 반복문.
    body.append(rect(160, 150, 560, 150, "#ffffff", ORANGE, 10, 1.5))  # noqa: F405
    body.append(text(176, 172, "걸음마다 되풀이 (20번)", ORANGE, 12, "start", "bold"))  # noqa: F405
    body += box(20, 200, 120, 50, "잡음", "(64, 64, 4)", GRAY_BG, MUTED)  # noqa: F405
    body += box(180, 200, 150, 50, "U-Net × 2", "글을 주고, 안 주고", ORANGE_BG, ORANGE)  # noqa: F405
    body += box(370, 200, 130, 50, "가이던스", "두 짐작을 섞는다", ORANGE_BG, ORANGE)  # noqa: F405
    body += box(540, 200, 160, 50, "스케줄러", "잡음을 조금 걷는다", ORANGE_BG, ORANGE)  # noqa: F405
    body.append(line(142, 225, 177, 225, INK, 1.5, True))  # noqa: F405
    body.append(line(332, 225, 367, 225, INK, 1.5, True))  # noqa: F405
    body.append(line(502, 225, 537, 225, INK, 1.5, True))  # noqa: F405
    # 글의 벡터가 U-Net으로.
    body.append(f"<path d='M435,112 L435,135 L255,135 L255,196' fill='none' stroke='{BLUE}' stroke-width='1.5' marker-end='url(#arrow-{BLUE[1:]})'/>")  # noqa: F405
    body.append(text(345, 130, "교차 어텐션의 키와 값", BLUE, 11))  # noqa: F405
    # 되돌아가는 길.
    body.append(f"<path d='M620,252 L620,280 L255,280 L255,254' fill='none' stroke='{ORANGE}' stroke-width='1.5' stroke-dasharray='5,4' marker-end='url(#arrow-{ORANGE[1:]})'/>")  # noqa: F405
    body.append(text(438, 275, "조금 덜 흐린 잠재 그림", ORANGE, 11))  # noqa: F405
    # 디코더.
    body += box(760, 200, 130, 50, "VAE 디코더", "한 번", GREEN_BG, GREEN)  # noqa: F405
    body += box(930, 200, 120, 50, "그림", "(512, 512, 3)", GRAY_BG, MUTED)  # noqa: F405
    body.append(line(702, 225, 757, 225, INK, 1.5, True))  # noqa: F405
    body.append(line(892, 225, 927, 225, INK, 1.5, True))  # noqa: F405
    body.append(text(20, 330, "U-Net은 가로세로가 8분의 1인 잠재 그림에서 일한다. 20걸음이면 U-Net을 40번 부른다. 시간의 대부분이 거기에 든다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "pipeline.svg", 1070, 348, body)  # noqa: F405


def fig_unet():
    """U-Net의 모양."""
    body = [text(20, 28, "U-Net: 줄이며 내려갔다가 늘리며 올라온다", INK, 14, "start", "bold")]  # noqa: F405
    levels = [("64 × 64", 320, 4096), ("32 × 32", 640, 1024), ("16 × 16", 1280, 256), ("8 × 8", 1280, 64)]
    left, right, top, gap = 150, 610, 70, 62
    for index, (size, channels, cells) in enumerate(levels):
        y = top + index * gap
        height = 40
        width = 60 + channels / 1280 * 110
        body.append(text(20, y + 25, size, INK, 13, "start", "bold"))  # noqa: F405
        body.append(text(84, y + 25, f"자리 {cells:,}", MUTED, 11, "start"))  # noqa: F405
        attention = index < 3
        fill, stroke = (ORANGE_BG, ORANGE) if attention else (BLUE_BG, BLUE)  # noqa: F405
        body.append(rect(left + index * 26, y, width, height, fill, stroke, 6, 1.5))
        body.append(text(left + index * 26 + width / 2, y + 25, f"채널 {channels:,}", INK, 12))  # noqa: F405
        body.append(rect(right - index * 26, y, width, height, fill, stroke, 6, 1.5))
        body.append(text(right - index * 26 + width / 2, y + 25, f"채널 {channels:,}", INK, 12))  # noqa: F405
        # 건너뛰는 길.
        body.append(line(left + index * 26 + width + 4, y + 20, right - index * 26 - 6, y + 20, MUTED, 1.5, True, "5,4"))  # noqa: F405
        if index < 3:
            x_down = left + index * 26 + 40
            body.append(line(x_down, y + height + 2, x_down + 20, y + gap - 4, INK, 1.5, True))  # noqa: F405
            x_up = right - index * 26 + 40
            body.append(line(x_up - 20, y + gap - 2, x_up, y + height + 4, INK, 1.5, True))  # noqa: F405
    y_mid = top + 4 * gap
    body.append(rect(380, y_mid - 10, 170, 40, ORANGE_BG, ORANGE, 6, 1.5))  # noqa: F405
    body.append(text(465, y_mid + 15, "가운데 (8 × 8)", INK, 12))  # noqa: F405
    body.append(line(left + 3 * 26 + 60, top + 3 * gap + 42, 378, y_mid + 6, INK, 1.5, True))  # noqa: F405
    body.append(line(552, y_mid + 6, right - 3 * 26 + 60, top + 3 * gap + 44, INK, 1.5, True))  # noqa: F405
    body.append(text(left, top - 12, "내려간다", MUTED, 12, "start"))  # noqa: F405
    body.append(text(right + 170, top - 12, "올라온다", MUTED, 12, "end"))  # noqa: F405
    body.append(text(430, top + 12, "건너뛰는 길: 이어 붙인다", MUTED, 11))  # noqa: F405
    note = y_mid + 58
    body.append(rect(20, note - 11, 14, 12, ORANGE_BG, ORANGE, 3))  # noqa: F405
    body.append(text(42, note, "잔차 블록 + 어텐션(자리들끼리, 그리고 글과)", INK, 12, "start"))  # noqa: F405
    body.append(rect(330, note - 11, 14, 12, BLUE_BG, BLUE, 3))  # noqa: F405
    body.append(text(352, note, "잔차 블록만", INK, 12, "start"))  # noqa: F405
    body.append(text(20, note + 22, "크기가 절반이 되면 자리는 4분의 1이 되고 채널은 1,280까지 두 배씩 는다. 입력과 출력은 둘 다 (64, 64, 4)다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "unet.svg", 800, note + 38, body)  # noqa: F405


def medians():
    """bench/ch35/ 의 줄들을 모아 중앙값으로."""
    values = {}
    for path in sorted(BENCH.glob("badak_*.txt")) + sorted(BENCH.glob("python_*.txt")):
        key = None
        for row in path.read_text(encoding="utf-8").splitlines():
            found = re.match(r"(unet|vae) +(.+?) +스레드 +(\d+) 중앙값 +([\d.]+)초", row)
            if found:
                key = (found[1], found[2].strip(), int(found[3]))
                values.setdefault(key, []).append(float(found[4]))
                continue
            found = re.match(r"  (unet|vae) +(\w+) +(\d+)번 +([\d.]+)ms", row)
            if found and key:
                values.setdefault((*key, found[2]), []).append(float(found[4]))
    return {key: statistics.median(samples) for key, samples in values.items()}


def fig_breakdown(values):
    """U-Net 한 번의 시간이 어디에 드는가."""
    body = [text(20, 28, "U-Net 한 번의 시간이 드는 곳 (스레드 12개, 밀리초)", INK, 14, "start", "bold")]  # noqa: F405
    parts = [("conv", "3 × 3 합성곱", BLUE), ("linear", "선형 층, 1 × 1 합성곱", GREEN), ("attention", "어텐션 안의 곱과 소프트맥스", ORANGE), ("group_norm", "그룹 정규화", PURPLE)]  # noqa: F405
    x_legend = 20
    for _, label, color in parts:
        body.append(rect(x_legend, 44, 14, 10, color, None, 2))
        body.append(text(x_legend + 20, 54, label, INK, 12, "start"))  # noqa: F405
        x_legend += 50 + 13 * len(label)
    scale = 560 / max(values[("unet", name, 12)] for name in ("기본", "빠른 길")) / 1000
    for row, (name, label) in enumerate((("기본", "기본 (NEON)"), ("빠른 길", "빠른 길 (SME)"))):
        y = 78 + row * 44
        body.append(text(130, y + 19, label, INK, 12, "end"))  # noqa: F405
        x = 140
        total = values[("unet", name, 12)] * 1000
        known = 0
        for part, _, color in parts:
            value = values.get(("unet", name, 12, part), 0)
            known += value
            w = value * scale
            body.append(rect(x, y, max(w - 2, 1), 28, color, None, 3))
            if w > 34:
                body.append(text(x + w / 2 - 1, y + 19, f"{value:.0f}", "#ffffff", 12))  # noqa: F405
            x += w
        rest = max(total - known, 0)
        body.append(rect(x, y, max(rest * scale - 2, 1), 28, "#d1d5db", None, 3))
        body.append(text(140 + total * scale + 8, y + 19, f"합 {total:.0f}", INK, 12, "start"))  # noqa: F405
    body.append(text(20, 186, "회색은 나머지(창 늘어놓기를 뺀 복사, 더하기, 층 정규화, GELU 등). 창을 늘어놓는 시간은 합성곱에 들어 있다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "breakdown.svg", 860, 202, body)  # noqa: F405


if __name__ == "__main__":
    fig_pipeline()
    fig_unet()
    if list(BENCH.glob("badak_*.txt")):
        fig_breakdown(medians())
