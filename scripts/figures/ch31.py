"""31장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch31.py
출력: book/src/images/ch31/*.svg
"""

import re
import statistics

from common import *  # noqa: F403

OUT = IMAGES / "ch31"  # noqa: F405
BENCH = IMAGES.parents[2] / "bench/ch31"  # noqa: F405
PURPLE, PURPLE_BG = "#7c3aed", "#ede9fe"


def rect(x, y, w, h, fill, stroke=None, rx=0, width=1):
    edge = f" stroke='{stroke}' stroke-width='{width}'" if stroke else ""
    return f"<rect x='{x:.1f}' y='{y:.1f}' width='{w:.1f}' height='{h:.1f}' rx='{rx}' fill='{fill}'{edge}/>"


def fig_outer():
    """NEON의 곱하고 더하기와 SME의 외적."""
    body = [text(20, 28, "명령 하나가 하는 곱셈: NEON은 넷, SME의 외적은 256", INK, 14, "start", "bold")]  # noqa: F405
    # 왼쪽: NEON.
    x0, y0, cell = 40, 92, 26
    body.append(text(x0, 62, "NEON: FMLA", INK, 13, "start", "bold"))  # noqa: F405
    for row, (label, fill, stroke) in enumerate([("가중치 넷", BLUE_BG, BLUE), ("입력 값 하나", ORANGE_BG, ORANGE), ("누산기", GREEN_BG, GREEN)]):  # noqa: F405
        y = y0 + row * 46
        for column in range(4):
            wide = cell if row != 1 or column == 0 else 0
            if wide:
                body.append(rect(x0 + column * cell, y, cell, cell, fill, stroke, 0, 1.2))
        body.append(text(x0 + 4 * cell + 12, y + 18, label, INK, 12, "start"))  # noqa: F405
    body.append(text(x0, y0 + 3 * 46 + 16, "누산기[j] += 입력 × 가중치[j]", MUTED, 12, "start"))  # noqa: F405
    body.append(text(x0, y0 + 3 * 46 + 34, "곱셈 4번. 출력 넷이 한 스텝 나아간다.", MUTED, 12, "start"))  # noqa: F405

    # 오른쪽: FMOPA.
    x1, y1, small, n = 400, 110, 13, 16
    body.append(text(x1 - 40, 62, "SME: FMOPA", INK, 13, "start", "bold"))  # noqa: F405
    # 세로 벡터 (입력 열여섯 개의 k번째 값).
    for index in range(n):
        body.append(rect(x1 - 40, y1 + index * small, small, small, ORANGE_BG, ORANGE, 0, 0.8))  # noqa: F405
    # 가로 벡터 (가중치 행 열여섯 개의 k번째 값).
    for index in range(n):
        body.append(rect(x1 + index * small, y1 - 30, small, small, BLUE_BG, BLUE, 0, 0.8))  # noqa: F405
    # 타일.
    for row in range(n):
        for column in range(n):
            body.append(rect(x1 + column * small, y1 + row * small, small, small, GREEN_BG, GREEN, 0, 0.5))  # noqa: F405
    body.append(rect(x1, y1 + 5 * small, n * small, small, "none", INK, 0, 1.6))  # noqa: F405
    right = x1 + n * small + 18
    body.append(text(right, y1 - 20, "가중치 행 16개의 k번째 값", BLUE, 12, "start"))  # noqa: F405
    body.append(text(right, y1 + 20, "타일 za0: 16 × 16", GREEN, 12, "start", "bold"))  # noqa: F405
    body.append(text(right, y1 + 38, "za[i][j] += 입력[i] × 가중치[j]", INK, 12, "start"))  # noqa: F405
    body.append(text(right, y1 + 56, "곱셈 256번. 입력 벡터 16개의", MUTED, 12, "start"))  # noqa: F405
    body.append(text(right, y1 + 72, "출력 16개씩이 한 스텝 나아간다.", MUTED, 12, "start"))  # noqa: F405
    body.append(text(right, y1 + 5 * small + 44, "가로줄 하나가 입력 벡터 하나의 출력이다.", INK, 12, "start"))  # noqa: F405
    body.append(line(right - 4, y1 + 5 * small + 38, x1 + n * small + 3, y1 + 5 * small + 8, INK, 1, arrow=True))  # noqa: F405
    body.append(text(x1 - 34, y1 + n * small + 18, "입력 벡터 16개의 k번째 값", ORANGE, 12, "start"))  # noqa: F405
    note = y1 + n * small + 50
    body.append(text(20, note, "k를 0부터 끝까지 돌며 외적을 더하면 타일에 y[i][r] = Σ x[i][k] × w[r][k]가 쌓인다. 그러려면 입력도 가중치도 \"k번째 값들\"이", MUTED, 12, "start"))  # noqa: F405
    body.append(text(20, note + 18, "메모리에 이어져 있어야 한다. 두 행렬을 그렇게 다시 늘어놓는 일이 필요하다. M4의 스트리밍 모드에서 벡터 하나는 f32 16개다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "outer.svg", 980, note + 34, body)  # noqa: F405


def engines():
    """bench/ch31 에서 (모델, 엔진, 스레드)마다의 중앙값을 읽는다."""
    samples = {}
    for path in sorted(BENCH.glob("engines_*.txt")):
        for row in path.read_text(encoding="utf-8").splitlines():
            found = re.match(r"(resnet50|mobilenetv2)\s+(.+?)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)$", row)
            if found:
                samples.setdefault((found[1], found[2], int(found[3])), []).append(float(found[4]))
    for path in sorted(BENCH.glob("badak_[0-9]*.txt")):
        for row in path.read_text(encoding="utf-8").splitlines():
            found = re.match(r"(\w+) (neon|sme)\s+스레드\s+(\d+) 중앙값\s+([\d.]+)ms", row)
            if found:
                label = "badak (NEON)" if found[2] == "neon" else "badak (SME)"
                samples.setdefault((found[1], label, int(found[3])), []).append(float(found[4]))
    return {key: statistics.median(values) for key, values in samples.items()}


# 엔진마다, 합성곱을 무엇으로 계산하는가. 색은 이 구분을 따른다.
KIND = {
    "LiteRT + XNNPACK": "SME",
    "ONNX Runtime": "SME",
    "MNN": "SME",
    "badak (SME)": "SME",
    "NCNN f32": "NEON",
    "badak (NEON)": "NEON",
    "LiteRT 자체 커널": "NEON",
    "NCNN fp16 (기본값)": "16비트",
    "MNN 낮은 정밀도": "16비트",
}
COLORS = {"SME": BLUE, "NEON": ORANGE, "16비트": PURPLE}  # noqa: F405


def fig_ranking():
    """스레드의 수에 따라 순위가 바뀐다."""
    times = engines()
    body = [text(20, 28, "ResNet-50, 사진 한 장의 시간 (밀리초). 스레드의 수에 따라 순위가 바뀐다", INK, 14, "start", "bold")]  # noqa: F405
    x_legend = 20
    for kind, color in COLORS.items():
        body.append(rect(x_legend, 42, 14, 10, color, None, 2))
        label = {"SME": "SME로 계산 (f32)", "NEON": "NEON으로 계산 (f32)", "16비트": "16비트로 계산"}[kind]
        body.append(text(x_legend + 20, 52, label, INK, 12, "start"))  # noqa: F405
        x_legend += 190
    label_w, bar_w, row_h, panel_w = 150, 190, 22, 400
    for index, threads in enumerate((1, 4, 12)):
        x = 20 + index * panel_w
        y = 92
        body.append(text(x, y - 10, f"스레드 {threads}개", INK, 13, "start", "bold"))  # noqa: F405
        rows = sorted(((value, key[1]) for key, value in times.items() if key[0] == "resnet50" and key[2] == threads))
        top = max(value for value, _ in rows)
        for rank, (value, engine) in enumerate(rows):
            row = y + rank * row_h
            body.append(text(x + label_w - 8, row + 15, engine, INK, 11, "end"))  # noqa: F405
            body.append(rect(x + label_w, row + 3, max(2, value / top * bar_w), row_h - 6, COLORS[KIND[engine]], None, 2))
            body.append(text(x + label_w + value / top * bar_w + 6, row + 15, f"{value:.1f}", INK, 11, "start"))  # noqa: F405
        body.append(line(x + label_w, y, x + label_w, y + len(rows) * row_h, MUTED, 1))  # noqa: F405
    note = 92 + 9 * row_h + 26
    body.append(text(20, note, "판마다 빠른 순서로 놓았다. 막대의 길이는 판 안에서만 견줄 수 있다(판마다 가장 느린 것이 끝까지 간다).", MUTED, 12, "start"))  # noqa: F405
    body.append(text(20, note + 18, "스레드 하나에서는 SME를 쓰는 엔진들이 앞에 모인다. 열둘에서는 NEON으로 계산하는 엔진들이 따라붙는다. LiteRT 자체 커널의 색은 속도로 미루어 정한 것이다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "ranking.svg", 1240, note + 34, body)  # noqa: F405


if __name__ == "__main__":
    fig_outer()
    if list(BENCH.glob("engines_*.txt")):
        fig_ranking()
