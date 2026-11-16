"""11장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch11.py
출력: book/src/images/ch11/*.svg

시간 그림의 숫자는 book/src/part2/output/ch11_*.txt에서 읽는다. 예제를 다시
실행해 출력 파일이 바뀌면 그림도 다시 만들어야 한다.
"""

import math
import re

from common import *  # noqa: F403

OUT = IMAGES / "ch11"  # noqa: F405
OUTPUT = IMAGES.parent / "part2/output"  # noqa: F405
GRID = "#e5e7eb"


def fig_recompute():
    """토큰을 뽑을 때마다 어느 토큰을 계산하는가."""
    body = []
    prompt, steps, size = 4, 5, 26
    panels = [("캐시 없이", 60, False), ("KV 캐시 사용", 470, True)]
    for title, x0, cached in panels:
        body.append(text(x0 + 150, 28, title, INK, 14, weight="bold"))
        body.append(text(x0 + 150, 48, "가로: 글 속의 토큰,  세로: 토큰을 뽑는 차례", MUTED, 12))
        total = 0
        for step in range(steps):
            length = prompt + step
            y = 70 + step * (size + 8)
            body.append(text(x0 - 10, y + 18, f"{step + 1}번째", MUTED, 12, "end"))
            for token in range(length):
                # 캐시가 있으면 첫 차례에는 프롬프트 전체, 그 뒤로는 마지막 토큰만 계산한다.
                computed = (not cached) or step == 0 or token == length - 1
                is_prompt = token < prompt
                fill = (ORANGE_BG if is_prompt else BLUE_BG) if computed else "#ffffff"
                stroke = (ORANGE if is_prompt else BLUE) if computed else GRID
                body.append(
                    f"<rect x='{x0 + token * (size + 4)}' y='{y}' width='{size}' height='{size}' rx='4' "
                    f"fill='{fill}' stroke='{stroke}'/>"
                )
                total += computed
            # 이번 차례에 뽑힌 토큰.
            nx = x0 + length * (size + 4) + 12
            body.append(line(nx - 10, y + size / 2, nx + 6, y + size / 2, GREEN, 1.5, arrow=True))
            body.append(f"<rect x='{nx + 10}' y='{y}' width='{size}' height='{size}' rx='4' fill='{GREEN_BG}' stroke='{GREEN}'/>")
        body.append(text(x0 + 150, 70 + steps * (size + 8) + 22, f"계산한 칸: {total}개", INK, 13, weight="bold"))
    ly = 70 + steps * (size + 8) + 52
    for i, (label, fill, stroke) in enumerate(
        [("프롬프트의 토큰", ORANGE_BG, ORANGE), ("앞에서 뽑은 토큰", BLUE_BG, BLUE), ("이번에 뽑는 토큰", GREEN_BG, GREEN), ("다시 계산하지 않는 토큰", "#ffffff", GRID)]
    ):
        lx = 60 + i * 195
        body.append(f"<rect x='{lx}' y='{ly}' width='16' height='16' rx='3' fill='{fill}' stroke='{stroke}'/>")
        body.append(text(lx + 24, ly + 13, label, MUTED, 12, "start"))
    svg(OUT / "recompute.svg", 860, ly + 34, body)


def fig_cache():
    """블록 하나의 캐시: 새 토큰의 질의가 쌓여 있는 키와 값을 읽는다."""
    body = []
    rows, rh, w = 5, 30, 150
    kx, vx, y0 = 330, 520, 80
    body.append(text((kx + vx + w) / 2, 30, "블록 하나의 KV 캐시", INK, 14, weight="bold"))
    for x, title in ((kx, "키"), (vx, "값")):
        body.append(text(x + w / 2, 62, f"{title}  (토큰 수, 512)", MUTED, 12))
        for row in range(rows):
            new = row == rows - 1
            body.append(
                f"<rect x='{x}' y='{y0 + row * rh}' width='{w}' height='{rh}' "
                f"fill='{BLUE_BG if new else GRAY_BG}' stroke='{BLUE if new else MUTED}'/>"
            )
            label = "새 토큰" if new else f"토큰 {row}"
            body.append(text(x + w / 2, y0 + row * rh + 20, label, INK, 12))
    ny = y0 + (rows - 1) * rh + rh / 2
    body += box(40, ny - 26, 170, 52, "새 토큰의 벡터", "(1, 2048)", ORANGE_BG, ORANGE)
    body.append(line(210, ny, kx - 6, ny, BLUE, 1.5, arrow=True))
    body.append(text(270, ny - 8, "① 덧붙인다", BLUE, 12))
    # 질의가 모든 키를 읽는다.
    qy = 280
    body += box(40, qy, 170, 46, "새 토큰의 질의", None, ORANGE_BG, ORANGE)
    body.append(line(125, ny + 26, 125, qy - 2, ORANGE, 1.5, arrow=True))
    body.append(line(210, qy + 23, kx + w / 2, qy + 23, ORANGE, 1.5))
    body.append(line(kx + w / 2, qy + 23, kx + w / 2, y0 + rows * rh + 4, ORANGE, 1.5, arrow=True))
    body.append(text(kx + w / 2 + 12, qy + 12, "② 모든 키와 내적 → 비중", ORANGE, 12, "start"))
    body.append(line(vx + w / 2, y0 + rows * rh + 2, vx + w / 2, qy + 44, GREEN, 1.5))
    body.append(line(vx + w / 2, qy + 44, vx + w + 60, qy + 44, GREEN, 1.5, arrow=True))
    body.append(text(vx + w / 2 + 12, qy + 62, "③ 값을 비중대로 섞는다", GREEN, 12, "start"))
    body.append(text(40, 368, "앞선 토큰의 키와 값은 다시 계산하지 않고 읽기만 한다. 캐시는 토큰 하나에 한 줄씩 자란다.", MUTED, 12, "start"))
    svg(OUT / "cache.svg", 800, 388, body)


def read_steps(name):
    """측정표를 (글의 길이, 캐시 없이, 캐시 사용)의 목록으로 읽는다."""
    rows = []
    for row in (OUTPUT / name).read_text(encoding="utf-8").splitlines():
        match = re.match(r"\s*\d+\s+(\d+)\s+([\d.]+) 초\s+([\d.]+) 초", row)
        if match:
            rows.append((int(match[1]), float(match[2]), float(match[3])))
    return rows


def fig_per_token():
    body = []
    rows = read_steps("ch11_bench_llama32.txt")
    x0, y0, w, h, top = 80, 50, 520, 240, 14.0
    lo, hi = rows[0][0], rows[-1][0]
    body.append(text(x0, 28, "Llama 3.2 1B: 토큰 하나를 뽑는 데 걸린 시간", INK, 13, "start", "bold"))
    for value in range(0, int(top) + 1, 2):
        yy = y0 + h - value / top * h
        body.append(line(x0, yy, x0 + w, yy, GRID, 1))
        body.append(text(x0 - 8, yy + 4, f"{value}초", MUTED, 11, "end"))
    body.append(line(x0, y0 + h, x0 + w, y0 + h, MUTED, 1))
    for length in range(lo, hi + 1):
        if (length - lo) % 4 == 0 or length == hi:
            xx = x0 + (length - lo) / (hi - lo) * w
            body.append(line(xx, y0 + h, xx, y0 + h + 5, MUTED, 1))
            body.append(text(xx, y0 + h + 19, str(length), MUTED, 11))
    body.append(text(x0 + w / 2, y0 + h + 40, "모델에 들어간 글의 길이 (토큰 수)", MUTED, 12))

    def points(column):
        return [(x0 + (row[0] - lo) / (hi - lo) * w, y0 + h - row[column] / top * h) for row in rows]

    for column, color, label in ((1, ORANGE, "캐시 없이"), (2, BLUE, "KV 캐시 사용")):
        pts = points(column)
        path = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
        body.append(f"<polyline points='{path}' fill='none' stroke='{color}' stroke-width='2'/>")
        body += [f"<circle cx='{px:.1f}' cy='{py:.1f}' r='3' fill='{color}' stroke='#ffffff'/>" for px, py in pts]
        body.append(text(pts[-1][0] + 10, pts[-1][1] + 4, label, color, 12, "start", "bold"))
    first = points(2)[0]
    body.append(text(first[0] + 34, y0 + h - 1.7 / top * h, "첫 차례에는 프롬프트 전체를 처리한다", MUTED, 11, "start"))
    svg(OUT / "per_token_time.svg", 760, y0 + h + 56, body)


def read_prompt(name, length):
    """프롬프트 측정표에서 (토큰당 프롬프트 처리, 새 토큰 하나)를 읽는다."""
    for row in (OUTPUT / name).read_text(encoding="utf-8").splitlines():
        match = re.match(rf"\s*{length}개\s+[\d.]+ 초\s+([\d.]+) 초\s+([\d.]+) 초", row)
        if match:
            return float(match[1]), float(match[2])
    raise ValueError(name)


def fig_prompt_vs_generation():
    body = []
    engines = [("badak", "ch11_prompt_bench_llama32.txt"), ("PyTorch (CPU)", "ch11_bench_pytorch_llama32.txt")]
    x0, w, lo, hi = 190, 520, -4, 0
    body.append(text(x0, 28, "Llama 3.2 1B, 프롬프트 256개: 토큰 하나에 드는 시간 (로그 눈금)", INK, 13, "start", "bold"))

    def xpos(seconds):
        return x0 + (math.log10(seconds) - lo) / (hi - lo) * w

    for exponent in range(lo, hi + 1):
        xx = x0 + (exponent - lo) / (hi - lo) * w
        body.append(line(xx, 60, xx, 190, GRID, 1))
        label = {0: "1초", -1: "0.1초", -2: "10밀리초", -3: "1밀리초", -4: "0.1밀리초"}[exponent]
        body.append(text(xx, 208, label, MUTED, 11))
    for i, (name, file) in enumerate(engines):
        y = 95 + i * 60
        prompt, new = read_prompt(file, 256)
        body.append(text(x0 - 16, y + 5, name, INK, 13, "end", "bold"))
        body.append(line(xpos(prompt), y, xpos(new), y, MUTED, 2))
        body.append(f"<circle cx='{xpos(prompt):.1f}' cy='{y}' r='6' fill='{ORANGE}' stroke='#ffffff' stroke-width='2'/>")
        body.append(f"<circle cx='{xpos(new):.1f}' cy='{y}' r='6' fill='{BLUE}' stroke='#ffffff' stroke-width='2'/>")
        body.append(text(xpos(prompt), y - 14, f"{prompt * 1e3:.1f}밀리초" if prompt < 0.1 else f"{prompt:.2f}초", ORANGE, 11))
        body.append(text(xpos(new), y + 24, f"{new * 1e3:.1f}밀리초" if new < 0.1 else f"{new:.2f}초", BLUE, 11))
    for i, (label, color) in enumerate((("프롬프트의 토큰 하나", ORANGE), ("새로 만드는 토큰 하나", BLUE))):
        body.append(f"<circle cx='{x0 + 8 + i * 190}' cy='240' r='5' fill='{color}'/>")
        body.append(text(x0 + 20 + i * 190, 244, label, MUTED, 12, "start"))
    svg(OUT / "prompt_vs_generation.svg", 780, 264, body)


if __name__ == "__main__":
    for figure in (fig_recompute, fig_cache, fig_per_token, fig_prompt_vs_generation):
        figure()
    print(f"wrote {len(list(OUT.glob('*.svg')))} figures to {OUT}")
