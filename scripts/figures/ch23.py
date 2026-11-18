"""23장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch23.py
출력: book/src/images/ch23/*.svg

측정값은 book/src/part3/output/ch23_*.txt에서 읽는다. 다시 측정하면 그림도
다시 만들어야 한다.
"""

import math
import re

from common import *  # noqa: F403

OUT = IMAGES / "ch23"  # noqa: F405
OUTPUT = IMAGES.parent / "part3/output"  # noqa: F405
GRID = "#e5e7eb"
LIGHT_BLUE = "#93c5fd"
PURPLE = "#7c3aed"


def read(name):
    return (OUTPUT / name).read_text(encoding="utf-8")


def fig_structure():
    body = [text(20, 28, "CPU와 GPU: 빠른 일꾼 몇과 단순한 일꾼 아주 많이", INK, 14, "start", "bold")]
    body += box(30, 50, 380, 250, "", None, "#ffffff", MUTED)
    body.append(text(220, 74, "CPU: 코어 16개", INK, 13, "middle", "bold"))
    for index in range(16):
        x, y = 54 + (index % 8) * 42, 92 + (index // 8) * 78
        fill = BLUE_BG if index < 12 else GRAY_BG
        body.append(f"<rect x='{x}' y='{y}' width='34' height='64' rx='4' fill='{fill}' stroke='{BLUE if index < 12 else MUTED}'/>")
    body.append(text(220, 262, "코어마다 다른 일을 한다. 분기를 예측하고,", MUTED, 12))
    body.append(text(220, 280, "순서를 바꿔 실행하고, 큰 캐시를 가진다.", MUTED, 12))
    body += box(440, 50, 420, 250, "", None, "#ffffff", MUTED)
    body.append(text(650, 74, "GPU: 스레드 수만 개를 한꺼번에", INK, 13, "middle", "bold"))
    for group in range(6):
        gx, gy = 462 + (group % 3) * 130, 92 + (group // 3) * 78
        body.append(f"<rect x='{gx}' y='{gy}' width='118' height='64' rx='4' fill='{ORANGE_BG}' stroke='{ORANGE}'/>")
        for thread in range(32):
            tx, ty = gx + 7 + (thread % 8) * 13.5, gy + 7 + (thread // 8) * 13.5
            body.append(f"<rect x='{tx}' y='{ty}' width='9' height='9' rx='1.5' fill='{ORANGE}'/>")
    body.append(text(650, 262, "스레드 32개가 한 묶음으로 같은 명령을 실행한다.", MUTED, 12))
    body.append(text(650, 280, "묶음 하나는 단순하지만, 묶음이 아주 많다.", MUTED, 12))
    svg(OUT / "structure.svg", 880, 316, body)


def fig_pipeline():
    body = [text(20, 28, "GPU에 일을 시키는 순서", INK, 14, "start", "bold")]
    body.append(f"<rect x='20' y='44' width='400' height='250' rx='8' fill='#ffffff' stroke='{MUTED}' stroke-dasharray='6 4'/>")
    body.append(text(36, 66, "CPU (우리 프로그램)", MUTED, 12, "start"))
    body.append(f"<rect x='480' y='44' width='380' height='250' rx='8' fill='#ffffff' stroke='{MUTED}' stroke-dasharray='6 4'/>")
    body.append(text(496, 66, "GPU", MUTED, 12, "start"))
    steps = [
        ("1. 셰이더를 옮긴다", "WGSL 글 → GPU의 기계어", 78),
        ("2. 버퍼에 값을 담는다", "입력, 가중치, 결과를 받을 자리", 132),
        ("3. 명령을 적어 보낸다", "어느 커널을, 어느 버퍼로, 몇 그룹", 186),
        ("4. 기다렸다가 읽는다", "결과 버퍼를 열어 값을 꺼낸다", 240),
    ]
    for title, sub, y in steps:
        body += box(40, y, 360, 44, title, sub, BLUE_BG, BLUE)
    body += box(500, 78, 340, 44, "커널", "스레드마다 실행되는 함수", ORANGE_BG, ORANGE)
    body += box(500, 132, 340, 44, "버퍼", "GPU가 읽고 쓰는 메모리", ORANGE_BG, ORANGE)
    body += box(500, 186, 340, 44, "실행", "그룹 × 그룹의 스레드 수만큼 동시에", ORANGE_BG, ORANGE)
    body += box(500, 240, 340, 44, "결과", "버퍼에 적혀 있다", ORANGE_BG, ORANGE)
    for y in (100, 154, 208):
        body.append(line(400, y, 498, y, INK, 1.5, arrow=True))
    body.append(line(500, 262, 402, 262, INK, 1.5, arrow=True))
    svg(OUT / "pipeline.svg", 880, 308, body)


def log_axis(x0, w, low, high):
    return lambda value: x0 + (math.log10(value) - low) / (high - low) * w


def fig_launch():
    rows = []
    for row in read("ch23_first.txt").splitlines():
        parts = row.split()
        if len(parts) == 6 and parts[0].isdigit():
            rows.append([float(v) for v in parts])
    x0, y0, w, h = 90, 60, 560, 250
    xs = log_axis(x0, w, 2, 7.5)
    ys = lambda value: y0 + h - (math.log10(value) + 1) / 6 * h  # noqa: E731
    body = [text(20, 28, "원소마다 SiLU: 원소가 몇 개부터 GPU가 이기는가", INK, 14, "start", "bold")]
    for power in range(-1, 6):
        y = ys(10**power)
        body.append(line(x0, y, x0 + w, y, GRID, 1))
        label = {-1: "0.1", 0: "1", 1: "10", 2: "100", 3: "1,000", 4: "10,000", 5: "100,000"}[power]
        body.append(text(x0 - 8, y + 4, label, MUTED, 11, "end"))
    for power in range(2, 8):
        x = xs(10**power)
        body.append(line(x, y0, x, y0 + h, GRID, 1))
        body.append(text(x, y0 + h + 16, f"{10 ** power:,}", MUTED, 11))
    body.append(text(x0 + w / 2, y0 + h + 36, "원소의 수", MUTED, 12))
    body.append(text(20, y0 - 12, "시간 (마이크로초)", MUTED, 12, "start"))
    series = [(1, "CPU, 스레드 1개", MUTED), (2, "CPU, 스레드 16개", BLUE), (3, "GPU, 계산만", ORANGE), (4, "GPU, 값을 올리고 내리는 것까지", PURPLE)]
    for column, label, color in series:
        points = " ".join(f"{xs(r[0]):.1f},{ys(max(r[column], 0.1)):.1f}" for r in rows)
        body.append(f"<polyline points='{points}' fill='none' stroke='{color}' stroke-width='2'/>")
        for r in rows:
            body.append(f"<circle cx='{xs(r[0]):.1f}' cy='{ys(max(r[column], 0.1)):.1f}' r='3' fill='{color}'/>")
    for index, (_, label, color) in enumerate(series):
        y = y0 + 20 + index * 22
        body.append(line(x0 + w + 20, y, x0 + w + 44, y, color, 2))
        body.append(text(x0 + w + 50, y + 4, label, INK, 12, "start"))
    body.append(text(20, y0 + h + 62, "두 축 모두 눈금 하나가 열 배다. GPU는 일이 아무리 작아도 100마이크로초가 든다.", MUTED, 12, "start"))
    svg(OUT / "launch.svg", 880, y0 + h + 78, body)


def read_linear(shape):
    rows, inside = [], False
    for row in read("ch23_linear.txt").splitlines():
        if row.startswith("토큰"):
            inside = row.startswith(shape)
            continue
        if inside and row.startswith("  "):
            name = row[2:20].strip()
            numbers = row[20:].split()
            rows.append((name, float(numbers[0]), float(numbers[2])))
    return rows


def fig_kernels():
    rows = read_linear("토큰 256개, MLP의 첫 층")
    x0, w = 200, 460
    top = max(r[2] for r in rows)
    body = [text(20, 28, "선형 층의 커널들 (벡터 256개 × 행 8,192개 × 길이 2,048)", INK, 14, "start", "bold")]
    body.append(text(20, 48, "1초에 하는 곱셈의 수 (억 번). 값을 올리고 내리는 시간은 넣지 않았다.", MUTED, 12, "start"))
    y = 66
    colors = {"CPU (18장)": MUTED, "단순": LIGHT_BLUE, "넷씩 네 줄기": LIGHT_BLUE, "CPU와 같은 순서": LIGHT_BLUE, "값 열여섯": BLUE, "행렬 연산": GREEN}
    for name, _, rate in rows:
        width = rate / top * w
        body.append(text(x0 - 12, y + 15, name, INK, 12, "end"))
        body.append(f"<rect x='{x0}' y='{y + 2}' width='{width:.1f}' height='18' rx='3' fill='{colors.get(name, BLUE)}'/>")
        body.append(text(x0 + width + 8, y + 16, f"{rate:,.0f}", MUTED, 12, "start"))
        y += 28
    body.append(text(20, y + 12, "회색은 CPU의 열여섯 스레드, 초록은 WebGPU의 표준에 없는 기능을 쓴 커널이다.", MUTED, 12, "start"))
    svg(OUT / "kernels.svg", 880, y + 28, body)


def read_profiles():
    profiles = []
    for block in read("ch23_llama.txt").split("프롬프트 256토큰을 한 번 처리하는 ")[1:]:
        head, *rows = block.splitlines()
        seconds = float(re.match(r"([\d.]+)초", head)[1])
        label = head.split(": ", 1)[1]
        parts = {}
        for row in rows[1:]:
            match = re.match(r"(.+?)\s+(\d+)\s+([\d.]+)\s+([\d.]+)%", row)
            if match:
                parts[match[1].strip()] = float(match[3])
        profiles.append((label, seconds, parts))
    return profiles


def fig_where():
    profiles = read_profiles()
    groups = [
        ("GPU의 계산", ["GPU: 명령을 보내고 기다리기"], ORANGE),
        ("값을 올리고 내리기", ["GPU: 입력을 버퍼에 적기", "GPU: 결과를 읽어 오기"], PURPLE),
        ("어텐션 (CPU)", ["어텐션의 점수와 섞기 (CPU)"], BLUE),
        ("그 밖의 CPU 일", None, MUTED),
    ]
    x0, w = 250, 520
    top = max(sum(parts.values()) for _, _, parts in profiles)
    body = [text(20, 28, "프롬프트 256토큰을 처리하는 시간이 쓰인 곳 (밀리초)", INK, 14, "start", "bold")]
    for index, (label, _, color) in enumerate(groups):
        body.append(f"<rect x='{20 + index * 190}' y='42' width='14' height='11' rx='2' fill='{color}'/>")
        body.append(text(40 + index * 190, 52, label, MUTED, 11, "start"))
    y = 74
    for label, _, parts in profiles:
        named = sum(parts.get(k, 0) for _, keys, _ in groups if keys for k in keys)
        rest = sum(parts.values()) - named
        body.append(text(x0 - 12, y + 17, label, INK, 12, "end"))
        x = x0
        for _, keys, color in groups:
            value = rest if keys is None else sum(parts.get(k, 0) for k in keys)
            width = value / top * w
            body.append(f"<rect x='{x:.1f}' y='{y}' width='{width:.1f}' height='24' fill='{color}'/>")
            if width > 30:
                body.append(text(x + width / 2, y + 16, f"{value:.0f}", "#ffffff", 11))
            x += width
        body.append(text(x + 8, y + 17, f"{sum(parts.values()):.0f}", MUTED, 12, "start"))
        y += 40
    body.append(text(20, y + 8, "값을 옮기는 시간을 줄이고 나면, GPU에 맡기지 않은 어텐션이 다음으로 큰 덩어리다.", MUTED, 12, "start"))
    svg(OUT / "where.svg", 880, y + 24, body)


def fig_split():
    rows = []
    for row in read("ch23_split.txt").splitlines():
        parts = row.replace("배", "").split()
        if len(parts) == 5 and parts[0].isdigit():
            rows.append([float(v) for v in parts])
    x0, y0, w, h = 80, 60, 520, 230
    xs = lambda value: x0 + math.log2(value) / 8 * w  # noqa: E731
    top = 8.0
    ys = lambda value: y0 + h - value / top * h  # noqa: E731
    body = [text(20, 28, "선형 층 하나를 부르는 시간: 벡터가 몇 개부터 GPU에 맡길까", INK, 14, "start", "bold")]
    for tick in range(0, 9, 2):
        body.append(line(x0, ys(tick), x0 + w, ys(tick), GRID, 1))
        body.append(text(x0 - 8, ys(tick) + 4, str(tick), MUTED, 11, "end"))
    for r in rows:
        body.append(text(xs(r[0]), y0 + h + 16, f"{int(r[0])}", MUTED, 11))
    body.append(text(x0 + w / 2, y0 + h + 36, "입력 벡터의 수", MUTED, 12))
    body.append(text(20, y0 - 12, "시간 (밀리초)", MUTED, 12, "start"))
    series = [(1, "CPU", BLUE), (2, "GPU, 값 열여섯", ORANGE), (3, "GPU, 행렬 연산", GREEN)]
    for column, label, color in series:
        points = " ".join(f"{xs(r[0]):.1f},{ys(r[column]):.1f}" for r in rows)
        body.append(f"<polyline points='{points}' fill='none' stroke='{color}' stroke-width='2'/>")
        for r in rows:
            body.append(f"<circle cx='{xs(r[0]):.1f}' cy='{ys(r[column]):.1f}' r='3' fill='{color}'/>")
    for index, (_, label, color) in enumerate(series):
        y = y0 + 20 + index * 22
        body.append(line(x0 + w + 30, y, x0 + w + 54, y, color, 2))
        body.append(text(x0 + w + 60, y + 4, label, INK, 12, "start"))
    body.append(text(20, y0 + h + 62, "GPU 쪽은 입력을 올리고 결과를 내려받는 시간까지 넣은 것이다. 벡터 8개와 16개 사이에서 갈린다.", MUTED, 12, "start"))
    svg(OUT / "split.svg", 880, y0 + h + 78, body)


if __name__ == "__main__":
    for figure in (fig_structure, fig_pipeline, fig_launch, fig_kernels, fig_where, fig_split):
        figure()
    print(f"wrote {len(list(OUT.glob('*.svg')))} figures to {OUT}")
