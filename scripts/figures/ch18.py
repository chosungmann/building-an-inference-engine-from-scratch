"""18장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch18.py
출력: book/src/images/ch18/*.svg

측정값은 book/src/part3/output/ch18_*.txt와 bench/의 기록에서 읽는다.
다시 측정하면 그림도 다시 만들어야 한다.
"""

import re

from common import *  # noqa: F403

OUT = IMAGES / "ch18"  # noqa: F405
ROOT = IMAGES.parents[2]  # noqa: F405
OUTPUT = IMAGES.parent / "part3/output"  # noqa: F405
GRID = "#e5e7eb"
LIGHT_BLUE = "#93c5fd"


def read(name):
    return (OUTPUT / name).read_text(encoding="utf-8")


def fig_pool():
    body = [text(20, 30, "스레드 풀: 일 하나가 지나가는 길", INK, 14, "start", "bold")]
    body += box(20, 118, 170, 64, "맡기는 스레드", "run(...)을 부른다", BLUE_BG, BLUE)
    # 함께 보는 상태
    body.append(
        f"<rect x='250' y='48' width='380' height='252' rx='8' fill='#ffffff' stroke='{MUTED}' "
        "stroke-width='1.5' stroke-dasharray='6 4'/>"
    )
    body.append(text(266, 70, "함께 보는 상태", MUTED, 12, "start"))
    states = [
        ("job: 부를 함수와 번호의 수", "① 맡기는 쪽이 적는다", GRAY_BG, MUTED),
        ("call: 일의 번호 + 참여할 일꾼의 수", "② 맡기는 쪽이 바꾼다  ③ 일꾼이 알아챈다", ORANGE_BG, ORANGE),
        ("next: 다음에 가져갈 번호", "④ 모두가 번호를 하나씩 뽑아 가며 일한다", GREEN_BG, GREEN),
        ("running: 아직 안 끝난 일꾼의 수", "⑤ 일꾼이 줄인다  ⑥ 0이면 맡긴 쪽이 돌아간다", GRAY_BG, MUTED),
    ]
    for i, (title, sub, fill, stroke) in enumerate(states):
        body += box(266, 82 + i * 54, 348, 46, title, sub, fill, stroke)
    workers = [
        ("일꾼 0", "일한다", BLUE_BG, BLUE),
        ("일꾼 1", "일한다", BLUE_BG, BLUE),
        ("일꾼 2", "부르지 않았다: 잔다", GRAY_BG, MUTED),
    ]
    for i, (label, sub, fill, stroke) in enumerate(workers):
        body += box(690, 62 + i * 84, 170, 60, label, sub, fill, stroke)
    body.append(line(190, 150, 246, 150, INK, 1.5, arrow=True))
    body.append(line(690, 92, 634, 136, INK, 1.5, arrow=True))
    body.append(line(690, 176, 634, 164, INK, 1.5, arrow=True))
    body.append(
        text(
            20,
            326,
            "맡기는 스레드도 ④에서 함께 일한다. 일꾼 2처럼 부르지 않은 일꾼은 깨어날 일이 없다.",
            MUTED,
            12,
            "start",
        )
    )
    svg(OUT / "pool.svg", 880, 342, body)


def fig_split():
    body = [text(20, 30, "선형 층의 일을 나누는 두 가지 방법 (스레드 3개)", INK, 14, "start", "bold")]
    colors = [(BLUE_BG, BLUE), (ORANGE_BG, ORANGE), (GREEN_BG, GREEN)]

    def panel(x, title, sub, by_rows):
        out = [text(x, 62, title, INK, 13, "start", "bold"), text(x, 80, sub, MUTED, 11, "start")]
        # 입력 벡터들
        out.append(text(x, 108, "입력 벡터", MUTED, 11, "start"))
        for v in range(6):
            fill, stroke = (GRAY_BG, MUTED) if by_rows else colors[v // 2]
            out.append(f"<rect x='{x}' y='{116 + v * 14}' width='120' height='11' rx='2' fill='{fill}' stroke='{stroke}'/>")
        # 가중치
        out.append(text(x + 160, 108, "가중치", MUTED, 11, "start"))
        for r in range(12):
            fill, stroke = colors[r // 4] if by_rows else (GRAY_BG, MUTED)
            out.append(f"<rect x='{x + 160}' y='{116 + r * 14}' width='190' height='11' rx='2' fill='{fill}' stroke='{stroke}'/>")
        return out

    body += panel(30, "행을 나눈다", "스레드마다 가중치의 한 구간만 읽는다", True)
    body.append(text(30, 306, "구간 하나가 캐시에 머무는 동안 모든 벡터가 지나간다", MUTED, 11, "start"))
    body += panel(460, "벡터를 나눈다", "스레드마다 가중치 전체를 읽는다", False)
    body.append(text(460, 306, "같은 가중치를 스레드 수만큼 되풀이해 메모리에서 읽어 온다", MUTED, 11, "start"))
    for i, (fill, stroke) in enumerate(colors):
        body.append(f"<rect x='{660 + i * 70}' y='20' width='14' height='11' rx='2' fill='{fill}' stroke='{stroke}'/>")
        body.append(text(680 + i * 70, 30, f"스레드 {i}", MUTED, 11, "start"))
    svg(OUT / "split.svg", 880, 322, body)


def read_scaling(name):
    rows = []
    for row in read(name).splitlines():
        match = re.match(r"\s+(\d+)\s+([\d.]+) 토큰/초\s+([\d.]+)배\s+([\d.]+) 토큰/초\s+([\d.]+)배", row)
        if match:
            rows.append((int(match[1]), float(match[2]), float(match[3]), float(match[4]), float(match[5])))
    return rows


def fig_scaling():
    final = read_scaling("ch18_scaling_llama32.txt")
    linear_only = read_scaling("ch18_scaling_linear_only.txt")
    x0, y0, w, h, top = 70, 56, 560, 300, 16
    body = [text(20, 30, "Llama 3.2 1B: 스레드의 수와 속도 (스레드 1개의 몇 배인가)", INK, 14, "start", "bold")]

    def pos(threads, factor):
        return x0 + threads / 16 * w, y0 + h - factor / top * h

    for value in range(0, 17, 4):
        _, yy = pos(0, value)
        body.append(line(x0, yy, x0 + w, yy, GRID, 1))
        body.append(text(x0 - 8, yy + 4, f"{value}배", MUTED, 11, "end"))
    for threads in (1, 2, 4, 6, 8, 10, 12, 14, 16):
        xx, _ = pos(threads, 0)
        body.append(text(xx, y0 + h + 18, str(threads), MUTED, 11))
    body.append(text(x0 + w / 2, y0 + h + 42, "스레드의 수", MUTED, 12))
    # 성능 코어와 효율 코어의 경계
    xx, _ = pos(12, 0)
    body.append(line(xx, y0, xx, y0 + h, MUTED, 1, dash="3 4"))
    body.append(text(xx + 6, y0 + h - 8, "여기부터 효율 코어", MUTED, 11, "start"))
    # 이상적인 선
    ax, ay = pos(1, 1)
    bx, by = pos(16, 16)
    body.append(line(ax, ay, bx, by, MUTED, 1.5, dash="6 4"))
    body.append(text(pos(11.2, 12.2)[0], pos(11.2, 12.2)[1], "스레드의 수만큼", MUTED, 11, "end"))

    def series(points, color, label, dy=4):
        path = " ".join("{:.1f},{:.1f}".format(*pos(t, f)) for t, f in points)
        out = [f"<polyline points='{path}' fill='none' stroke='{color}' stroke-width='2'/>"]
        for t, f in points:
            px, py = pos(t, f)
            out.append(f"<circle cx='{px:.1f}' cy='{py:.1f}' r='4' fill='{color}' stroke='#ffffff' stroke-width='2'/>")
        px, py = pos(*points[-1])
        out.append(text(px + 12, py + dy, label, INK, 12, "start"))
        return out

    body += series([(r[0], r[2]) for r in final], BLUE, f"프롬프트 처리 ({final[-1][2]:.1f}배)")
    body += series(
        [(r[0], r[2]) for r in linear_only], LIGHT_BLUE, f"프롬프트 처리, 선형 층만 나눔 ({linear_only[-1][2]:.1f}배)"
    )
    body += series([(r[0], r[4]) for r in final], ORANGE, f"토큰 생성 ({final[-1][4]:.1f}배)")
    svg(OUT / "scaling.svg", 900, y0 + h + 58, body)


def read_profile(name):
    text_ = read(name)
    section = text_.split("연산", 1)[1]
    parts = {m[1]: float(m[2]) for m in re.finditer(r"^(\w+)\s+([\d.]+)\s+[\d.]+%", section, re.M)}
    total = float(re.search(r"프로파일러를 켜고 잰 시간\s+([\d.]+)", text_)[1])
    return parts, total


def fig_amdahl():
    rows = [
        ("스레드 1개", "ch18_profile_pp_one_thread.txt"),
        ("스레드 16개, 선형 층만 나눔", "ch18_profile_pp_linear_only.txt"),
        ("스레드 16개, 어텐션과 map도 나눔", "ch18_profile_pp.txt"),
    ]
    kinds = [("linear", "선형 층", BLUE), ("attention", "어텐션", ORANGE), ("map", "map (SiLU)", GREEN), (None, "나머지", MUTED)]
    x0, w = 250, 480
    body = [text(20, 30, "프롬프트 256토큰을 처리하는 시간이 어디에 쓰이는가", INK, 14, "start", "bold")]
    for i, (_, label, color) in enumerate(kinds):
        body.append(f"<rect x='{x0 + i * 120}' y='46' width='14' height='11' rx='2' fill='{color}'/>")
        body.append(text(x0 + 20 + i * 120, 56, label, MUTED, 11, "start"))
    y = 78
    for label, name in rows:
        parts, total = read_profile(name)
        known = sum(parts.get(key, 0.0) for key, _, _ in kinds if key)
        own = sum(parts.values())
        body.append(text(x0 - 12, y + 17, label, INK, 12, "end"))
        x = x0
        for key, _, color in kinds:
            value = parts.get(key, 0.0) if key else own - known
            width = value / own * w
            if width > 3:
                body.append(f"<rect x='{x:.1f}' y='{y}' width='{width - 2:.1f}' height='24' rx='3' fill='{color}'/>")
            if width > 44:
                body.append(text(x + width / 2 - 1, y + 16, f"{value / own * 100:.0f}%", "#ffffff", 11))
            x += width
        body.append(text(x0 + w + 10, y + 17, f"{total:.2f}초", INK, 12, "start"))
        y += 40
    body.append(
        text(20, y + 14, "막대의 길이는 비중이다. 전체 시간은 오른쪽에 적었다. 나누지 않은 일의 비중은 스레드를 늘릴수록 커진다.", MUTED, 12, "start")
    )
    svg(OUT / "amdahl.svg", 860, y + 30, body)


def speed(path, test):
    text_ = (ROOT / path).read_text(encoding="utf-8")
    return float(re.search(rf"{test}\s+([\d.]+)", text_)[1])


def fig_result():
    sections = (ROOT / "bench/ch13/llama_cpp_llama32.txt").read_text(encoding="utf-8").split("## ")
    theirs = sections[2]
    twelve = next(row for row in read_scaling("ch18_scaling_llama32.txt") if row[0] == 12)
    groups = [
        ("토큰 생성", "tg32", twelve[3], float(re.search(r"tg32 \|\s+([\d.]+)", theirs)[1]), 70.0, 55.6),
        ("프롬프트 처리", "pp256", twelve[1], float(re.search(r"pp256 \|\s+([\d.]+)", theirs)[1]), 420.0, None),
    ]
    x0, w = 250, 440
    body = [text(20, 28, "Llama 3.2 1B: 1초에 처리하는 토큰", INK, 14, "start", "bold")]
    y = 56
    for title, test, ours, llama, top_value, bound in groups:
        bars = [
            ("17장 (스레드 1개)", speed("bench/baseline/badak_llama32_ch17.txt", test), LIGHT_BLUE),
            ("이 장 (스레드 12개)", ours, BLUE),
            ("llama.cpp (스레드 12개)", llama, ORANGE),
        ]
        body.append(text(20, y + 16, title, INK, 13, "start", "bold"))
        start = y
        for label, value, color in bars:
            width = value / top_value * w
            body.append(text(x0 - 12, y + 15, label, INK, 12, "end"))
            body.append(
                f"<path d='M{x0},{y + 2} h{max(width - 3, 1):.1f} a3,3 0 0 1 3,3 v12 a3,3 0 0 1 -3,3 h-{max(width - 3, 1):.1f} z' fill='{color}'/>"
            )
            if bound and 0 < bound - value < 0.1 * top_value:
                # 상한선과 겹치지 않게 막대 안에 적는다.
                body.append(text(x0 + width - 8, y + 16, f"{value:.1f}", "#ffffff", 12, "end"))
            else:
                body.append(text(x0 + width + 8, y + 16, f"{value:.1f}", MUTED, 12, "start"))
            y += 28
        if bound:
            bx = x0 + bound / top_value * w
            body.append(line(bx, start - 4, bx, y - 4, INK, 1.5, dash="4 3"))
            body.append(text(bx + 6, start + 16, f"15장의 상한 {bound}", INK, 11, "start"))
        y += 18
    body.append(text(20, y + 6, "두 막대그래프의 눈금은 서로 다르다. llama.cpp는 13.5절의 측정이다(직접 짠 CPU 코드만, F32).", MUTED, 12, "start"))
    svg(OUT / "result.svg", 860, y + 22, body)


if __name__ == "__main__":
    for figure in (fig_pool, fig_split, fig_scaling, fig_amdahl, fig_result):
        figure()
    print(f"wrote {len(list(OUT.glob('*.svg')))} figures to {OUT}")
