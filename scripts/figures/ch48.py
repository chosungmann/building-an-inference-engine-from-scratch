"""48장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch48.py
출력: book/src/images/ch48/*.svg

bench/ch48/ 에 남긴 측정값을 그린다.
"""

import math

from common import *  # noqa: F403

OUT = IMAGES / "ch48"  # noqa: F405
BENCH = IMAGES.parents[2] / "bench/ch48"  # noqa: F405
PURPLE, RED = "#7c3aed", "#dc2626"
BROWSERS = [("chrome", "Chrome", BLUE), ("edge", "Edge", ORANGE), ("firefox", "Firefox", GREEN), ("safari", "Safari", PURPLE)]  # noqa: F405


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


def number(parts, key):
    return float(parts[parts.index(key) + 1])


def fig_stack():
    """페이지, 엔진 워커, 스레드 워커."""
    body = [text(20, 28, "브라우저 안의 badak: 워커 안의 모듈, 그리고 같은 메모리를 보는 워커들", INK, 14, "start", "bold")]  # noqa: F405
    # 페이지.
    body.append(rect(20, 50, 210, 250, GRAY_BG, MUTED, 8, 1.5))  # noqa: F405
    body.append(text(125, 74, "페이지 (주 스레드)", INK, 13, "middle", "bold"))  # noqa: F405
    for index, label in enumerate(["화면을 그린다", "입력을 받는다", "client.js", "기다릴 수 없다"]):
        body.append(text(125, 104 + index * 22, label, MUTED if index != 2 else INK, 12))  # noqa: F405
    # 엔진 워커.
    body.append(rect(300, 50, 300, 250, BLUE_BG, BLUE, 8, 1.5))  # noqa: F405
    body.append(text(450, 74, "엔진 워커 (engine.js)", INK, 13, "middle", "bold"))  # noqa: F405
    body.append(rect(320, 90, 260, 44, "#ffffff", BLUE, 6))  # noqa: F405
    body.append(text(450, 109, "badak.js: 모듈을 올리고 감싼다", INK, 12))  # noqa: F405
    body.append(text(450, 126, "파일을 받아 캐시에 넣는다", MUTED, 11))  # noqa: F405
    body.append(rect(320, 144, 260, 40, "#ffffff", BLUE, 6))  # noqa: F405
    body.append(text(450, 162, "wasi.js: 엔진이 들여오는 함수 23개", INK, 12))  # noqa: F405
    body.append(text(450, 177, "파일, 시계, 난수, 환경 변수", MUTED, 11))  # noqa: F405
    body.append(rect(320, 198, 260, 86, ORANGE_BG, ORANGE, 6, 1.5))  # noqa: F405
    body.append(text(450, 220, "badak_*.wasm", INK, 13, "middle", "bold"))  # noqa: F405
    body.append(text(450, 240, "엔진 전체 (Rust, 0.9MB)", INK, 12))  # noqa: F405
    body.append(text(450, 258, "내놓는 함수: badak_web_*", MUTED, 11))  # noqa: F405
    body.append(text(450, 274, "들여오는 함수: WASI, piece", MUTED, 11))  # noqa: F405
    # 스레드 워커.
    for index in range(3):
        y = 60 + index * 62
        body.append(rect(680, y, 200, 50, GREEN_BG, GREEN, 8, 1.5))  # noqa: F405
        body.append(text(780, y + 21, f"스레드 워커 {index + 1} (thread.js)", INK, 12, "middle", "bold"))  # noqa: F405
        body.append(text(780, y + 39, "같은 모듈을 한 번 더 올린다", MUTED, 11))  # noqa: F405
    body.append(rect(680, 250, 200, 50, ORANGE_BG, ORANGE, 8, 1.5, "5 3"))  # noqa: F405
    body.append(text(780, 271, "공유 메모리", INK, 12, "middle", "bold"))  # noqa: F405
    body.append(text(780, 289, "격리된 페이지에서만", MUTED, 11))  # noqa: F405
    # 화살표.
    body.append(line(232, 150, 296, 150, INK, 1.5, True))  # noqa: F405
    body.append(line(296, 190, 232, 190, INK, 1.5, True))  # noqa: F405
    body.append(text(264, 142, "요청", MUTED, 11))  # noqa: F405
    body.append(text(264, 206, "답, 글 조각", MUTED, 11))  # noqa: F405
    for index in range(3):
        body.append(line(602, 110 + index * 20, 676, 85 + index * 62, GREEN, 1.5, True))  # noqa: F405
    body.append(text(640, 76, "시작하라", MUTED, 11))  # noqa: F405
    body.append(line(582, 262, 676, 272, ORANGE, 1.5, True, "5 3"))  # noqa: F405
    body.append(line(780, 234, 780, 248, ORANGE, 1.5, True, "5 3"))  # noqa: F405
    svg(OUT / "stack.svg", 900, 320, body)  # noqa: F405


def fig_isolation():
    """서비스 워커가 헤더를 붙이는 순서."""
    body = [text(20, 28, "헤더를 정할 수 없는 서버에서 페이지를 격리하기", INK, 14, "start", "bold")]  # noqa: F405
    columns = [("페이지", 130), ("서비스 워커 (coi.js)", 450), ("서버 (GitHub Pages)", 770)]
    for label, x in columns:
        body.append(rect(x - 95, 46, 190, 30, GRAY_BG, MUTED, 6))  # noqa: F405
        body.append(text(x, 66, label, INK, 12, "middle", "bold"))  # noqa: F405
        body.append(line(x, 76, x, 372, MUTED, 1, False, "3 4"))  # noqa: F405
    steps = [
        (104, 130, 770, "① 페이지를 달라", INK),
        (130, 770, 130, "헤더 없는 응답 → 격리되지 않음", ORANGE),  # noqa: F405
        (164, 130, 450, "② 서비스 워커를 등록한다", INK),
        (198, 130, 130, "③ 스스로를 다시 읽는다", INK),
        (232, 130, 450, "④ 페이지를 달라", INK),
        (258, 450, 770, "그대로 전한다", MUTED),
        (284, 770, 450, "헤더 없는 응답", MUTED),
        (318, 450, 130, "⑤ COOP, COEP 헤더를 붙인 응답 → 격리됨", GREEN),
        (352, 130, 130, "⑥ SharedArrayBuffer를 쓸 수 있다", GREEN),
    ]
    for y, start, end, label, color in steps:
        if start == end:
            body.append(text(start + 12, y + 4, label, color, 12, "start"))  # noqa: F405
            body.append(f"<circle cx='{start}' cy='{y}' r='4' fill='{color}'/>")
            continue
        body.append(line(start, y, end + (6 if end < start else -6), y, color, 1.5, True))  # noqa: F405
        body.append(text((start + end) / 2, y - 7, label, color, 12))  # noqa: F405
    svg(OUT / "isolation.svg", 900, 386, body)  # noqa: F405


def fig_kernels():
    """같은 일을 커널 셋으로: 스레드 하나."""
    shape = "1536x576x64"
    rows = []
    native = [parts for parts in lines("native_1.txt") if parts[:2] == ["linear", shape]]
    if native:
        rows.append(("맥에서 그대로 (NEON)", [("", MUTED, float(native[0][-1]))]))  # noqa: F405
    for kernel, label in [("relaxed", "relaxed-simd"), ("simd", "simd128"), ("scalar", "벡터 명령 없이")]:
        bars = []
        for name, shown, color in BROWSERS:
            found = [p for p in lines(f"{name}.txt") if p[0] == "linear" and p[1] == f"badak_{kernel}.wasm" and p[4] == shape]
            if found:
                bars.append((shown, color, number(found[0], "us")))
        if bars:
            rows.append((label, bars))
    if len(rows) < 2:
        return
    body = [text(20, 28, "선형 층 하나(576 → 1536, 벡터 64개)에 걸린 시간: 스레드 하나", INK, 14, "start", "bold")]  # noqa: F405
    x = 20
    for _, shown, color in BROWSERS:
        body.append(rect(x, 42, 14, 10, color))
        body.append(text(x + 20, 52, shown, INK, 12, "start"))  # noqa: F405
        x += 100
    left, width = 190, 600
    low, high = math.log10(500), math.log10(200_000)
    place = lambda value: left + (math.log10(value) - low) / (high - low) * width  # noqa: E731
    y = 74
    top = y
    for label, bars in rows:
        body.append(text(left - 10, y + len(bars) * 9 + 4, label, INK, 12, "end"))  # noqa: F405
        for shown, color, value in bars:
            body.append(rect(left, y, place(value) - left, 14, color, None, 2))
            pretty = f"{value / 1000:.2f}ms" if value < 10_000 else f"{value / 1000:.0f}ms"
            body.append(text(place(value) + 6, y + 11, pretty, INK, 11, "start"))  # noqa: F405
            y += 18
        y += 12
    for tick in (1000, 10_000, 100_000):
        body.append(line(place(tick), top - 4, place(tick), y - 8, "#e5e7eb", 1))  # noqa: F405
        body.append(text(place(tick), y + 8, f"{tick // 1000}ms", MUTED, 11))  # noqa: F405
    body.append(text(880, 52, "가로축은 로그", MUTED, 11, "end"))  # noqa: F405
    svg(OUT / "kernels.svg", 900, y + 22, body)  # noqa: F405


def fig_threads():
    """스레드 수에 따른 언어 모델의 속도."""
    series = []
    for name, shown, color in BROWSERS:
        points = {}
        for parts in lines(f"{name}.txt"):
            if parts[0] == "llm-chat" and "_threads" in parts[1] and parts[6] == "long":
                points[int(parts[3])] = number(parts, "per-s")
        if points:
            series.append((shown, color, points))
    native = {}
    for file, label in [("native_1.txt", "맥에서 그대로, 스레드 1"), ("native.txt", "맥에서 그대로, 엔진이 고른 만큼")]:
        for parts in lines(file):
            if parts[0] == "llm-chat" and parts[4] == "long":
                native[label] = number(parts, "per-s")
    if not series:
        return
    body = [text(20, 28, "SmolLM2-135M이 초당 쓰는 토큰: 브라우저의 스레드 수에 따라", INK, 14, "start", "bold")]  # noqa: F405
    left, top, width, height = 70, 60, 560, 260
    highest = max([max(points.values()) for _, _, points in series] + list(native.values())) * 1.08
    xs = {1: left + 20, 2: left + 20 + (width - 40) / 3, 4: left + 20 + 2 * (width - 40) / 3, 8: left + width - 20}
    place = lambda value: top + height - value / highest * height  # noqa: E731
    step = 100 if highest > 300 else 50
    for tick in range(0, int(highest) + 1, step):
        body.append(line(left, place(tick), left + width, place(tick), "#e5e7eb", 1))  # noqa: F405
        body.append(text(left - 8, place(tick) + 4, str(tick), MUTED, 11, "end"))  # noqa: F405
    for threads, x in xs.items():
        body.append(text(x, top + height + 18, str(threads), MUTED, 11))  # noqa: F405
    body.append(text(left + width / 2, top + height + 36, "스레드의 수", MUTED, 11))  # noqa: F405
    body.append(text(left - 8, top - 10, "토큰/초", MUTED, 11, "end"))  # noqa: F405
    for label, value in native.items():
        body.append(line(left, place(value), left + width, place(value), MUTED, 1.5, False, "6 4"))  # noqa: F405
        body.append(text(left + width + 8, place(value) + 4, f"{label} ({value:.0f})", MUTED, 11, "start"))  # noqa: F405
    for index, (shown, color, points) in enumerate(series):
        ordered = sorted(points.items())
        body.append(path([(xs[t], place(v)) for t, v in ordered], color))
        for t, v in ordered:
            body.append(f"<circle cx='{xs[t]:.1f}' cy='{place(v):.1f}' r='4' fill='{color}' stroke='#ffffff' stroke-width='1.5'/>")
        body.append(rect(left + width + 8, top + 150 + index * 20, 14, 10, color))
        best = max(points.items(), key=lambda item: item[1])
        body.append(text(left + width + 28, top + 160 + index * 20, f"{shown} (가장 빠른 것: {best[0]}개에서 {best[1]:.0f})", INK, 12, "start"))  # noqa: F405
    svg(OUT / "threads.svg", 900, top + height + 50, body)  # noqa: F405


def fig_memory():
    """모델 하나가 지나가는 자리들."""
    body = [text(20, 28, "269MB짜리 파일이 지나가는 자리: 같은 바이트가 여러 번 있다", INK, 14, "start", "bold")]  # noqa: F405
    places = [
        ("Hugging Face", "파일 269MB", GRAY_BG, MUTED),  # noqa: F405
        ("브라우저의 캐시 저장소", "디스크에 269MB", GREEN_BG, GREEN),  # noqa: F405
        ("JavaScript의 배열", "메모리에 269MB", BLUE_BG, BLUE),  # noqa: F405
        ("모듈의 메모리", "281MB (가중치 + 엔진)", ORANGE_BG, ORANGE),  # noqa: F405
    ]
    x = 20
    for index, (title, sub, fill, stroke) in enumerate(places):
        body.extend(box(x, 56, 190, 62, title, sub, fill, stroke))  # noqa: F405
        if index:
            body.append(line(x - 30, 87, x - 4, 87, INK, 1.5, True))  # noqa: F405
        x += 222
    notes = [
        (115, "버전을 고정한 주소", "CORS를 허락한다"),
        (337, "다음에는 여기에서 꺼낸다", "브라우저가 지울 수 있다"),
        (559, "엔진이 읽고 나면 내려놓는다", "그 전까지 두 벌이다"),
        (781, "자라기만 하고 줄지 않는다", "한도는 4GB (스레드 빌드 1GB)"),
    ]
    for x, first, second in notes:
        body.append(text(x, 140, first, INK, 12))  # noqa: F405
        body.append(text(x, 158, second, MUTED, 11))  # noqa: F405
    svg(OUT / "memory.svg", 900, 176, body)  # noqa: F405


if __name__ == "__main__":
    fig_stack()
    fig_isolation()
    fig_kernels()
    fig_threads()
    fig_memory()
