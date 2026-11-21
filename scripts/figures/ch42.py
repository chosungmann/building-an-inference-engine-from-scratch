"""42장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch42.py
출력: book/src/images/ch42/*.svg

routing.svg 는 bench/ch42/ 에 남긴 측정값의 중앙값을 그린다. 그 파일이
없으면 그 그림은 건너뛴다.
"""

import statistics

from common import *  # noqa: F403

OUT = IMAGES / "ch42"  # noqa: F405
BENCH = IMAGES.parents[2] / "bench/ch42"  # noqa: F405
PURPLE, PURPLE_BG = "#7c3aed", "#ede9fe"


def rect(x, y, w, h, fill, stroke=None, rx=0, width=1):
    edge = f" stroke='{stroke}' stroke-width='{width}'" if stroke else ""
    return f"<rect x='{x:.1f}' y='{y:.1f}' width='{w:.1f}' height='{h:.1f}' rx='{rx}' fill='{fill}'{edge}/>"


def fig_cluster():
    """라우터와 서버 넷."""
    body = [text(20, 28, "서버가 여러 대이면 누군가 보낼 곳을 골라야 한다", INK, 14, "start", "bold")]  # noqa: F405
    for index, (name, color) in enumerate((("가게 A의 손님", BLUE), ("가게 B의 손님", ORANGE), ("가게 A의 손님", BLUE))):  # noqa: F405
        y = 66 + index * 56
        body += box(20, y, 150, 40, name, None, "#ffffff", color)  # noqa: F405
        body.append(line(172, y + 20, 246, 150, MUTED, 1.2, True))  # noqa: F405
    body += box(250, 120, 150, 60, "라우터", "어느 서버로 보낼까", PURPLE_BG, PURPLE)
    caches = [["A의 지시문", "A의 대화 둘"], ["B의 지시문", "B의 대화"], ["(비어 있음)"], ["C의 지시문"]]
    for index, held in enumerate(caches):
        y = 48 + index * 58
        body += box(520, y, 120, 44, f"서버 {index + 1}", "모델 한 벌", GRAY_BG, MUTED)  # noqa: F405
        color = [BLUE, ORANGE, MUTED, GREEN][index]  # noqa: F405
        body.append(rect(660, y, 200, 44, "#ffffff", color, 6, 1.5))
        body.append(text(670, y + 18, "쥐고 있는 KV 캐시", MUTED, 11, "start"))  # noqa: F405
        body.append(text(670, y + 36, ", ".join(held), INK, 12, "start"))  # noqa: F405
        body.append(line(402, 150, 516, y + 22, MUTED if index else BLUE, 1.2 if index else 2, True))  # noqa: F405
    body.append(text(20, 298, "가게 A의 손님이 다시 물으면 서버 1로 가야 한다. 다른 서버로 가면 지시문과 대화를 처음부터 다시 읽는다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "cluster.svg", 900, 314, body)  # noqa: F405


def fig_parallel():
    """모델을 GPU 여럿에 나누는 세 가지."""
    body = [text(20, 28, "GPU 두 장에 무엇을 나누는가", INK, 14, "start", "bold")]  # noqa: F405
    titles = [("데이터 병렬", "요청을 나눈다"), ("파이프라인 병렬", "층을 나눈다"), ("텐서 병렬", "행렬을 나눈다")]
    for panel, (title, sub) in enumerate(titles):
        x = 30 + panel * 300
        body.append(text(x + 120, 62, title, INK, 13, "middle", "bold"))  # noqa: F405
        body.append(text(x + 120, 80, sub, MUTED, 12))  # noqa: F405
        for gpu in range(2):
            gx = x + gpu * 130
            body.append(rect(gx, 96, 112, 150, "#ffffff", MUTED, 6, 1.5))  # noqa: F405
            body.append(text(gx + 56, 262, f"GPU {gpu + 1}", MUTED, 11))  # noqa: F405
            for layer in range(4):
                ly = 106 + layer * 34
                if panel == 0:
                    body.append(rect(gx + 10, ly, 92, 26, BLUE_BG, BLUE, 3))  # noqa: F405
                    body.append(text(gx + 56, ly + 18, f"층 {layer + 1}", INK, 11))  # noqa: F405
                elif panel == 1:
                    mine = (layer < 2) == (gpu == 0)
                    if mine:
                        body.append(rect(gx + 10, ly, 92, 26, GREEN_BG, GREEN, 3))  # noqa: F405
                        body.append(text(gx + 56, ly + 18, f"층 {layer + 1}", INK, 11))  # noqa: F405
                else:
                    body.append(rect(gx + 10, ly, 92, 26, ORANGE_BG, ORANGE, 3))  # noqa: F405
                    body.append(text(gx + 56, ly + 18, f"층 {layer + 1}의 절반", INK, 11))  # noqa: F405
        if panel == 1:
            body.append(line(x + 114, 153, x + 128, 153, INK, 1.5, True))  # noqa: F405
        if panel == 2:
            for layer in range(4):
                ly = 119 + layer * 34
                body.append(line(x + 114, ly, x + 128, ly, ORANGE, 1.2, True))  # noqa: F405
                body.append(line(x + 128, ly + 6, x + 114, ly + 6, ORANGE, 1.2, True))  # noqa: F405
    notes = ["서로 주고받는 것이 없다", "은닉 벡터가 한 번 건너간다", "층마다 결과를 합친다"]
    for panel, note in enumerate(notes):
        body.append(text(30 + panel * 300 + 120, 284, note, MUTED, 12))  # noqa: F405
    svg(OUT / "parallel.svg", 920, 300, body)  # noqa: F405


def fig_split():
    """읽는 서버와 쓰는 서버."""
    body = [text(20, 28, "글을 읽는 서버와 답을 쓰는 서버를 따로 둔다", INK, 14, "start", "bold")]  # noqa: F405
    body += box(20, 110, 130, 50, "요청", "긴 글 + 질문", "#ffffff", MUTED)  # noqa: F405
    body += box(220, 60, 200, 60, "읽는 서버", "프롬프트를 한 번에 읽는다", ORANGE_BG, ORANGE)  # noqa: F405
    body += box(220, 170, 200, 60, "쓰는 서버", "토큰을 하나씩 쓴다", GREEN_BG, GREEN)  # noqa: F405
    body.append(line(152, 128, 216, 96, INK, 1.5, True))  # noqa: F405
    body.append(text(176, 96, "1", INK, 12, "middle", "bold"))  # noqa: F405
    body.append(f"<path d='M320,122 L320,166' fill='none' stroke='{PURPLE}' stroke-width='2.5' marker-end='url(#arrow-{INK[1:]})'/>")  # noqa: F405
    body.append(rect(336, 130, 150, 28, PURPLE_BG, PURPLE, 4, 1.5))
    body.append(text(411, 149, "2  KV 캐시를 넘긴다", INK, 12))  # noqa: F405
    body.append(line(422, 200, 520, 200, INK, 1.5, True))  # noqa: F405
    body.append(text(470, 192, "3", INK, 12, "middle", "bold"))  # noqa: F405
    body += box(524, 174, 150, 52, "답", "흘러나온다", "#ffffff", MUTED)  # noqa: F405
    body.append(text(700, 84, "계산을 꽉 채워 쓴다.", ORANGE, 12, "start"))  # noqa: F405
    body.append(text(700, 102, "스텝이 길어도 된다.", ORANGE, 12, "start"))  # noqa: F405
    body.append(text(700, 196, "다른 요청들의 답도 함께 쓴다.", GREEN, 12, "start"))  # noqa: F405
    body.append(text(700, 214, "스텝이 고르고 짧아야 한다.", GREEN, 12, "start"))  # noqa: F405
    body.append(text(20, 268, "넘어가는 것은 글이 아니라 KV 캐시다. 토큰 하나에 수십 킬로바이트여서, 넘기는 길이 빨라야 한다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "split.svg", 920, 284, body)  # noqa: F405


def rows():
    out = []
    for path in sorted(BENCH.glob("round_*.txt")):
        out += [row.split() for row in path.read_text(encoding="utf-8").splitlines()]
    return out


def fig_routing(data, model, name):
    """보내는 방법마다: 캐시에서 온 비율과 첫 글까지의 시간."""
    policies = [("round_robin", "차례대로"), ("least_busy", "가장 한가한 서버로"), ("prefix_hash", "앞부분의 해시로"), ("longest_prefix", "가장 길게 겹치는 서버로")]
    found = {}
    for policy, _ in policies:
        hit, first = [], []
        by_round = {}
        for parts in data:
            if parts[:4] != [model, "keep4", "route", policy]:
                continue
            if parts[4] == "차례" and parts[5] in ("2", "3"):
                by_round.setdefault(len(first), []).append((int(parts[parts.index("읽음") + 1]), int(parts[parts.index("캐시") + 1])))
            if parts[4] == "전체":
                first.append(float(parts[parts.index("첫글") + 1]))
                read = sum(r for r, _ in by_round.get(len(first) - 1, []))
                cached = sum(c for _, c in by_round.get(len(first) - 1, []))
                if read:
                    hit.append(cached / read)
        if first and hit:
            found[policy] = (statistics.median(hit), statistics.median(first))
    if len(found) < len(policies):
        return
    body = [text(20, 28, f"보내는 방법에 따라: 서버 넷, 대화 열여섯 ({name})", INK, 14, "start", "bold")]  # noqa: F405
    body.append(text(430, 56, "캐시에서 온 토큰 (둘째·셋째 차례)", INK, 12, "middle", "bold"))  # noqa: F405
    body.append(text(790, 56, "첫 글까지, 가운데 값 (ms)", INK, 12, "middle", "bold"))  # noqa: F405
    slowest = max(first for _, first in found.values())
    for index, (policy, label) in enumerate(policies):
        hit, first = found[policy]
        y = 72 + index * 38
        body.append(text(200, y + 17, label, INK, 12, "end"))  # noqa: F405
        body.append(rect(210, y, 440, 24, GRAY_BG))  # noqa: F405
        body.append(rect(210, y, 440 * hit, 24, GREEN))  # noqa: F405
        body.append(text(210 + 440 * hit + 6, y + 17, f"{hit * 100:.0f}%", INK, 12, "start"))  # noqa: F405
        w = 190 * first / slowest
        body.append(rect(700, y, max(w, 2), 24, ORANGE))  # noqa: F405
        body.append(text(700 + w + 6, y + 17, f"{first:.0f}", INK, 12, "start"))  # noqa: F405
    note = 72 + 4 * 38 + 18
    body.append(text(20, note, "서버의 계산은 네 경우 모두 같다. 달라진 것은 요청이 어느 서버로 갔는가뿐이다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "routing.svg", 960, note + 16, body)  # noqa: F405


if __name__ == "__main__":
    fig_cluster()
    fig_parallel()
    fig_split()
    data = rows()
    if data:
        models = {parts[0] for parts in data if parts}
        model, name = ("llama32_1b", "Llama 3.2 1B") if "llama32_1b" in models else ("smollm2_135m", "SmolLM2-135M")
        fig_routing(data, model, name)
