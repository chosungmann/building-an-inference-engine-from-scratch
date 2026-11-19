"""33장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch33.py
출력: book/src/images/ch33/*.svg

mel.svg 는 내려받은 정답 파일(models/whisper/reference.safetensors)에 든
실제 스펙트로그램을 그린다. 그 파일이 없으면 그 그림은 건너뛴다.
"""

import base64
import io
import json
import struct

from common import *  # noqa: F403

OUT = IMAGES / "ch33"  # noqa: F405
PURPLE, PURPLE_BG = "#7c3aed", "#ede9fe"
REFERENCE = IMAGES.parents[2] / "models/whisper/reference.safetensors"  # noqa: F405


def rect(x, y, w, h, fill, stroke=None, rx=0, width=1):
    edge = f" stroke='{stroke}' stroke-width='{width}'" if stroke else ""
    return f"<rect x='{x:.1f}' y='{y:.1f}' width='{w:.1f}' height='{h:.1f}' rx='{rx}' fill='{fill}'{edge}/>"


def read_mel():
    """정답 파일에서 jfk.mel (80, 3000)을 읽는다. 외부 패키지 없이."""
    with open(REFERENCE, "rb") as file:
        (length,) = struct.unpack("<Q", file.read(8))
        header = json.loads(file.read(length))
        start, end = header["jfk.mel"]["data_offsets"]
        file.seek(8 + length + start)
        values = struct.unpack(f"<{(end - start) // 4}f", file.read(end - start))
    return [values[row * 3000 : (row + 1) * 3000] for row in range(80)]


def png(rows, width, height):
    """값의 표를 흑백 PNG로 만든다. 큰 값이 어둡다. 표준 라이브러리만 쓴다."""
    import zlib

    low = min(min(row) for row in rows)
    high = max(max(row) for row in rows)
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        row = rows[len(rows) - 1 - y * len(rows) // height]
        for x in range(width):
            value = row[x * len(row) // width]
            raw.append(int(255 - 235 * (value - low) / (high - low)))

    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    out = io.BytesIO()
    out.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b""))
    return base64.b64encode(out.getvalue()).decode()


def fig_mel():
    """소리가 멜 스펙트로그램이 되기까지, 그리고 실제 그림."""
    body = [text(20, 28, "소리가 멜 스펙트로그램이 되기까지", INK, 14, "start", "bold")]  # noqa: F405
    steps = [
        ("표본들", "1초에 16,000개", GRAY_BG, MUTED),  # noqa: F405
        ("토막", "400개씩, 160개 간격", BLUE_BG, BLUE),  # noqa: F405
        ("창을 곱함", "양 끝을 0으로", BLUE_BG, BLUE),  # noqa: F405
        ("푸리에 변환", "세기 201개", ORANGE_BG, ORANGE),  # noqa: F405
        ("멜 필터", "201칸 → 80칸", GREEN_BG, GREEN),  # noqa: F405
        ("로그", "자르고 옮기고", PURPLE_BG, PURPLE),
    ]
    x, y, w, h, gap = 20, 50, 140, 46, 22
    for index, (title, sub, fill, stroke) in enumerate(steps):
        left = x + index * (w + gap)
        body += box(left, y, w, h, title, sub, fill, stroke)  # noqa: F405
        if index:
            body.append(line(left - gap + 2, y + h / 2, left - 3, y + h / 2, INK, 1.2, arrow=True))  # noqa: F405
    rows = read_mel()
    # 소리가 있는 앞의 11초(토막 1,100개)만 그린다.
    shown = [row[:1100] for row in rows]
    px, py, pw, ph = 70, 140, 880, 200
    data = png(shown, 440, 80)
    body.append(f"<image x='{px}' y='{py}' width='{pw}' height='{ph}' preserveAspectRatio='none' href='data:image/png;base64,{data}'/>")
    body.append(rect(px, py, pw, ph, "none", MUTED, 0, 1))  # noqa: F405
    for second in range(0, 12, 2):
        sx = px + second * 100 / 1100 * pw
        body.append(line(sx, py + ph, sx, py + ph + 5, MUTED, 1))  # noqa: F405
        body.append(text(sx, py + ph + 18, f"{second}초", INK, 11))  # noqa: F405
    body.append(text(px - 8, py + ph - 2, "낮은 소리", INK, 11, "end"))  # noqa: F405
    body.append(text(px - 8, py + 10, "높은 소리", INK, 11, "end"))  # noqa: F405
    body.append(text(px + pw + 10, py + 14, "멜 칸 80개", MUTED, 11, "start"))  # noqa: F405
    body.append(text(px + pw + 10, py + 32, "어두울수록", MUTED, 11, "start"))  # noqa: F405
    body.append(text(px + pw + 10, py + 46, "세다", MUTED, 11, "start"))  # noqa: F405
    note = py + ph + 44
    body.append(text(20, note, "jfk.wav의 실제 스펙트로그램(앞의 11초). 낱말마다 덩어리가 보이고, 덩어리 사이의 밝은 띠가 말을 쉰 곳이다. 모델은 30초 가운데", MUTED, 12, "start"))  # noqa: F405
    body.append(text(20, note + 18, "나머지 19초를 침묵으로 채운 (80, 3000) 그림을 받는다. 가로줄 무늬는 목소리의 높이와 그 배수들이다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "mel.svg", 1060, note + 34, body)  # noqa: F405


def fig_whisper():
    """Whisper의 인코더와 디코더."""
    body = [text(20, 28, "Whisper: 소리를 한 번 읽고, 글을 한 토큰씩 쓴다", INK, 14, "start", "bold")]  # noqa: F405
    # 인코더.
    ex, ey = 40, 60
    body.append(text(ex, ey - 8, "인코더 (소리 하나에 한 번)", INK, 12, "start", "bold"))  # noqa: F405
    parts = [
        ("멜 스펙트로그램", "(80, 3000)", GRAY_BG, MUTED),  # noqa: F405
        ("합성곱 둘", "길이를 절반으로", BLUE_BG, BLUE),  # noqa: F405
        ("자리 벡터를 더함", "(1500, 384)", GREEN_BG, GREEN),  # noqa: F405
        ("블록 4개", "어텐션, MLP", BLUE_BG, BLUE),  # noqa: F405
        ("소리의 벡터 1,500개", "하나가 20밀리초", ORANGE_BG, ORANGE),  # noqa: F405
    ]
    w, h, gap = 200, 44, 16
    for index, (title, sub, fill, stroke) in enumerate(parts):
        top = ey + index * (h + gap)
        body += box(ex, top, w, h, title, sub, fill, stroke)  # noqa: F405
        if index:
            body.append(line(ex + w / 2, top - gap + 1, ex + w / 2, top - 3, INK, 1.2, arrow=True))  # noqa: F405
    out_y = ey + 4 * (h + gap)
    # 디코더.
    dx, dy = 420, 60
    body.append(text(dx, dy - 8, "디코더 (토큰마다 한 번)", INK, 12, "start", "bold"))  # noqa: F405
    body += box(dx, dy, 240, h, "지금까지 쓴 토큰들", "시작, 영어, 받아쓰기, … And, so", GRAY_BG, MUTED)  # noqa: F405
    block_y = dy + h + gap
    body.append(rect(dx - 10, block_y - 6, 260, 3 * (h + 8) + 10, "#ffffff", MUTED, 8, 1.2))  # noqa: F405
    inner = [
        ("셀프 어텐션", "쓴 글을 본다 (KV 캐시)", BLUE_BG, BLUE),  # noqa: F405
        ("교차 어텐션", "소리의 벡터를 본다", ORANGE_BG, ORANGE),  # noqa: F405
        ("MLP", "", BLUE_BG, BLUE),  # noqa: F405
    ]
    for index, (title, sub, fill, stroke) in enumerate(inner):
        top = block_y + index * (h + 8)
        body += box(dx, top, 240, h, title, sub or None, fill, stroke)  # noqa: F405
    body.append(text(dx + 262, block_y + 16, "× 4", INK, 13, "start", "bold"))  # noqa: F405
    cross_y = block_y + (h + 8) + h / 2
    body.append(line(dx + 120, dy + h + 1, dx + 120, block_y - 8, INK, 1.2, arrow=True))  # noqa: F405
    score_y = block_y + 3 * (h + 8) + 18
    body += box(dx, score_y, 240, h, "다음 토큰의 점수", "51,865개", PURPLE_BG, PURPLE)  # noqa: F405
    body.append(line(dx + 120, score_y - 12, dx + 120, score_y - 3, INK, 1.2, arrow=True))  # noqa: F405
    # 인코더의 출력에서 교차 어텐션으로.
    body.append(f"<path d='M {ex + w + 2} {out_y + h / 2} C {ex + w + 110} {out_y + h / 2}, {dx - 110} {cross_y}, {dx - 3} {cross_y}' fill='none' stroke='{ORANGE}' stroke-width='1.8' marker-end='url(#arrow-{ORANGE[1:]})'/>")  # noqa: F405
    body.append(text(ex + w + 14, out_y + h / 2 + 22, "키와 값은 한 번만 만들어 둔다", ORANGE, 11, "start"))  # noqa: F405
    # 고른 토큰이 다시 입력으로.
    loop_x = dx + 300
    body.append(f"<path d='M {dx + 242} {score_y + h / 2} L {loop_x + 40} {score_y + h / 2} L {loop_x + 40} {dy + h / 2} L {dx + 244} {dy + h / 2}' fill='none' stroke='{PURPLE}' stroke-width='1.5' stroke-dasharray='5 3' marker-end='url(#arrow-{INK[1:]})'/>")  # noqa: F405
    body.append(text(loop_x + 50, (score_y + dy + h) / 2 - 8, "고른 토큰을", PURPLE, 11, "start"))
    body.append(text(loop_x + 50, (score_y + dy + h) / 2 + 8, "다음 입력으로", PURPLE, 11, "start"))
    body.append(text(loop_x + 50, (score_y + dy + h) / 2 + 30, "끝 토큰이", MUTED, 11, "start"))  # noqa: F405
    body.append(text(loop_x + 50, (score_y + dy + h) / 2 + 44, "나올 때까지", MUTED, 11, "start"))  # noqa: F405
    note = score_y + h + 30
    body.append(text(20, note, "인코더는 32장의 ViT와, 디코더는 2부의 Llama와 뼈대가 같다. 둘을 잇는 것이 교차 어텐션이다. 질의는 글에서, 키와 값은 소리에서 온다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "whisper.svg", 920, note + 18, body)  # noqa: F405


if __name__ == "__main__":
    fig_whisper()
    if REFERENCE.exists():
        fig_mel()
