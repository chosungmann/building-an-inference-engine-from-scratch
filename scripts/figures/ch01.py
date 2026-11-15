"""1장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch01.py
출력: book/src/images/ch01/*.svg
"""

from common import *  # noqa: F403

OUT = IMAGES / "ch01"  # noqa: F405


def fig_pipeline():
    body = []
    w, h = 190, 58
    left, right, mid = 40, 410, 225
    # 왼쪽 줄: 모델 파일 → 읽기 → 준비
    body.append(text(left + w / 2, 34, "모델 파일", INK, 14, weight="bold"))
    body.append(line(left + w / 2, 42, left + w / 2, 62, MUTED, 2, arrow=True))
    body += box(left, 66, w, h, "① 읽기", "가중치와 구조를 적재", BLUE_BG, BLUE)
    body.append(line(left + w / 2, 66 + h, left + w / 2, 150, MUTED, 2, arrow=True))
    body += box(left, 154, w, h, "② 준비", "실행 계획, 메모리 확보", BLUE_BG, BLUE)
    # 오른쪽 줄: 입력 → 전처리
    body.append(text(right + w / 2, 34, "입력 (글, 사진, 소리)", INK, 14, weight="bold"))
    body.append(line(right + w / 2, 42, right + w / 2, 62, MUTED, 2, arrow=True))
    body += box(right, 66, w, h, "③ 전처리", "입력을 숫자 배열로 변환", ORANGE_BG, ORANGE)
    # 가운데: 실행 → 후처리 → 출력
    body.append(line(left + w / 2, 154 + h, mid + 50, 262, MUTED, 2, arrow=True))
    body.append(line(right + w / 2, 66 + h, mid + w - 50, 262, MUTED, 2, arrow=True))
    body += box(mid, 266, w, h, "④ 실행", "연산을 순서대로 계산", GREEN_BG, GREEN)
    body.append(line(mid + w / 2, 266 + h, mid + w / 2, 350, MUTED, 2, arrow=True))
    body += box(mid, 354, w, h, "⑤ 후처리", "숫자 배열을 결과로 변환", ORANGE_BG, ORANGE)
    body.append(line(mid + w / 2, 354 + h, mid + w / 2, 438, MUTED, 2, arrow=True))
    body.append(text(mid + w / 2, 460, "출력 (글, 분류 결과, 사진)", INK, 14, weight="bold"))
    body.append(text(mid + w + 20, 300, "시간의 대부분이", GREEN, 13, "start"))
    body.append(text(mid + w + 20, 318, "여기서 쓰인다", GREEN, 13, "start"))
    svg(OUT / "pipeline.svg", 640, 480, body)


if __name__ == "__main__":
    fig_pipeline()
    print(f"wrote {len(list(OUT.glob('*.svg')))} figures to {OUT}")
