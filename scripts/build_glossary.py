"""부록 E의 용어집 표를 만든다.

실행: python3 scripts/build_glossary.py
출력: book/src/appendix/generated/glossary.md, glossary_english.md

reference/glossary.tsv 를 읽어 한국어 가나다순 표와 영어 알파벳순
표를 만든다. 표준 라이브러리만 쓴다.
"""

import re
from pathlib import Path

BOOK = Path(__file__).resolve().parents[1] / "book/src"
OUT = BOOK / "appendix/generated"


def main():
    paths = {}
    for path in BOOK.glob("part*/ch*.md"):
        paths[int(re.search(r"ch(\d+)", path.name).group(1))] = path.relative_to(BOOK).as_posix()
    rows = []
    for line in (BOOK.parents[1] / "reference/glossary.tsv").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        korean, english, chapter, meaning = line.split("\t")
        rows.append((korean, english, int(chapter), meaning))
    names = [row[0] for row in rows]
    repeated = {name for name in names if names.count(name) > 1}
    if repeated:
        raise SystemExit(f"두 번 나오는 낱말: {sorted(repeated)}")

    def link(chapter):
        return f"[{chapter}장](../{paths[chapter]})"

    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["| 한국어 | 영어 | 처음 나오는 곳 | 뜻 |", "|---|---|---|---|"]
    # 한글 음절은 유니코드에서 가나다순이다. 영문으로 시작하는 낱말(KV 캐시)은 앞에 온다.
    for korean, english, chapter, meaning in sorted(rows):
        lines.append(f"| {korean} | {english} | {link(chapter)} | {meaning} |")
    (OUT / "glossary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = ["| 영어 | 한국어 |", "|---|---|"]
    for korean, english, _, _ in sorted(rows, key=lambda row: row[1].lower()):
        lines.append(f"| {english} | {korean} |")
    (OUT / "glossary_english.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"낱말 {len(rows)}개")


if __name__ == "__main__":
    main()
