"""부록 F의 표를 장들의 각주에서 만든다.

실행: python3 scripts/build_references.py [fetch]
출력: book/src/appendix/generated/papers.md, repositories.md, counts.md

각 장 끝의 각주(`[^이름]: ...`)를 모두 읽어,
- arXiv 번호가 있는 논문을 번호로 모으고(같은 논문을 여러 장이 인용한다),
- GitHub 저장소를 저장소별로 모은다.

논문의 제목과 저자와 연도는 arXiv가 알려 주는 값을 쓴다. `fetch`를 붙이면
arXiv의 API에 물어 `reference/arxiv_titles.tsv`를 다시 쓰고, 붙이지 않으면
그 파일을 읽는다. 표준 라이브러리만 쓴다.
"""

import collections
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "book/src"
OUT = BOOK / "appendix/generated"
TITLES = ROOT / "reference/arxiv_titles.tsv"
FOOTNOTE = re.compile(r"^\[\^([^\]]+)\]: (.*?)(?=^\[\^|\Z|^\S)", re.S | re.M)
ARXIV = re.compile(r"(?:arXiv:\s?|arxiv\.org/abs/)(\d{4}\.\d{4,5})")
GITHUB = re.compile(r"github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)")


def chapters():
    """(장 번호, 파일의 길, 글)을 장 번호 순으로."""
    found = []
    for path in BOOK.glob("part*/ch*.md"):
        number = int(re.search(r"ch(\d+)", path.name).group(1))
        found.append((number, path.relative_to(BOOK).as_posix(), path.read_text(encoding="utf-8")))
    return sorted(found)


def footnotes():
    for number, path, text in chapters():
        for match in FOOTNOTE.finditer(text):
            yield number, path, " ".join(match.group(2).split())


def fetch(identifiers):
    rows = []
    for start in range(0, len(identifiers), 40):
        batch = identifiers[start : start + 40]
        url = f"http://export.arxiv.org/api/query?id_list={','.join(batch)}&max_results={len(batch)}"
        with urllib.request.urlopen(url, timeout=60) as response:
            feed = response.read().decode("utf-8")
        for entry in re.findall(r"<entry>(.*?)</entry>", feed, re.S):
            identifier = re.search(r"<id>.*?/abs/(\d{4}\.\d{4,5})", entry).group(1)
            title = " ".join(re.search(r"<title>(.*?)</title>", entry, re.S).group(1).split())
            authors = re.findall(r"<name>(.*?)</name>", entry)
            year = re.search(r"<published>(\d{4})", entry).group(1)
            rows.append((identifier, year, authors[0], str(len(authors)), title))
    found = {row[0] for row in rows}
    missing = [identifier for identifier in identifiers if identifier not in found]
    if missing:
        raise SystemExit(f"arXiv가 모르는 번호: {missing}")
    TITLES.write_text("".join("\t".join(row) + "\n" for row in sorted(rows)), encoding="utf-8")


def link(number, path):
    return f"[{number}](../{path})"


def main():
    papers, repositories = collections.defaultdict(dict), collections.defaultdict(dict)
    total, per_part = 0, collections.Counter()
    for number, path, body in footnotes():
        total += 1
        per_part[path.split("/")[0]] += 1
        for identifier in ARXIV.findall(body):
            papers[identifier][number] = path
        for repository in GITHUB.findall(body):
            repository = repository.removesuffix(".git").rstrip(".")
            repositories[repository.lower()][number] = (path, repository)
    if "fetch" in sys.argv[1:]:
        fetch(sorted(papers))
    known = {}
    for line in TITLES.read_text(encoding="utf-8").splitlines():
        identifier, year, first, count, title = line.split("\t")
        known[identifier] = (int(year), first, int(count), title)
    unknown = sorted(set(papers) - set(known))
    if unknown:
        raise SystemExit(f"reference/arxiv_titles.tsv에 없는 번호: {unknown} (fetch를 붙여 다시 돌린다)")

    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["| 논문 | 연도 | arXiv | 인용한 장 |", "|---|---:|---|---|"]
    ordered = sorted(papers, key=lambda identifier: (min(papers[identifier]), known[identifier][0], identifier))
    for identifier in ordered:
        year, first, count, title = known[identifier]
        who = first if count == 1 else f"{first} 외"
        cited = ", ".join(link(number, path) for number, path in sorted(papers[identifier].items()))
        title = title.replace("|", "\\|")
        lines.append(f"| {who}, \"{title}\" | {year} | [{identifier}](https://arxiv.org/abs/{identifier}) | {cited} |")
    (OUT / "papers.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = ["| 저장소 | 인용한 장 |", "|---|---|"]
    ordered = sorted(repositories, key=lambda name: (-len(repositories[name]), name))
    for name in ordered:
        cited = repositories[name]
        shown = next(iter(cited.values()))[1]
        where = ", ".join(link(number, path) for number, (path, _) in sorted(cited.items()))
        lines.append(f"| [{shown}](https://github.com/{shown}) | {where} |")
    (OUT / "repositories.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = ["| 부 | 각주의 수 |", "|---|---:|"]
    for part in sorted(per_part):
        lines.append(f"| {part[4:]}부 | {per_part[part]} |")
    lines.append(f"| 모두 | {total} |")
    (OUT / "counts.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"각주 {total}개, 논문 {len(papers)}편, 저장소 {len(repositories)}곳")


if __name__ == "__main__":
    main()
