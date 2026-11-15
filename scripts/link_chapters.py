"""원고에서 다른 장을 언급한 곳을 링크로 바꾼다.

이미 쓰인 장만 링크한다. 어떤 장이 쓰였는지는 `book/src/SUMMARY.md`에
연결되어 있는지로 판단한다. 아직 쓰이지 않은 장을 가리키는 링크가 있으면
일반 글자로 되돌린다.

새 장을 쓴 뒤에 실행하면, 앞서 쓴 장들이 그 장을 언급한 곳에도 링크가
걸린다. 그렇게 바뀐 파일은 각 장의 원래 커밋에 합친다.

실행: python3 scripts/link_chapters.py
"""

import os
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "book/src"
CHAPTER_FILE = re.compile(r"part\d+/ch(\d+)_[a-z0-9_]+\.md")
APPENDIX_FILE = re.compile(r"appendix/([a-f])_[a-z0-9_]+\.md")


def written_targets():
    """SUMMARY.md에 연결된 장과 부록을 `{번호 또는 글자: 경로}`로 돌려준다."""
    summary = (SRC / "SUMMARY.md").read_text(encoding="utf-8")
    chapters, appendices = {}, {}
    for target in re.findall(r"\]\(([^)]+\.md)\)", summary):
        if match := CHAPTER_FILE.fullmatch(target):
            chapters[int(match[1])] = target
        elif match := APPENDIX_FILE.fullmatch(target):
            appendices[match[1].upper()] = target
    return chapters, appendices


def relink(path, chapters, appendices):
    """파일 하나의 장·부록 언급을 현재 상태에 맞게 다시 링크한다."""
    relative = path.relative_to(SRC).as_posix()
    own = int(match[1]) if (match := CHAPTER_FILE.fullmatch(relative)) else None

    def href(target):
        return os.path.relpath(SRC / target, path.parent)

    def chapter(match):
        number = int(match[1])
        if number == own or number not in chapters:
            return match[0]
        return f"[{number}장]({href(chapters[number])})"

    def appendix(match):
        if match[1] not in appendices:
            return match[0]
        return f"[부록 {match[1]}]({href(appendices[match[1]])})"

    lines, in_fence = [], False
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.startswith("```"):
            in_fence = not in_fence
        # 코드, 제목, 각주 정의와 그 이어지는 줄은 건드리지 않는다.
        if not (in_fence or line.startswith(("#", "[^", "    "))):
            # 먼저 기존 링크를 모두 풀고, 쓰인 장만 다시 건다.
            line = re.sub(r"\[(\d+장)\]\([^)]*\)", r"\1", line)
            line = re.sub(r"\[(부록 [A-F])\]\([^)]*\)", r"\1", line)
            line = re.sub(r"(?<![\d~.])(\d+)장", chapter, line)
            line = re.sub(r"부록 ?([A-F])(?![\w])", appendix, line)
        lines.append(line)
    updated = "\n".join(lines)
    if updated != path.read_text(encoding="utf-8"):
        path.write_text(updated, encoding="utf-8")
        return True
    return False


def main():
    chapters, appendices = written_targets()
    files = [SRC / "preface.md", *(SRC / target for target in chapters.values())]
    changed = [f for f in files if f.exists() and relink(f, chapters, appendices)]
    for path in changed:
        print(f"updated {path.relative_to(SRC)}")
    print(f"{len(chapters)} chapters written, {len(changed)} files updated")


if __name__ == "__main__":
    main()
