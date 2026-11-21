#!/usr/bin/env python3
"""Apply line-range replacements to a Markdown file and rewrap the prose.

Usage: apply_edits.py FILE SPEC

SPEC holds blocks of the form

    @@ START-END
    replacement text

START and END are 1-based line numbers of FILE as it is before any block
is applied; the whole range is replaced.  In the replacement, every
non-empty line is one paragraph or one list item and is wrapped to WIDTH
display columns (a Hangul syllable counts as two).  Lines that start a
heading, table row, image, HTML block or fenced code are copied as they
are.  An empty replacement deletes the range.
"""
import re
import sys
import unicodedata

WIDTH = 72
ATOM = re.compile(r"`[^`]*`|!?\[[^\]]*\]\([^)]*\)|<sup>[^<]*</sup>|\[\^[^\]]*\]")


def width(text):
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def tokens(text):
    """Split on spaces, but keep code spans and links in one piece."""
    shielded = ATOM.sub(lambda m: m.group(0).replace(" ", "\0"), text)
    return [t.replace("\0", " ") for t in shielded.split(" ") if t]


def wrap(line):
    stripped = line.lstrip()
    if not stripped or stripped.startswith(("#", "|", "![", "<", "```")):
        return [line]
    m = re.match(r"(\s*)((?:[-*]|\d+\.)\s+|\[\^[^\]]+\]:\s+)?(.*)", line)
    indent, bullet, body = m.group(1), m.group(2) or "", m.group(3)
    hang = 4 if bullet.startswith("[^") else width(bullet)
    lines, cur = [], ""
    for tok in tokens(body):
        prefix = indent + (bullet if not lines else " " * hang)
        if not cur:
            cur = prefix + tok
        elif width(cur) + 1 + width(tok) > WIDTH:
            lines.append(cur)
            cur = indent + " " * hang + tok
        else:
            cur += " " + tok
    lines.append(cur)
    return lines


def main():
    path, spec_path = sys.argv[1], sys.argv[2]
    src = open(path, encoding="utf-8").read().split("\n")
    blocks, cur = [], None
    for raw in open(spec_path, encoding="utf-8").read().split("\n"):
        m = re.match(r"@@ (\d+)(?:-(\d+))?\s*$", raw)
        if m:
            start = int(m.group(1))
            cur = [start, int(m.group(2) or start), []]
            blocks.append(cur)
        elif cur is not None:
            cur[2].append(raw)
    blocks.sort(key=lambda b: b[0])
    for a, b in zip(blocks, blocks[1:]):
        assert a[1] < b[0], f"overlapping ranges {a[0]}-{a[1]} and {b[0]}-{b[1]}"
    # A range must cover whole paragraphs: a range that stops one line short
    # leaves the tail of the old paragraph next to the new text.
    covered = {n for start, end, _ in blocks for n in range(start, end + 1)}

    def prose(n):
        if not 1 <= n <= len(src):
            return False
        line = src[n - 1]
        return bool(line.strip()) and not line.lstrip().startswith(("#", "|", "![", "```", ">", "<"))

    def item(n):
        return bool(re.match(r"\s*(?:[-*]|\d+\.)\s|\[\^", src[n - 1]))

    for start, end, _ in blocks:
        assert 1 <= start <= end <= len(src), f"bad range {start}-{end}"
        if prose(start) and prose(start - 1) and start - 1 not in covered and not item(start):
            sys.exit(f"range {start}-{end} starts inside a paragraph: {src[start - 2][:40]!r}")
        if prose(end) and prose(end + 1) and end + 1 not in covered and not item(end + 1):
            sys.exit(f"range {start}-{end} ends inside a paragraph: {src[end][:40]!r}")
    for start, end, body in reversed(blocks):
        while body and not body[-1].strip():
            body.pop()
        # A new heading at the end of a block replaces the next heading; if
        # that heading is outside the range it would be left behind.
        following = next((n for n in range(end + 1, len(src) + 1) if src[n - 1].strip()), None)
        if body and body[-1].startswith("#") and following and src[following - 1].startswith("#") \
                and following not in covered:
            sys.exit(f"range {start}-{end} ends with a heading but leaves {src[following - 1][:40]!r}")
        new, in_code = [], False
        for line in body:
            if line.lstrip().startswith("```"):
                in_code = not in_code
                new.append(line)
            elif in_code:
                new.append(line)
            else:
                new.extend(wrap(line))
        src[start - 1:end] = new
    open(path, "w", encoding="utf-8").write("\n".join(src))
    print(f"{path}: applied {len(blocks)} blocks")


if __name__ == "__main__":
    main()
