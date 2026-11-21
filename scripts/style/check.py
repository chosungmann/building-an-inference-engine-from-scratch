#!/usr/bin/env python3
"""Compare an edited chapter with its committed version.

Usage: check.py FILE [REV]

Prose editing must not touch facts.  This script compares the working
copy of FILE with `git show REV:FILE` (REV defaults to HEAD) and reports

- fenced code blocks, inline code, tables, images, link targets and
  footnote markers that differ,
- numbers that appear a different number of times,
- how much of the prose changed,
- the style measurements of docs/style.md before and after.

It exits with status 1 when code, tables, images, links or footnotes
differ.  Number differences are printed for a human to judge, because
moving a gloss or joining two sentences can legitimately drop a repeated
figure.
"""
import difflib
import re
import statistics
import subprocess
import sys
import unicodedata
from collections import Counter


def split(text):
    fences = re.findall(r"^```.*?^```", text, flags=re.S | re.M)
    rest = re.sub(r"^```.*?^```", "", text, flags=re.S | re.M)
    lines = rest.split("\n")
    tables = [l for l in lines if l.startswith("|")]
    images = [l for l in lines if l.startswith("![")]
    prose_lines = [l for l in lines if not l.startswith(("|", "![", "#"))]
    heads = [l for l in lines if l.startswith("#")]
    prose = "\n".join(prose_lines)
    return fences, tables, images, heads, prose


def facts(prose, heads):
    body = prose + "\n" + "\n".join(heads)
    inline = Counter(re.findall(r"`[^`\n]+`", body))
    links = Counter(re.findall(r"\]\(([^)]+)\)", body))
    notes = Counter(re.findall(r"\[\^[^\]]+\]", body))
    plain = re.sub(r"`[^`\n]+`|\]\([^)]+\)|\[\^[^\]]+\]", " ", body)
    numbers = Counter(re.findall(r"\d[\d,.]*\d|\d", plain))
    return inline, links, notes, numbers


def measures(prose):
    paras = [re.sub(r"\s*\n\s*", " ", p.strip()) for p in re.split(r"\n\s*\n", prose)]
    paras = [p for p in paras if p and not p.startswith(("-", "*", ">", "<", "[^")) and not re.match(r"\d+\.", p)]
    text = "\n".join(paras)
    sents = [s for p in paras for s in re.split(r"(?<=[다요까]\.)\s+", p) if s.strip()]
    lens = [len(s) for s in sents] or [0]

    def n(pat):
        return len(re.findall(pat, text))

    return {
        "sentences": len(sents),
        "median length": int(statistics.median(lens)),
        "<=25 chars %": round(100 * sum(l <= 25 for l in lens) / len(lens), 1),
        ">=100 chars %": round(100 * sum(l >= 100 for l in lens) / len(lens), 1),
        "입니다. endings": n(r"입니다\."),
        "connective + comma": n(r"(?:[가-힣]고|며|지만|면서|아서|어서|해서|는데),"),
        "bold": n(r"\*\*[^*]+\*\*"),
        "것/뜻/셈입니다": n(r"(?:뜻입니다|것입니다|셈입니다)\."),
        "들 plural": n(r"들[을이은의에]? "),
        "paren gloss": n(r"[가-힣]\*{0,2}\([A-Za-z][A-Za-z \-]+\)"),
        "sup gloss": n(r"<sup>"),
        "수 있습니다": n(r"수 있습니다"),
    }


def width(text):
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def main():
    path = sys.argv[1]
    rev = sys.argv[2] if len(sys.argv) > 2 else "HEAD"
    old = subprocess.run(["git", "show", f"{rev}:./{path}"], capture_output=True, text=True, check=True).stdout
    new = open(path, encoding="utf-8").read()
    of, ot, oi, oh, op = split(old)
    nf, nt, ni, nh, np_ = split(new)
    bad = False
    for name, a, b in [("code blocks", of, nf), ("tables", ot, nt), ("images", oi, ni)]:
        if a != b:
            bad = True
            print(f"FAIL {name} differ")
            for line in list(difflib.unified_diff(a, b, lineterm="", n=0))[:12]:
                print("   ", line[:160])
    oin, ol, on, onum = facts(op, oh)
    nin, nl, nn, nnum = facts(np_, nh)
    for name, a, b, fatal in [("inline code", oin, nin, True), ("link targets", ol, nl, True),
                              ("footnote markers", on, nn, True), ("numbers", onum, nnum, False)]:
        if a != b:
            bad = bad or fatal
            gone = dict(a - b)
            came = dict(b - a)
            print(f"{'FAIL' if fatal else 'NOTE'} {name}: removed {gone} added {came}")
    ow, nw = op.split(), np_.split()
    ratio = difflib.SequenceMatcher(None, ow, nw, autojunk=False).ratio()
    print(f"prose changed: {100 * (1 - ratio):.0f}% of words")
    if len(oh) != len(nh):
        bad = True
        print(f"FAIL heading count {len(oh)} -> {len(nh)}")
    changed = sum(a != b for a, b in zip(oh, nh))
    noun = [h for h in nh if h.startswith("##") and not re.search(r"[다까]$", h.strip())]
    print(f"headings changed: {changed} of {len(nh)}; not ending in 다: {len(noun)}")
    for h in noun:
        print("   ", h)
    om, nm = measures(op), measures(np_)
    for k in om:
        print(f"  {k:20s} {om[k]:>7} -> {nm[k]:>7}")
    long = [(i + 1, width(l)) for i, l in enumerate(new.split("\n"))
            if width(l) > 76 and not l.startswith(("|", "![", "#", "[^", "    ")) and "](" not in l and "`" not in l]
    in_code, wide = False, []
    for i, l in enumerate(new.split("\n")):
        if l.startswith("```"):
            in_code = not in_code
        elif not in_code and (i + 1, width(l)) in long:
            wide.append(i + 1)
    if wide:
        print("wide prose lines:", wide[:20])
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
