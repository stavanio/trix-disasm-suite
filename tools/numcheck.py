import csv, re, sys

NUM = re.compile(r"(?<![\w.{])\d+(?:[.,]\d+)*(?:\\?%|~?\$?\\mu\$?s)?")
SKIP = re.compile(r"\\(label|ref|suppref|suppsection|cite|includegraphics|url|path|setlength|setcounter|renewcommand|usepackage|documentclass)\{[^}]*\}")


def body_lines(path):
    lines = open(path, encoding="utf-8").read().split("\n")
    start = next(i for i, l in enumerate(lines) if r"\begin{abstract}" in l)
    out = []
    for i, line in enumerate(lines[start:], start + 1):
        if r"\begin{thebibliography}" in line:
            break
        line = re.sub(r"(?<!\\)%.*$", "", line)
        out.append((i, SKIP.sub("", line)))
    return out


def inventory(path):
    rows = []
    for n, line in body_lines(path):
        for m in NUM.finditer(line):
            rows.append((m.group(0).replace("\\", ""), n, line.strip()[:140]))
    return rows


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "inventory":
        rows = inventory(sys.argv[2])
        w = csv.writer(sys.stdout)
        w.writerow(["value", "line", "context"])
        w.writerows(rows)
    elif len(sys.argv) == 3:
        base = {r[0] for r in inventory(sys.argv[1])}
        new = {r[0] for r in inventory(sys.argv[2])}
        added = sorted(new - base)
        dropped = sorted(base - new)
        print("NOT IN BASELINE (must be justified or fixed):")
        for v in added:
            print("  ", v)
        print("IN BASELINE, ABSENT FROM REWRITE (confirm intentional):")
        for v in dropped:
            print("  ", v)
        sys.exit(1 if added else 0)
    else:
        sys.exit("usage: numcheck.py inventory FILE | numcheck.py BASELINE REWRITE")
