#!/usr/bin/env python3
"""Builds a real-text benchmark corpus from source trees: Markdown files
interleaved with code (two code files per document), each file preceded by
a `===== path =====` line, hidden directories and build output skipped.

    tools/bench/corpus.py --out docs/bench/prompts/corpus.txt ~/projects/a ~/projects/b

`lily-bench --prompt-text FILE` takes the first `--prompt-len` tokens of
such a file; `--offset BYTES` cuts a prompt file at a different place so
several prompts of the same length can be drawn from one corpus:

    tools/bench/corpus.py --out p1.txt --offset 900000 --limit 400000 corpus.txt
"""
import argparse
import os
import sys

EXTS = ('.md', '.rs', '.py', '.ts', '.tsx', '.txt', '.metal', '.toml', '.go', '.js', '.yaml')
SKIP_DIRS = {'node_modules', 'target', 'dist', 'bench'}


def collect(root):
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith('.') and d not in SKIP_DIRS)
        for name in sorted(filenames):
            if name.endswith(EXTS):
                files.append(os.path.join(dirpath, name))
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=6_000_000, help='bytes of text to keep')
    ap.add_argument('--offset', type=int, default=0, help='with one input file: start at this byte')
    ap.add_argument('inputs', nargs='+')
    args = ap.parse_args()
    if len(args.inputs) == 1 and os.path.isfile(args.inputs[0]):
        text = open(args.inputs[0], encoding='utf-8').read()
        open(args.out, 'w').write(text[args.offset:args.offset + args.limit])
        return
    files = [f for root in args.inputs for f in collect(root)]
    docs = [f for f in files if f.endswith('.md')]
    code = [f for f in files if not f.endswith('.md')]
    order, i, j = [], 0, 0
    while i < len(docs) or j < len(code):
        if i < len(docs):
            order.append(docs[i]); i += 1
        for _ in range(2):
            if j < len(code):
                order.append(code[j]); j += 1
    total = 0
    with open(args.out, 'w') as out:
        for f in order:
            try:
                t = open(f, encoding='utf-8').read()
            except (UnicodeDecodeError, OSError):
                continue
            out.write(f"\n\n===== {f} =====\n\n" + t)
            total += len(t)
            if total > args.limit:
                break
    print(f"{args.out}: {total} bytes from {len(order)} files", file=sys.stderr)


if __name__ == '__main__':
    main()
