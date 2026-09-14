#!/usr/bin/env python3
"""Turn the lily-bench records under docs/bench/<date>-<sha>/ into the tables in
docs/performance-timeline.md.

Both the records and that document are local measurement output and are not
tracked; the document is created when it is missing.

Each record directory holds a run.json (written by tools/bench/timeline.sh)
and one lily-bench JSON per cell and repeat, named p<prompt>-d<drafts>-r<n>.json.
Cells are summarised as the median over repeats, with the min and max shown so
run-to-run noise stays visible. Standard library only.

    tools/bench/summarize.py                       # print the tables
    tools/bench/summarize.py --write docs/performance-timeline.md
    tools/bench/summarize.py --only docs/bench/<date>-<sha>     # echo one run's rows
"""
import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

BEGIN = "<!-- timeline:begin -->"
END = "<!-- timeline:end -->"
CELL_RE = re.compile(r"^p(\d+)-d(\d+)-r(\d+)\.json$")


def load_run(directory: Path):
    meta_path = directory / "run.json"
    if not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text())
    cells = {}
    for path in sorted(directory.iterdir()):
        m = CELL_RE.match(path.name)
        if not m:
            continue
        prompt, drafts = int(m.group(1)), int(m.group(2))
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            print(f"warning: skipping unreadable {path}: {exc}", file=sys.stderr)
            continue
        results = data.get("results", {})
        decode = results.get("decode", {})
        cells.setdefault((prompt, drafts), []).append({
            "prefill_tok_s": results.get("prefill", {}).get("tok_s"),
            "decode_tok_s": decode.get("tok_s"),
            "drafted": decode.get("drafted", 0),
            "accepted": decode.get("accepted", 0),
            "digest": decode.get("token_digest"),
        })
    return {"dir": directory, "meta": meta, "cells": cells}


def summarise(samples, key):
    values = [s[key] for s in samples if s.get(key) is not None]
    if not values:
        return None
    return statistics.median(values), min(values), max(values)


def fmt_cell(summary, decimals):
    if summary is None:
        return "-"
    median, lo, hi = summary
    if len({round(median, decimals), round(lo, decimals), round(hi, decimals)}) == 1:
        return f"{median:.{decimals}f}"
    return f"{median:.{decimals}f} ({lo:.{decimals}f} to {hi:.{decimals}f})"


def acceptance(samples):
    drafted = sum(s["drafted"] or 0 for s in samples)
    accepted = sum(s["accepted"] or 0 for s in samples)
    if not drafted:
        return None
    return 100.0 * accepted / drafted


def digest_note(cells, prompt, draft_values):
    """'same' when every repeat and draft setting produced the same tokens,
    otherwise which settings disagree."""
    digests = {}
    for d in draft_values:
        for s in cells.get((prompt, d), []):
            if s["digest"]:
                digests.setdefault(d, set()).add(s["digest"])
    if not digests:
        return "-"
    if any(len(v) > 1 for v in digests.values()):
        return "varies between repeats"
    distinct = {next(iter(v)) for v in digests.values()}
    return "same" if len(distinct) == 1 else "d0 differs from drafts"


def commit_time(meta):
    """ISO committer date from run.json, else from git, else the run date."""
    if meta.get("commit_time"):
        return meta["commit_time"]
    sha = meta.get("commit")
    if sha:
        try:
            out = subprocess.run(["git", "log", "-1", "--format=%cI", sha], capture_output=True, text=True, check=True).stdout.strip()
            if out:
                return out
        except Exception:
            pass
    return meta.get("date", "")


def label(run):
    meta = run["meta"]
    short = meta.get("commit_short", "?")
    if meta.get("dirty"):
        short += "-dirty"
    return f"{meta.get('date', '?')} `{short}`"


def render(runs):
    if not runs:
        return "_No records yet. Run `tools/bench/timeline.sh` on mains power._\n"
    prompts = sorted({p for run in runs for (p, _) in run["cells"]})
    draft_values = sorted({d for run in runs for (_, d) in run["cells"]})
    out = []

    out.append("### Runs\n")
    out.append("| date | commit | power | steps x repeats | what changed |")
    out.append("|------|--------|-------|-----------------|--------------|")
    for run in runs:
        meta = run["meta"]
        matrix = meta.get("matrix", {})
        note = meta.get("note") or meta.get("subject") or ""
        note = note.replace("|", "\\|")
        short = f"`{meta.get('commit_short', '?')}`" + (" (dirty)" if meta.get("dirty") else "")
        out.append(
            f"| {meta.get('date', '?')} | {short} | {meta.get('host', {}).get('power', '?')} "
            f"| {matrix.get('decode_steps', '?')} x {matrix.get('repeats', '?')} | {note} |"
        )
    out.append("")

    def table(title, key, drafts, decimals, with_acceptance=False, with_digest=False):
        out.append(f"### {title}\n")
        header = "| run | " + " | ".join(f"{p} prompt" for p in prompts)
        sep = "|-----|" + "|".join("---" for _ in prompts)
        if with_digest:
            header += " | tokens"
            sep += "|---"
        out.append(header + " |")
        out.append(sep + "|")
        for run in runs:
            row = [label(run)]
            for p in prompts:
                samples = run["cells"].get((p, drafts), [])
                cell = fmt_cell(summarise(samples, key), decimals)
                if with_acceptance and samples:
                    acc = acceptance(samples)
                    if acc is not None:
                        cell += f", {acc:.0f}% accepted"
                row.append(cell)
            if with_digest:
                notes = {digest_note(run["cells"], p, draft_values) for p in prompts if any((p, d) in run["cells"] for d in draft_values)}
                row.append("; ".join(sorted(notes)) if notes else "-")
            out.append("| " + " | ".join(row) + " |")
        out.append("")

    table("Prefill, tok/s (median over repeats, min to max when they differ)", "prefill_tok_s", 0, 0)
    table("Decode without drafts, tok/s", "decode_tok_s", 0, 1, with_digest=True)
    for d in draft_values:
        if d == 0:
            continue
        table(f"Decode with {d} draft{'s' if d != 1 else ''} per step, tok/s", "decode_tok_s", d, 1, with_acceptance=True)
    out.append(
        "Prefill is taken from the no-draft runs; the tokens column says whether the no-draft "
        "and speculative runs produced the same output (they legitimately differ on near-ties, "
        "see docs/architecture.md). Cells show the median over repeats; a range in parentheses "
        "is the min and max, which is the run-to-run noise band."
    )
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="docs/bench", help="directory holding the record directories")
    ap.add_argument("--write", metavar="FILE", help="replace the block between the timeline markers in FILE")
    ap.add_argument("--only", metavar="DIR", help="after writing, echo the rows of this record directory only")
    args = ap.parse_args()

    root = Path(args.root)
    runs = [r for r in (load_run(d) for d in sorted(root.iterdir()) if d.is_dir()) if r] if root.is_dir() else []
    # Commit order first (interleaved runs share one start time), then start time.
    runs.sort(key=lambda r: (commit_time(r["meta"]), r["meta"].get("started") or "", str(r["dir"])))
    body = render(runs)

    if args.write:
        path = Path(args.write)
        text = path.read_text() if path.is_file() else "# Performance timeline\n\n"
        if BEGIN in text and END in text:
            head, rest = text.split(BEGIN, 1)
            _, tail = rest.split(END, 1)
            text = f"{head}{BEGIN}\n{body}{END}{tail}"
        else:
            text = text.rstrip("\n") + f"\n\n{BEGIN}\n{body}{END}\n"
        path.write_text(text)
        print(f"wrote {path}")

    if args.only:
        only = Path(args.only).resolve()
        selected = [r for r in runs if r["dir"].resolve() == only]
        print(render(selected) if selected else f"no record at {only}")
    elif not args.write:
        print(body, end="")


if __name__ == "__main__":
    main()
