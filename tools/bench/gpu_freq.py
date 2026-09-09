#!/usr/bin/env python3
"""Join a powermetrics log to timeline records: GPU clock and power per run.

Capture while tools/bench/timeline.sh runs (own terminal, needs sudo):

    sudo powermetrics --samplers gpu_power -i 1000 | \\
        grep --line-buffered -E "GPU HW active frequency|GPU Power" | \\
        while read -r line; do echo "$(date +%H:%M:%S) $line"; done > gpu-power.log

Then, for one or more record directories:

    tools/bench/gpu_freq.py gpu-power.log docs/bench/2026-09-09-*/ [--annotate]

A run's window is the timestamp in its .env.txt (UTC, taken right before the
process started) to the modification time of its result JSON (written at the
end). The table shows the median and range of the GPU clock over the seconds the GPU
was busy (power above 20 W, which excludes model loading) and the median power
over the whole window; a falling busy clock across repeats is throttling.
--annotate writes the same numbers into each result JSON under "gpu_during_run".
Standard library only.
"""
import argparse
import datetime as dt
import json
import re
import statistics
import sys
from pathlib import Path

LINE_RE = re.compile(r"^(\d\d):(\d\d):(\d\d)\s+GPU (HW active frequency|Power):\s+(\d+)\s*(MHz|mW)")
CELL_RE = re.compile(r"^p(\d+)-d(\d+)-r(\d+)\.json$")


def parse_log(path: Path, anchor: dt.datetime):
    """Local HH:MM:SS timestamps -> epoch seconds. The log has no date, so the
    first sample is placed on the anchor's day and every backwards jump of the
    clock adds a day."""
    freq, power = [], []
    day = anchor.replace(hour=0, minute=0, second=0, microsecond=0)
    last = None
    for line in path.read_text(errors="replace").splitlines():
        m = LINE_RE.match(line)
        if not m:
            continue
        h, mi, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
        t = day + dt.timedelta(hours=h, minutes=mi, seconds=s)
        if last is not None and t < last - dt.timedelta(hours=1):
            day += dt.timedelta(days=1)
            t += dt.timedelta(days=1)
        last = t
        value = int(m.group(5))
        (freq if m.group(4).startswith("HW") else power).append((t.timestamp(), value))
    return freq, power


BUSY_MW = 20_000  # the bench draws 60 to 80 W; model loading and idle sit far below


def window(samples, start, end):
    return [v for (t, v) in samples if start <= t <= end]


def busy_freq(freq, power, start, end):
    """Clock samples from seconds in which the GPU drew more than BUSY_MW, i.e.
    while the bench was computing rather than loading the model."""
    hot = {int(t) for (t, v) in power if start <= t <= end and v >= BUSY_MW}
    return [v for (t, v) in freq if int(t) in hot]


def fmt_range(values, unit_scale=1.0, decimals=0):
    if not values:
        return "-"
    med = statistics.median(values) * unit_scale
    lo, hi = min(values) * unit_scale, max(values) * unit_scale
    if round(lo, decimals) == round(hi, decimals):
        return f"{med:.{decimals}f}"
    return f"{med:.{decimals}f} ({lo:.{decimals}f} to {hi:.{decimals}f})"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path)
    ap.add_argument("records", nargs="+", type=Path)
    ap.add_argument("--annotate", action="store_true", help="write gpu_during_run into each result JSON")
    args = ap.parse_args()

    # Anchor the dateless log on the day the log file was last written (local time).
    anchor = dt.datetime.fromtimestamp(args.log.stat().st_mtime)
    freq, power = parse_log(args.log, anchor)
    if not freq:
        sys.exit(f"no GPU frequency samples in {args.log}")
    log_start, log_end = freq[0][0], freq[-1][0]

    rows = []
    for rec in args.records:
        rec = Path(rec)
        meta_path = rec / "run.json"
        label = json.loads(meta_path.read_text()).get("commit_short", rec.name) if meta_path.is_file() else rec.name
        for js in sorted(rec.iterdir()):
            m = CELL_RE.match(js.name)
            if not m:
                continue
            env = js.with_suffix(".env.txt")
            if not env.is_file():
                continue
            first = env.read_text().splitlines()[0].strip()
            start = dt.datetime.strptime(first, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).timestamp()
            end = js.stat().st_mtime
            if end < log_start or start > log_end:
                continue
            f = window(freq, start, end)
            p = window(power, start, end)
            b = busy_freq(freq, power, start, end)
            row = {
                "run": label, "cell": js.stem, "start": start, "secs": end - start,
                "freq": f, "power": p, "busy": b,
            }
            rows.append(row)
            if args.annotate and f:
                data = json.loads(js.read_text())
                data["gpu_during_run"] = {
                    "samples": len(f),
                    "busy_samples": len(b),
                    "busy_hw_freq_mhz_median": statistics.median(b) if b else None,
                    "busy_hw_freq_mhz_min": min(b) if b else None,
                    "hw_freq_mhz_max": max(f),
                    "power_w_median": statistics.median(p) / 1000 if p else None,
                    "source": str(args.log),
                }
                js.write_text(json.dumps(data, indent=2) + "\n")

    rows.sort(key=lambda r: r["start"])
    if not rows:
        sys.exit("no runs overlap the log's time span")
    print("| run | cell | start (local) | secs | busy GPU MHz median (min to max) | busy s | GPU W median | samples |")
    print("|-----|------|---------------|------|----------------------------------|--------|--------------|---------|")
    for r in rows:
        local = dt.datetime.fromtimestamp(r["start"]).strftime("%H:%M:%S")
        print(f"| `{r['run']}` | {r['cell']} | {local} | {r['secs']:.0f} | {fmt_range(r['busy'])} | {len(r['busy'])} "
              f"| {fmt_range(r['power'], 1 / 1000, 1)} | {len(r['freq'])} |")


if __name__ == "__main__":
    main()
