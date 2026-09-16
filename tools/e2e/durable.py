#!/usr/bin/env python3
"""Durable-prefix check: the script starts its own lily server (never the one
on port 8000) against a small checkpoint and an empty disk directory, e.g.

    python3 tools/e2e/durable.py --lily target/release/lily \
        --model ~/models/Qwen3.8-Flash-Next-lily-q4-l4 \
        --disk-dir /tmp/lily-durable-e2e --log /tmp/lily-durable-e2e.log

Five prompts share a preamble of about 2 400 tokens (a system message) and
then ask different questions, the way two agent runs share their preamble
and diverge at the user's message. Run 1 finds nothing cached. Run 2 agrees with
run 1 for the whole preamble but cannot resume there (the only checkpoint sits
at the end of run 1's prompt), so the server writes a durable prefix entry at
the agreement. Run 3 resumes from it: `cached_tokens` equals the durable
position, and the write did not add a resident session. The server is then
restarted: runs 4 and 5 (two more fresh questions, since an identical prompt
would resume from its own end checkpoint instead) both resume from the entry,
which proves it survived the restart and was not consumed by a hit. Finally a
server with the feature off and an empty disk directory answers run 3's
prompt cold; the answer must be byte-identical (greedy) to the resumed one.

The server's stderr goes to `--log`; the `durable prefix ... written` and
`divergence at ...` lines are printed at the end."""
import argparse, json, os, re, shutil, signal, subprocess, sys, time, urllib.error, urllib.request

MODEL_ID = "Qwen3.8-Flash-Next"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lily", default="target/release/lily", help="server binary")
    p.add_argument("--model", required=True, help="checkpoint directory (use a small conversion)")
    p.add_argument("--disk-dir", required=True, help="disk tier directory; emptied first, never the service's")
    p.add_argument("--log", required=True, help="where the server's stderr goes")
    p.add_argument("--port", type=int, default=8123)
    p.add_argument("--min-tokens", type=int, default=512, help="--durable-min-tokens for the server under test")
    p.add_argument("--filler-words", type=int, default=600, help="preamble length in filler words (about 4 tokens each)")
    return p.parse_args()


class Server:
    """One lily process on the test port, stderr appended to the log file."""

    def __init__(self, args, disk_dir, min_tokens, log):
        self.args, self.disk_dir, self.min_tokens, self.log_path = args, disk_dir, min_tokens, log
        self.url = f"http://127.0.0.1:{args.port}"
        self.proc = None

    def __enter__(self):
        cmd = [self.args.lily, "--model", self.args.model, "--bind", f"127.0.0.1:{self.args.port}",
               "--max-seq", "16384", "--cache-bytes", "2G", "--max-sessions", "4",
               "--disk-cache-dir", self.disk_dir, "--disk-cache-bytes", "20G",
               "--durable-min-tokens", str(self.min_tokens), "--ngram-preload", "false"]
        for attempt in range(3):
            self.log = open(self.log_path, "ab")
            self.log.write(f"\n=== {time.strftime('%H:%M:%S')} start: {' '.join(cmd)}\n".encode())
            self.log.flush()
            self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=self.log)
            deadline = time.time() + 600
            while time.time() < deadline:
                if self.proc.poll() is not None:
                    break
                try:
                    with urllib.request.urlopen(self.url + "/health", timeout=2) as r:
                        if json.load(r).get("state") == "ready":
                            return self
                except (urllib.error.URLError, ConnectionError, json.JSONDecodeError):
                    pass
                time.sleep(0.5)
            if self.proc.poll() is not None:
                # The previous instance may still own the port for a moment.
                print(f"server exited with {self.proc.returncode} during startup (attempt {attempt + 1}), retrying")
                time.sleep(2)
                continue
            raise SystemExit("server did not become ready in time")
        raise SystemExit("server failed to start; see the log")

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.log.close()
        # Wait for the port to be released before another instance binds it.
        for _ in range(40):
            try:
                urllib.request.urlopen(self.url + "/health", timeout=1)
                time.sleep(0.5)
            except (urllib.error.URLError, ConnectionError):
                break

    def chat(self, messages, key=None):
        body = {"model": MODEL_ID, "messages": messages, "temperature": 0, "max_tokens": 40,
                "chat_template_kwargs": {"enable_thinking": False}}
        if key:
            body["prompt_cache_key"] = key
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t = time.time()
        with urllib.request.urlopen(req, timeout=600) as r:
            out = json.load(r)
        out["_secs"] = time.time() - t
        return out

    def timings(self):
        with urllib.request.urlopen(self.url + "/v1/timings", timeout=10) as r:
            return json.load(r)["data"]


def filler(n, seed=3):
    words = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu".split()
    return " ".join(f"{words[(i * 7 + seed) % len(words)]}{i}" for i in range(n))


def log_line(log_path, response_id):
    """The server log line for a response id (it starts with the id)."""
    with open(log_path, errors="replace") as f:
        for line in f:
            if line.startswith(response_id + ":"):
                return line.rstrip()
    return None


def field(line, pattern):
    m = re.search(pattern, line or "")
    return int(m.group(1)) if m else None


failures = []


def check(cond, what):
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        failures.append(what)


def run(server, label, preamble, question, log_path):
    msgs = [{"role": "system", "content": preamble}, {"role": "user", "content": question}]
    r = server.chat(msgs)
    u, t = r["usage"], r["timings"]
    answer = r["choices"][0]["message"]["content"]
    line = log_line(log_path, r["id"])
    print(f"{label}: prompt {u['prompt_tokens']} tokens, cached {t['cached_tokens']}, agreement {t['agreement_tokens']}, "
          f"durable {t.get('durable_prefix_tokens')}, prefill {t['prefill_ms']:.0f} ms, {r['_secs']:.2f}s, answer {answer!r}")
    print(f"  log: {line}")
    return dict(usage=u, timings=t, answer=answer, line=line, sessions=field(line, r"sessions=(\d+)"),
                disk=field(line, r"disk (\d+)"), from_disk_secs=(re.search(r"from disk in ([0-9.]+)s", line or "") or [None, None])[1])


def main():
    args = parse_args()
    if os.path.abspath(args.disk_dir).startswith(os.path.expanduser("~/Library/Caches/lily")):
        raise SystemExit("refusing to use the service's disk cache directory")
    shutil.rmtree(args.disk_dir, ignore_errors=True)
    os.makedirs(args.disk_dir)
    cold_dir = args.disk_dir.rstrip("/") + "-cold"
    shutil.rmtree(cold_dir, ignore_errors=True)
    os.makedirs(cold_dir)
    open(args.log, "w").close()

    preamble = ("You are a careful assistant. The reference list below is authoritative for this conversation.\n"
                f"Reference list:\n{filler(args.filler_words)}\n\nAnswer every question with just the item, nothing else.")
    questions = {
        "A": "Alpha question: what is the 5th item of the reference list?",
        "B": "Bravo question: what is the 9th item of the reference list?",
        "C": "Charlie question: what is the 12th item of the reference list?",
        "D": "Delta question: what is the 3rd item of the reference list?",
        "E": "Echo question: what is the 7th item of the reference list?",
    }

    print(f"== phase 1: fresh server, --durable-min-tokens {args.min_tokens}, empty disk dir {args.disk_dir}")
    with Server(args, args.disk_dir, args.min_tokens, args.log) as s:
        r1 = run(s, "run 1 (A)", preamble, questions["A"], args.log)
        check(r1["timings"]["cached_tokens"] == 0, "run 1 cached nothing")
        check("durable_prefix_tokens" not in r1["timings"], "run 1 wrote no durable prefix")
        check(r1["timings"]["agreement_tokens"] == 0, "run 1 agreed with nothing")

        r2 = run(s, "run 2 (B)", preamble, questions["B"], args.log)
        n = r2["usage"]["prompt_tokens"]
        b = r2["timings"].get("durable_prefix_tokens")
        check(r2["timings"]["cached_tokens"] == 0, "run 2 cached nothing (the checkpoint cliff)")
        check(b is not None and args.min_tokens <= b < n - 1, f"run 2 wrote a durable prefix at {b} (prompt {n})")
        check(b is not None and n - 40 <= b, f"the durable position {b} is close to the preamble length (prompt {n})")
        check(r2["timings"]["agreement_tokens"] == b, f"run 2 agreement {r2['timings']['agreement_tokens']} == durable position")
        check("durable prefix" in (r2["line"] or ""), "run 2 log line reports the durable write")
        check(r2["sessions"] == r1["sessions"] + 1, f"the write added no resident session (sessions {r1['sessions']} -> {r2['sessions']})")
        check(r2["disk"] == 1, f"disk tier holds exactly the durable entry after run 2 (disk {r2['disk']})")

        r3 = run(s, "run 3 (C)", preamble, questions["C"], args.log)
        check(r3["timings"]["cached_tokens"] == b, f"run 3 resumed from the durable position ({r3['timings']['cached_tokens']} == {b})")
        check(r3["from_disk_secs"] is not None, f"run 3 read the prefix back from disk in {r3['from_disk_secs']}s")
        check("durable_prefix_tokens" not in r3["timings"], "run 3 wrote nothing (agreement == reused)")
        check(r3["disk"] == 1, f"the hit did not consume the entry (disk {r3['disk']})")
        check(r3["sessions"] == r2["sessions"] + 1, f"run 3 is one more resident session ({r2['sessions']} -> {r3['sessions']})")
        newest = s.timings()[0]["timings"]
        check(newest["cached_tokens"] == b and newest["agreement_tokens"] == b, "GET /v1/timings carries the same numbers")
        answer3 = r3["answer"]

    print("== phase 2: restart on the same disk dir (resident sessions were spilled at shutdown)")
    with Server(args, args.disk_dir, args.min_tokens, args.log) as s:
        with open(args.log, errors="replace") as f:
            startup = [l for l in f if "disk tier at" in l][-1]
        print(f"  startup: {startup.strip()}")
        check(re.search(r"\(\d+ entries, 1 durable", startup) is not None, "the durable entry is indexed after the restart")
        r4 = run(s, "run 4 (D)", preamble, questions["D"], args.log)
        r5 = run(s, "run 5 (E)", preamble, questions["E"], args.log)
        check(r4["timings"]["cached_tokens"] == b, f"run 4 resumed from the durable entry after the restart ({r4['timings']['cached_tokens']} == {b})")
        check(r5["timings"]["cached_tokens"] == b, f"run 5 too, so the hit did not consume it ({r5['timings']['cached_tokens']} == {b})")
        check(r4["timings"]["cached_tokens"] == r5["timings"]["cached_tokens"], "runs 4 and 5 report the same cached count")
        check("durable_prefix_tokens" not in r4["timings"] and "durable_prefix_tokens" not in r5["timings"], "no second durable entry was written")
        r3b = run(s, "run 3 again (C, identical prompt)", preamble, questions["C"], args.log)
        check(r3b["timings"]["cached_tokens"] >= b, "an identical prompt resumes at least as far (its own spilled end checkpoint)")
        check(r3b["answer"] == answer3, "the identical prompt gives the same answer")

    print(f"== phase 3: cold reference, --durable-min-tokens 0, empty disk dir {cold_dir}")
    with Server(args, cold_dir, 0, args.log) as s:
        rc = run(s, "cold run (C)", preamble, questions["C"], args.log)
        check(rc["timings"]["cached_tokens"] == 0, "cold run cached nothing")
        check("durable_prefix_tokens" not in rc["timings"], "cold run wrote nothing with the feature off")
        check(rc["answer"] == answer3, f"resumed answer is byte-identical to the cold one ({answer3!r} vs {rc['answer']!r})")

    print("== server log lines of interest")
    with open(args.log, errors="replace") as f:
        for line in f:
            if "durable prefix" in line or line.startswith("divergence at"):
                print("  " + line.rstrip())
    print("DURABLE PREFIX", "OK" if not failures else f"FAILED ({len(failures)}): " + "; ".join(failures))
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
