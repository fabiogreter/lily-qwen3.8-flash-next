#!/usr/bin/env python3
"""Prefill, decode and prompt-cache behaviour over HTTP, for lily and for a
llama.cpp server (Unsloth Studio's), with the same prompts and each engine's
own timings.

    .venv/bin/python tools/bench/http_bench.py matrix --engine llama \\
        --url http://127.0.0.1:59765 --label studio-iq4xs --spec none,default \\
        --out docs/bench/2026-09-17-vs-unsloth/llama.jsonl
    .venv/bin/python tools/bench/http_bench.py matrix --engine lily \\
        --url http://127.0.0.1:8000 --label lily-q4-drafts2 \\
        --out docs/bench/2026-09-17-vs-unsloth/lily-d2.jsonl
    .venv/bin/python tools/bench/http_bench.py cache --engine ... --url ... --out ...
    .venv/bin/python tools/bench/http_bench.py report a.jsonl b.jsonl ...

`matrix` runs prompts of the requested lengths (default 1K, 4K, 16K, 32K, 64K)
with a fixed number of greedy tokens per run, repeats interleaved per repeat
(repeat 1 of every cell, then repeat 2, ...). Every run gets a prompt nobody
has sent before, cut from a different offset of the corpus, so a prompt
cache never shortens a prefill measurement. Prompt material is this
repository's own documentation and source, so both engines see prose and
code, the way a coding agent's context looks, tokenized with the model's
tokenizer to the target length.

For a llama.cpp server, `--spec none,default` interleaves plain decoding with
the server's speculative configuration inside one server: a request that
carries any `speculative.types` value runs whatever the server was started
with (Unsloth Studio's `--spec-default` is the `ngram-mod` n-gram drafter; the
value itself is not honoured, `draft-mtp` and `none` behave the same as
`ngram-mod`, measured 2026-09-17 on build b11007), a request without the key
runs plain. Studio's MTP drafter did not load in that session (a GGML
assertion in the fork's qwen4exp MTP graph at startup; Studio fell back to no
drafter), so what its users get as "MTP" is the n-gram drafter or nothing.
lily's draft count is a server flag (`--mtp-drafts`), so the lily matrix is
run once per launch with a `--label` saying which.

The generation task deliberately ignores the prompt ("explain how a
bicycle's gears work"): an n-gram drafter copying from the context reaches
near-total acceptance on a "continue the document" task and inflates decode
figures fourfold, which says nothing about generating new text. With a
novel-text task the prompt still costs its full prefill and decode measures
generation.

`cache` measures what a prompt cache gives back: the same 16K prefix with
different questions (how much of the prefix survives), a conversation that
grows turn by turn, and six long conversations interleaved (more than
llama.cpp's slots). It reports the tokens each engine recomputed and the
prefill time.

Only the engines' own counters are compared: llama.cpp's `timings` object
(`prompt_n`, `prompt_ms`, `predicted_n`, `predicted_ms`, `cache_n`,
`draft_n`, `draft_n_accepted`) and lily's `timings` object (`prefill_tokens`,
`prefill_ms`, `generated_tokens`, `decode_ms`, `cached_tokens`,
`drafted_tokens`, `accepted_tokens`). Wall time per request is recorded too.
Every record carries swap usage and the time, because repeated large model
loads with swap in use distort everything (docs/performance.md)."""
import argparse, glob, json, os, statistics, subprocess, sys, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_TOKENIZER = os.path.expanduser("~/projects/personal/local-llms/models/Qwen3.8-Flash-Next-lily-q4/tokenizer.json")
MODEL_IDS = {"lily": "Qwen3.8-Flash-Next", "llama": "unsloth/Qwen3.8-Flash-Next-GGUF"}
INSTRUCTION = ("\n\nIgnore the text above completely. Instead, explain in your own words how the gears of a "
               "bicycle work, in about 400 words of flowing prose, without lists.")


# --- prompt material ---------------------------------------------------------

class Corpus:
    """The repository's docs and source as one token sequence."""

    def __init__(self, tokenizer_path):
        import tokenizers
        self.tok = tokenizers.Tokenizer.from_file(tokenizer_path)
        files = (sorted(glob.glob(os.path.join(REPO, "docs", "*.md"))) + [os.path.join(REPO, "README.md")]
                 + sorted(glob.glob(os.path.join(REPO, "src", "**", "*.rs"), recursive=True))
                 + sorted(glob.glob(os.path.join(REPO, "src", "kernels", "metal", "*.metal"))))
        text = "\n\n".join(open(f, errors="replace").read() for f in files)
        self.ids = self.tok.encode(text).ids

    def slice(self, n_tokens, offset):
        """`n_tokens` tokens of corpus text starting at `offset` (wrapping)."""
        ids = [self.ids[(offset + i) % len(self.ids)] for i in range(n_tokens)]
        return self.tok.decode(ids)

    def count(self, text):
        return len(self.tok.encode(text).ids)


def prompt_of(corpus, target_tokens, offset, overhead=40):
    """Corpus text whose chat-templated prompt lands near `target_tokens`."""
    return corpus.slice(max(16, target_tokens - overhead), offset) + INSTRUCTION


# --- requests ----------------------------------------------------------------

def swap_usage():
    try:
        return subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return None


def post(url, body, timeout=1800):
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r), time.time() - t
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.load(e), time.time() - t
        except Exception:
            return e.code, {"error": str(e)}, time.time() - t


def chat(engine, url, messages, max_tokens, spec="none"):
    body = {"model": MODEL_IDS[engine], "messages": messages, "temperature": 0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}}
    if engine == "llama" and spec != "none":
        # Any value switches the server's own speculative configuration on
        # for this request (see the module docstring); name what it is.
        body["speculative.types"] = "ngram-mod"
    status, r, secs = post(url, body)
    if status != 200:
        raise RuntimeError(f"{engine} {status}: {json.dumps(r)[:400]}")
    return normalise(engine, r, secs)


def normalise(engine, r, secs):
    """One vocabulary for both engines' counters."""
    t = r.get("timings") or {}
    u = r.get("usage") or {}
    if engine == "llama":
        out = dict(prompt_tokens=u.get("prompt_tokens"), cached_tokens=t.get("cache_n"), prefill_tokens=t.get("prompt_n"),
                   prefill_ms=t.get("prompt_ms"), generated_tokens=t.get("predicted_n"), decode_ms=t.get("predicted_ms"),
                   drafted_tokens=t.get("draft_n"), accepted_tokens=t.get("draft_n_accepted"))
    else:
        out = dict(prompt_tokens=t.get("prompt_tokens"), cached_tokens=t.get("cached_tokens"), prefill_tokens=t.get("prefill_tokens"),
                   prefill_ms=t.get("prefill_ms"), generated_tokens=t.get("generated_tokens"), decode_ms=t.get("decode_ms"),
                   drafted_tokens=t.get("drafted_tokens"), accepted_tokens=t.get("accepted_tokens"))
    out["wall_s"] = secs
    out["prefill_per_second"] = rate(out["prefill_tokens"], out["prefill_ms"])
    out["decode_per_second"] = rate(out["generated_tokens"], out["decode_ms"])
    out["content"] = (r.get("choices") or [{}])[0].get("message", {}).get("content")
    out["finish_reason"] = (r.get("choices") or [{}])[0].get("finish_reason")
    return out


def rate(n, ms):
    return None if not n or not ms else n / (ms / 1e3)


def record(out_path, rec):
    rec["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    rec["swap"] = swap_usage()
    with open(out_path, "a") as f:
        f.write(json.dumps(rec) + "\n")


# --- matrix ------------------------------------------------------------------

def cmd_matrix(args):
    corpus = Corpus(args.tokenizer)
    lengths = [int(x) for x in args.prompt_tokens.split(",")]
    specs = args.spec.split(",") if args.engine == "llama" else ["server"]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print(f"{args.label}: {args.engine} at {args.url}; corpus {len(corpus.ids)} tokens; lengths {lengths}; specs {specs}; "
          f"{args.repeats} repeats, {args.decode_tokens} greedy tokens per run; swap {swap_usage()}")
    # Warm-up at the smallest shape, not recorded: the GPU is ramping after a load.
    chat(args.engine, args.url, [{"role": "user", "content": prompt_of(corpus, lengths[0], 7)}], 16)
    run = 0
    for rep in range(args.repeats):
        for n in lengths:
            for spec in specs:
                run += 1
                offset = (run * 104729 + rep * 7919) % len(corpus.ids)  # a fresh region every run
                messages = [{"role": "user", "content": prompt_of(corpus, n, offset)}]
                res = chat(args.engine, args.url, messages, args.decode_tokens, spec)
                rec = dict(kind="matrix", label=args.label, engine=args.engine, spec=spec, target_tokens=n, repeat=rep + 1, **res)
                record(args.out, rec)
                acc = "" if res["drafted_tokens"] is None else f", drafts {res['accepted_tokens']}/{res['drafted_tokens']}"
                print(f"  rep {rep + 1} {n:>6} {spec:>6}: prompt {res['prompt_tokens']} (cached {res['cached_tokens']}), "
                      f"prefill {res['prefill_per_second'] or 0:7.1f} tok/s ({(res['prefill_ms'] or 0) / 1e3:5.1f} s), "
                      f"decode {res['decode_per_second'] or 0:6.1f} tok/s over {res['generated_tokens']}{acc}, wall {res['wall_s']:.1f} s")
                time.sleep(args.cooldown)


# --- cache -------------------------------------------------------------------

def cmd_cache(args):
    corpus = Corpus(args.tokenizer)
    E, U, out = args.engine, args.url, args.out
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    print(f"{args.label}: cache behaviour of {E} at {U}; swap {swap_usage()}")

    def ask(messages, tag, max_tokens=24, spec="none"):
        res = chat(E, U, messages, max_tokens, spec)
        record(out, dict(kind="cache", label=args.label, engine=E, test=tag, **res))
        print(f"  {tag:<34} prompt {res['prompt_tokens']:>6}, cached {res['cached_tokens'] or 0:>6}, recomputed {res['prefill_tokens']:>6}, "
              f"prefill {(res['prefill_ms'] or 0) / 1e3:6.2f} s, wall {res['wall_s']:5.2f} s")
        return res

    # 1. One 16K prefix, different questions: how much of the prefix comes back.
    prefix = corpus.slice(16384, 500_003)
    questions = ["\n\nIn one sentence, what is the text above about?", "\n\nName one identifier that appears above.",
                 "\n\nIn one sentence, what is the text above about?", "\n\nWhat is the last word above?"]
    for i, q in enumerate(questions):
        ask([{"role": "user", "content": prefix + q}], f"same-prefix q{i + 1}")

    # 2. A conversation that grows: every turn appends the answer and a new question.
    messages = [{"role": "user", "content": corpus.slice(8192, 900_007) + "\n\nSummarise the text above in two sentences."}]
    for turn in range(6):
        res = ask(messages, f"growing turn {turn + 1}", max_tokens=64)
        messages = messages + [{"role": "assistant", "content": res["content"] or ""},
                               {"role": "user", "content": corpus.slice(512, 1_200_011 + turn * 4099) + "\n\nAnd this part, in one sentence?"}]

    # 3. Six long conversations interleaved, two rounds: more conversations than
    #    llama.cpp's four slots, fewer than lily's session cache holds.
    convs = [[{"role": "user", "content": corpus.slice(12288, 1_500_017 + c * 300_007) + "\n\nWhat is this about? One sentence."}] for c in range(6)]
    for rnd in range(2):
        for c, messages in enumerate(convs):
            res = ask(messages, f"interleaved conv {c + 1} round {rnd + 1}", max_tokens=32)
            convs[c] = messages + [{"role": "assistant", "content": res["content"] or ""},
                                   {"role": "user", "content": "Name one more detail from it. One sentence."}]


# --- report ------------------------------------------------------------------

def load(paths):
    recs = []
    for p in paths:
        with open(p) as f:
            recs += [json.loads(line) for line in f if line.strip()]
    return recs


def med(xs):
    xs = [x for x in xs if x is not None]
    return (statistics.median(xs), min(xs), max(xs)) if xs else None


def fmt(m, digits=0):
    if m is None:
        return "—"
    a, lo, hi = m
    return f"{a:.{digits}f} ({lo:.{digits}f}–{hi:.{digits}f})"


def cmd_report(args):
    recs = load(args.files)
    matrix = [r for r in recs if r["kind"] == "matrix"]
    if matrix:
        labels = sorted({(r["label"], r["spec"]) for r in matrix})
        lengths = sorted({r["target_tokens"] for r in matrix})
        cold = [r for r in matrix if not r["cached_tokens"]]  # a cache hit is not a prefill measurement
        skipped = len(matrix) - len(cold)
        print(f"Prefill, tok/s (median, min–max over repeats; every run a fresh prompt"
              f"{f'; {skipped} runs with a cache hit left out' if skipped else ''})\n")
        print("| context | " + " | ".join(f"{l} {s}" for l, s in labels) + " |")
        print("|---|" + "---|" * len(labels))
        for n in lengths:
            cells = [fmt(med([r["prefill_per_second"] for r in cold if (r["label"], r["spec"]) == ls and r["target_tokens"] == n])) for ls in labels]
            print(f"| {n} | " + " | ".join(cells) + " |")
        print("\nDecode, tok/s\n")
        print("| context | " + " | ".join(f"{l} {s}" for l, s in labels) + " |")
        print("|---|" + "---|" * len(labels))
        for n in lengths:
            cells = [fmt(med([r["decode_per_second"] for r in matrix if (r["label"], r["spec"]) == ls and r["target_tokens"] == n]), 1) for ls in labels]
            print(f"| {n} | " + " | ".join(cells) + " |")
        print("\nDraft acceptance (accepted / drafted over all repeats)\n")
        for ls in labels:
            rs = [r for r in matrix if (r["label"], r["spec"]) == ls and r["drafted_tokens"]]
            if rs:
                a, d = sum(r["accepted_tokens"] for r in rs), sum(r["drafted_tokens"] for r in rs)
                print(f"- {ls[0]} {ls[1]}: {a}/{d} = {a / d:.2f}")
    cache = [r for r in recs if r["kind"] == "cache"]
    if cache:
        labels = sorted({r["label"] for r in cache})
        tests = []
        for r in cache:
            if r["test"] not in tests:
                tests.append(r["test"])
        print("\nPrompt cache: tokens recomputed / prompt tokens, prefill seconds\n")
        print("| request | " + " | ".join(labels) + " |")
        print("|---|" + "---|" * len(labels))
        for t in tests:
            cells = []
            for l in labels:
                rs = [r for r in cache if r["label"] == l and r["test"] == t]
                cells.append("—" if not rs else f"{rs[0]['prefill_tokens']} / {rs[0]['prompt_tokens']}, {(rs[0]['prefill_ms'] or 0) / 1e3:.2f} s")
            print(f"| {t} | " + " | ".join(cells) + " |")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("matrix", cmd_matrix), ("cache", cmd_cache)):
        s = sub.add_parser(name)
        s.add_argument("--engine", choices=list(MODEL_IDS), required=True)
        s.add_argument("--url", required=True)
        s.add_argument("--label", required=True)
        s.add_argument("--out", required=True)
        s.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
        s.set_defaults(fn=fn)
    m = sub.choices["matrix"]
    m.add_argument("--prompt-tokens", default="1024,4096,16384,32768,65536")
    m.add_argument("--decode-tokens", type=int, default=256)
    m.add_argument("--repeats", type=int, default=3)
    m.add_argument("--cooldown", type=float, default=10, help="seconds between runs")
    m.add_argument("--spec", default="none", help="llama only: comma-separated `none` (plain) and `default` (the server's speculative configuration)")
    r = sub.add_parser("report")
    r.add_argument("files", nargs="+")
    r.set_defaults(fn=cmd_report)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
