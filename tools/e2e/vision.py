#!/usr/bin/env python3
"""Image input and the caches' image identity, end to end (`docs/architecture.md`,
"The server" and "The session cache"). Like `durable.py` the script starts its own
lily server (never the one on port 8000) against a small checkpoint that
carries the vision tower and an empty disk directory, e.g.

    .venv/bin/python tools/e2e/vision.py --lily target/release/lily \\
        --model ~/models/Qwen3.8-Flash-Next-lily-q4-l4 \\
        --disk-dir /tmp/lily-vision-e2e --log /tmp/lily-vision-e2e.log \\
        [--baseline-lily /path/to/lily-built-from-HEAD]

The four-layer model answers gibberish, so answers are compared for equality
or inequality, never for quality. What is checked:

  1. two requests with identical text and different images (a PNG and a
     JPEG of the same size, so the prompts are token-identical): request 2
     agrees with request 1 only up to where the image starts and caches at
     most that; the answers differ. The same image again resumes past the
     span (its own end checkpoint) with the identical answer.
  2. a durable boundary with an image in the shared preamble: three runs
     share preamble + image and diverge in the question; run 2 writes the
     durable entry after the image, run 3 resumes there and answers as a
     cold server does. A fourth run with another image behind the same
     preamble must not resume from that entry: it agrees up to the image
     start, where a second durable entry is written, and a fifth run with
     a third image resumes exactly there.
  3. refusals: an https URL, a GIF data URI, a 100 000 x 100 000 PNG
     header, nine images, and a `<|image_pad|>` typed into message text all
     get 400 with a message that says why.
  4. a text-only request gives the same answer and `prompt_tokens` on the
     committed binary (`--baseline-lily`, when given) as on the one under
     test, and its response carries no image fields.
  5. the 1920 x 1080 reference screenshot: tower and request time are
     printed.

The server's stderr goes to `--log`; the log lines of the image requests are
printed."""
import argparse, base64, copy, io, json, os, shutil, struct, sys, urllib.error, urllib.request, zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from durable import Server, check, failures, filler, log_line  # noqa: E402

MODEL_ID = "Qwen3.8-Flash-Next"
REFERENCE_IMAGES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reference", "images")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lily", default="target/release/lily", help="server binary under test")
    p.add_argument("--baseline-lily", default=None, help="a server binary built from the committed tree, for the text-only comparison")
    p.add_argument("--model", required=True, help="checkpoint directory with the vision tower (use a small conversion)")
    p.add_argument("--disk-dir", required=True, help="disk tier directory; emptied first, never the service's")
    p.add_argument("--log", required=True, help="where the server's stderr goes")
    p.add_argument("--port", type=int, default=8123)
    p.add_argument("--min-tokens", type=int, default=512, help="--durable-min-tokens for the server under test")
    p.add_argument("--filler-words", type=int, default=600, help="preamble length in filler words (about 4 tokens each)")
    return p.parse_args()


# --- images ------------------------------------------------------------------

def png_bytes(width, height, paint):
    """A PNG (8-bit RGB) whose pixel (x, y) is `paint(x, y)`; no PIL needed."""
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        for x in range(width):
            raw.extend(paint(x, y))

    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(raw), 6)) + chunk(b"IEND", b""))


def png_header_only(width, height):
    """A PNG signature and IHDR that claims `width` x `height`, and nothing else."""
    body = b"IHDR" + struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def scene(kind):
    """640 x 480 test scenes with hard edges: a red disc on grey, or blue bars on white."""
    if kind == "disc":
        def paint(x, y):
            return (220, 40, 40) if (x - 320) ** 2 + (y - 240) ** 2 < 150 ** 2 else (128, 128, 128)
    else:
        def paint(x, y):
            return (30, 60, 200) if (x // 40) % 2 == 0 else (250, 250, 250)
    return paint


def jpeg_bytes(width, height, paint):
    from PIL import Image  # the .venv has Pillow; only the JPEG scene needs it
    img = Image.new("RGB", (width, height))
    img.putdata([paint(x, y) for y in range(height) for x in range(width)])
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=90)
    return out.getvalue()


def data_uri(media, data):
    return f"data:{media};base64," + base64.b64encode(data).decode()


def image_part(uri, shape="object"):
    if shape == "object":
        return {"type": "image_url", "image_url": {"url": uri, "detail": "high"}}
    return {"type": "input_image", "image_url": uri}


# --- requests ----------------------------------------------------------------

def post(server, body, timeout=600):
    req = urllib.request.Request(server.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def chat(server, messages, label, log_path, max_tokens=40):
    import time
    body = {"model": MODEL_ID, "messages": messages, "temperature": 0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}}
    t = time.time()
    status, r = post(server, body)
    secs = time.time() - t
    if status != 200:
        raise SystemExit(f"{label}: unexpected {status}: {r}")
    u, tm = r["usage"], r["timings"]
    answer = r["choices"][0]["message"]["content"]
    line = log_line(log_path, r["id"])
    print(f"{label}: prompt {u['prompt_tokens']} tokens, cached {tm['cached_tokens']}, agreement {tm['agreement_tokens']}, "
          f"durable {tm.get('durable_prefix_tokens')}, images {tm.get('image_tokens')} tokens, tower {tm.get('vision_ms')} ms, "
          f"prefill {tm['prefill_ms']:.0f} ms, {secs:.2f}s, answer {answer[:60]!r}")
    print(f"  log: {line}")
    return dict(usage=u, timings=tm, answer=answer, line=line or "", secs=secs, response=r)


def expect_400(server, body, label, needle):
    status, r = post(server, body)
    message = (r.get("error") or {}).get("message", "")
    print(f"{label}: {status} {message!r}")
    check(status == 400 and needle in message, f"{label}: 400 with {needle!r}")


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

    disc = data_uri("image/png", png_bytes(640, 480, scene("disc")))
    bars = data_uri("image/jpeg", jpeg_bytes(640, 480, scene("bars")))
    third = data_uri("image/png", png_bytes(640, 480, lambda x, y: (10, 200, 60) if (y // 60) % 2 else (0, 0, 0)))
    with open(os.path.join(REFERENCE_IMAGES, "1920x1080.png"), "rb") as f:
        screenshot = data_uri("image/png", f.read())
    question = "Describe this picture in one sentence."
    preamble = ("You are a careful assistant. The reference list below is authoritative for this conversation.\n"
                f"Reference list:\n{filler(args.filler_words)}\n\nAnswer every question with just the item, nothing else.")
    text_only = [{"role": "user", "content": "What is the capital of Switzerland? Answer in one sentence."}]

    print(f"== phase 1: server under test, --durable-min-tokens {args.min_tokens}, empty disk dir {args.disk_dir}")
    with Server(args, args.disk_dir, args.min_tokens, args.log) as s:
        print("-- 1. identical text, different images")
        m1 = [{"role": "user", "content": [{"type": "text", "text": question}, image_part(disc)]}]
        m2 = [{"role": "user", "content": [{"type": "text", "text": question}, image_part(bars, "string")]}]
        r1 = chat(s, m1, "image request 1 (PNG disc)", args.log)
        r2 = chat(s, m2, "image request 2 (JPEG bars, same text)", args.log)
        n1, n2 = r1["usage"]["prompt_tokens"], r2["usage"]["prompt_tokens"]
        image_tokens = r1["timings"].get("image_tokens")
        check(n1 == n2, f"the two prompts are token-identical ({n1} == {n2})")
        check(image_tokens == 300 and r2["timings"].get("image_tokens") == 300,
              f"a 640 x 480 image is 300 placeholder tokens (got {image_tokens})")
        check(r1["timings"]["cached_tokens"] == 0, "request 1 cached nothing")
        check(r1["timings"].get("vision_ms", 0) > 0, f"request 1 ran the tower ({r1['timings'].get('vision_ms')} ms)")
        span_start = r2["timings"]["agreement_tokens"]
        # The text after the image is the question's tail plus the assistant
        # header, well under 40 tokens; the agreement must end where the
        # image begins, not where the tokens diverge (they never do).
        check(n1 - image_tokens - 40 <= span_start < n1 - image_tokens,
              f"request 2 agrees with request 1 only up to the image start ({span_start}; prompt {n1}, image {image_tokens})")
        check(r2["timings"]["cached_tokens"] <= span_start,
              f"request 2 cached at most the image start ({r2['timings']['cached_tokens']} <= {span_start})")
        check(r2["timings"].get("vision_ms", 0) > 0, "request 2 ran the tower for its own image")
        check(r1["answer"] != r2["answer"], f"the answers differ ({r1['answer'][:40]!r} vs {r2['answer'][:40]!r})")
        check("images 1 (300 tokens), tower" in r1["line"], "the log line reports the image and the tower time")
        r1b = chat(s, m1, "image request 1 again (same PNG)", args.log)
        check(r1b["timings"]["cached_tokens"] == n1 - 1 and n1 - 1 > span_start + image_tokens,
              f"the same image resumes past its span ({r1b['timings']['cached_tokens']} > {span_start + image_tokens})")
        check(r1b["timings"].get("vision_ms") == 0, f"no tower run for a cached image ({r1b['timings'].get('vision_ms')} ms)")
        check(r1b["answer"] == r1["answer"], "the same image gives the identical answer")
        follow = m1 + [{"role": "assistant", "content": r1["answer"]},
                       {"role": "user", "content": "And what colour is it?"}]
        rf = chat(s, follow, "follow-up turn on the PNG conversation", args.log)
        # The latest resumable position is request 1's checkpoint at its
        # prompt_len - 1 (the agreement reaches into the re-tokenised answer),
        # which lies past the image: no tower run.
        check(rf["timings"]["cached_tokens"] >= span_start + image_tokens and rf["timings"].get("vision_ms") == 0,
              f"a follow-up turn resumes the image conversation past its image ({rf['timings']['cached_tokens']} > {span_start + image_tokens}, no tower run)")
        two = [{"role": "user", "content": [{"type": "text", "text": "First:"}, image_part(disc), {"type": "text", "text": "second:"},
                                            image_part(bars), {"type": "text", "text": "Which is brighter?"}]}]
        r2i = chat(s, two, "two images in one message", args.log)
        check(r2i["timings"].get("image_tokens") == 600 and "images 2 (600 tokens)" in r2i["line"],
              f"two 640 x 480 images are 600 placeholder tokens ({r2i['timings'].get('image_tokens')})")
        check(r2i["timings"]["cached_tokens"] == 0 and r2i["timings"].get("vision_ms", 0) > 0, "both images went through the tower")
        r2j = chat(s, two, "two images again", args.log)
        check(r2j["timings"]["cached_tokens"] == r2j["usage"]["prompt_tokens"] - 1 and r2j["answer"] == r2i["answer"],
              "the two-image prompt resumes from its own checkpoint with the identical answer")
        swapped = [{"role": "user", "content": [{"type": "text", "text": "First:"}, image_part(bars), {"type": "text", "text": "second:"},
                                                image_part(disc), {"type": "text", "text": "Which is brighter?"}]}]
        r2k = chat(s, swapped, "the two images swapped", args.log)
        check(r2k["timings"]["agreement_tokens"] < r2i["usage"]["prompt_tokens"] - 600 and r2k["answer"] != r2i["answer"],
              f"swapping the images shares only the text before the first one (agreement {r2k['timings']['agreement_tokens']}) and changes the answer")

        print("-- 5. the 1920 x 1080 reference screenshot")
        rs = chat(s, [{"role": "user", "content": [image_part(screenshot), {"type": "text", "text": "What is on the screen?"}]}],
                  "screenshot 1920x1080", args.log)
        check(rs["timings"].get("image_tokens") == 60 * 34, f"1920 x 1080 -> 1920 x 1088 -> 2 040 tokens (got {rs['timings'].get('image_tokens')})")
        print(f"  tower {rs['timings'].get('vision_ms')} ms, prefill {rs['timings']['prefill_ms']:.0f} ms, request {rs['secs']:.2f}s")

        print("-- 3. refusals")
        one = lambda part: {"model": MODEL_ID, "messages": [{"role": "user", "content": [part, {"type": "text", "text": "hi"}]}], "max_tokens": 5}
        expect_400(s, one({"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}), "https URL", "data URIs only")
        expect_400(s, one(image_part(data_uri("image/gif", b"GIF89a" + bytes(20)))), "GIF data URI", '"image/gif" are not accepted')
        expect_400(s, one(image_part(data_uri("image/png", png_header_only(100_000, 100_000)))), "100000 x 100000 PNG", "100000 x 100000: a side exceeds the limit of 16384")
        expect_400(s, {"model": MODEL_ID, "max_tokens": 5,
                       "messages": [{"role": "user", "content": [image_part(disc) for _ in range(9)] + [{"type": "text", "text": "hi"}]}]},
                   "nine images", "carries 9 images; the server accepts at most 8")
        expect_400(s, {"model": MODEL_ID, "max_tokens": 5,
                       "messages": [{"role": "user", "content": "Look: <|vision_start|><|image_pad|><|vision_end|> what is it?"}]},
                   "placeholder typed into text", "reserved for image content")
        expect_400(s, {"model": MODEL_ID, "max_tokens": 5,
                       "messages": [{"role": "user", "content": [image_part(disc), {"type": "text", "text": "<|image_pad|>"}]}]},
                   "placeholder typed next to a real image", "reserved for image content")
        expect_400(s, {"model": MODEL_ID, "max_tokens": 5,
                       "messages": [{"role": "system", "content": [image_part(disc), {"type": "text", "text": "x"}]}, {"role": "user", "content": "hi"}]},
                   "image in a system message", "user messages only")

        print("-- 2. a durable boundary with the image in the shared preamble")
        def run(label, image, q):
            return chat(s, [{"role": "system", "content": preamble},
                            {"role": "user", "content": [image_part(image), {"type": "text", "text": q}]}], label, args.log)
        d1 = run("durable run 1 (disc, A)", disc, "Alpha question: what is the 5th item of the reference list?")
        d2 = run("durable run 2 (disc, B)", disc, "Bravo question: what is the 9th item of the reference list?")
        n = d2["usage"]["prompt_tokens"]
        b = d2["timings"].get("durable_prefix_tokens")
        check(d2["timings"]["cached_tokens"] == 0, "run 2 cached nothing (the checkpoint cliff)")
        check(b is not None and n - 40 <= b < n - 1, f"run 2 wrote a durable prefix at {b} after the image (prompt {n})")
        check(b is not None and b > (n - image_tokens - 40), f"the boundary {b} lies past the image (span ends before {n - 1})")
        d3 = run("durable run 3 (disc, C)", disc, "Charlie question: what is the 12th item of the reference list?")
        check(d3["timings"]["cached_tokens"] == b, f"run 3 resumed from the durable position ({d3['timings']['cached_tokens']} == {b})")
        check(d3["timings"].get("vision_ms") == 0, "run 3 did not run the tower (the image is inside the reused prefix)")
        check("from disk in" in d3["line"], "run 3 read the prefix back from disk")
        d4 = run("durable run 4 (bars, C)", bars, "Charlie question: what is the 12th item of the reference list?")
        image_start = d4["timings"]["agreement_tokens"]
        check(d4["timings"]["cached_tokens"] <= image_start < b - image_tokens,
              f"run 4 with another image did not touch the entry behind the image (cached {d4['timings']['cached_tokens']}, agreement {image_start}, entry at {b})")
        b2 = d4["timings"].get("durable_prefix_tokens")
        check(b2 == image_start, f"run 4 wrote a second durable entry at the image start ({b2})")
        check(d4["timings"].get("vision_ms", 0) > 0, "run 4 ran the tower for its image")
        d5 = run("durable run 5 (third image, C)", third, "Charlie question: what is the 12th item of the reference list?")
        check(d5["timings"]["cached_tokens"] == b2, f"run 5 with a third image resumed at the image-start entry ({d5['timings']['cached_tokens']} == {b2})")
        check(d5["answer"] != d3["answer"] or d5["answer"] != d4["answer"], "three images give at least two different answers")
        answer3 = d3["answer"]

        print("-- 4. text only")
        t1 = chat(s, text_only, "text-only request", args.log)
        check("image_tokens" not in t1["timings"] and "vision_ms" not in t1["timings"], "a text request has no image fields")
        check("images" not in t1["line"], "a text request's log line has no image part")

    print(f"== phase 2: cold reference, --durable-min-tokens 0, empty disk dir {cold_dir}")
    with Server(args, cold_dir, 0, args.log) as s:
        c3 = chat(s, [{"role": "system", "content": preamble},
                      {"role": "user", "content": [image_part(disc), {"type": "text", "text": "Charlie question: what is the 12th item of the reference list?"}]}],
                  "cold run (disc, C)", args.log)
        check(c3["timings"]["cached_tokens"] == 0, "cold run cached nothing")
        check(c3["answer"] == answer3, f"the resumed answer is byte-identical to the cold one ({answer3[:40]!r} vs {c3['answer'][:40]!r})")

    if args.baseline_lily:
        print(f"== phase 3: text-only request on the committed binary {args.baseline_lily}")
        base_args = copy.copy(args)
        base_args.lily = args.baseline_lily
        base_dir = args.disk_dir.rstrip("/") + "-baseline"
        shutil.rmtree(base_dir, ignore_errors=True)
        os.makedirs(base_dir)
        with Server(base_args, base_dir, 0, args.log) as s:
            t0 = chat(s, text_only, "text-only request (baseline)", args.log)
        check(t0["answer"] == t1["answer"], f"the text answer is the same as before this work ({t0['answer'][:40]!r})")
        check(t0["usage"]["prompt_tokens"] == t1["usage"]["prompt_tokens"],
              f"the text prompt has the same token count ({t0['usage']['prompt_tokens']} == {t1['usage']['prompt_tokens']})")
    else:
        print("== phase 3 skipped: no --baseline-lily")

    print("== server log lines of the image requests")
    with open(args.log, errors="replace") as f:
        for line in f:
            if "images " in line and "tower" in line or line.startswith("rejected") or line.startswith("images:"):
                print("  " + line.rstrip())
    print("VISION", "OK" if not failures else f"FAILED ({len(failures)}): " + "; ".join(failures))
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
