#!/usr/bin/env python3
"""Disk-tier check against a lily server started with a small GPU cache, e.g.

    lily --model ... --bind 127.0.0.1:8123 --cache-bytes 700M --max-sessions 2 \
         --disk-cache-dir /tmp/lily-sessions --disk-cache-bytes 20G

(q8 K/V caches, the default; with `--kv-cache bf16` the server refuses a
budget below 8 GiB.) Three conversations of ~1 000 prompt tokens each fill
the GPU tier, so the first one is spilled to disk; extending it afterwards
must report its whole first prompt as cached (read back from disk; the
checkpoint sits one token before the prompt's end). Whether the first
answer's tokens are reused too depends on the text retokenizing as the
model sampled it, which a small conversion's garbage often does not (its
answers start with a space the chat template strips), so that part is only
printed. Speculation statistics are printed from usage as well."""
import json, sys, time, urllib.request

URL = "http://127.0.0.1:8123"


def chat(messages, key, **kw):
    body = {"model": "Qwen3.8-Flash-Next", "messages": messages, "temperature": 0, "max_tokens": 40,
            "chat_template_kwargs": {"enable_thinking": False}, "prompt_cache_key": key}
    body.update(kw)
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    out["_secs"] = time.time() - t
    return out


def filler(n, seed):
    words = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu".split()
    return " ".join(f"{words[(i * 7 + seed) % len(words)]}{i}" for i in range(n))


convs = []
for c in range(3):
    msgs = [{"role": "user", "content": f"Here is list {c}:\n{filler(260, c)}\n\nWhat is the 5th item? Answer with just the item."}]
    r = chat(msgs, f"conv-{c}")
    u = r["usage"]
    print(f"conv {c}: prompt {u['prompt_tokens']} tokens, answer {r['choices'][0]['message']['content']!r}, "
          f"cached {u['prompt_tokens_details']['cached_tokens']}, drafts {u.get('completion_tokens_details')}, {r['_secs']:.2f}s")
    convs.append((msgs, r))

msgs, r = convs[0]
msgs2 = msgs + [{"role": "assistant", "content": r["choices"][0]["message"]["content"]},
                {"role": "user", "content": "And the 7th item? Just the item."}]
r2 = chat(msgs2, "conv-0")
u = r2["usage"]
first_prompt = r["usage"]["prompt_tokens"]
cached = u["prompt_tokens_details"]["cached_tokens"]
print(f"conv 0 extended: prompt {u['prompt_tokens']} tokens, cached {cached} "
      f"(expect at least the first prompt's {first_prompt - 1}, read back from disk; "
      f"{max(0, cached - (first_prompt - 1))} of the first answer's tokens reused too), "
      f"answer {r2['choices'][0]['message']['content']!r}, {r2['_secs']:.2f}s")
ok = cached >= first_prompt - 1
print("DISK RESUME", "OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
