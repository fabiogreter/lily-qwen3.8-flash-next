#!/usr/bin/env python3
"""Long-context checks against a running lily server: an ~8K-token prompt whose
decode crosses the 8192-token capacity step (state growth mid-decode), then a
cached extension of the same conversation, then a ~20K prompt."""
import json, sys, time, urllib.request

import os
URL = os.environ.get("LILY_URL", "http://127.0.0.1:8123")


def chat(messages, **kw):
    body = {"model": "Qwen3.8-Flash-Next", "messages": messages, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    body.update(kw)
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    out["_secs"] = time.time() - t
    return out


def filler(n_words):
    words = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau "
             "upsilon phi chi psi omega").split()
    return " ".join(f"{words[i % len(words)]}{i}" for i in range(n_words))


def probe(n_words):
    r = chat([{"role": "user", "content": f"Here is a list:\n{filler(n_words)}\n\nReply with the single word OK."}],
             max_tokens=4)
    return r["usage"]["prompt_tokens"]


# Calibrate words -> tokens on a small sample.
small = probe(500)
ratio = (small - 30) / 500
target = 8175
words = int((target - 30) / ratio)
# Second calibration pass: the ratio drifts with the numbers' width.
p = probe(words)
words = int(words * (target - 30) / (p - 30))
doc = filler(words)
msgs = [{"role": "user", "content": f"Here is a list:\n{doc}\n\nRepeat the first 60 items of the list exactly, comma separated, then say OK."}]
r = chat(msgs, max_tokens=400)
p = r["usage"]["prompt_tokens"]
print(f"8K request: prompt {p} tokens, generated {r['usage']['completion_tokens']}, "
      f"finish {r['choices'][0]['finish_reason']}, {r['_secs']:.1f}s; crosses 8192: {p + r['usage']['completion_tokens'] > 8192}")
print("  answer:", repr(r["choices"][0]["message"]["content"][:200]))

msgs2 = msgs + [{"role": "assistant", "content": r["choices"][0]["message"]["content"]},
                {"role": "user", "content": "What was the 100th item? Answer with just the item."}]
r2 = chat(msgs2, max_tokens=30)
print(f"extension: prompt {r2['usage']['prompt_tokens']} tokens, cached {r2['usage']['prompt_tokens_details']['cached_tokens']}, "
      f"{r2['_secs']:.1f}s, answer {r2['choices'][0]['message']['content']!r}")

doc3 = filler(int((20000 - 30) / ratio))
r3 = chat([{"role": "user", "content": f"Here is a list:\n{doc3}\n\nWhat is the last item? Answer with just the item."}], max_tokens=30)
print(f"20K request: prompt {r3['usage']['prompt_tokens']} tokens, {r3['_secs']:.1f}s, answer {r3['choices'][0]['message']['content']!r}")
