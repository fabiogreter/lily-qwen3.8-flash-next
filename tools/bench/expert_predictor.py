"""Whole-token expert predictor, trained and evaluated leave-one-session-out
(docs/low-ram-experts.md, "A whole-token predictor from the hidden state").

Input: recorded sessions as lily-trace-routing writes them with --hidden
(STEM.u32 token ids, STEM.routing.u32, STEM.hidden.bf16) plus STEM.json
(requests as [p0, p1, p2]). For each held-out session the model is trained
on the other sessions' positions: position p's final hidden state (and,
with --ids, token p+1's id) predicts position p+1's routing at every layer
(multi-hot over layers x experts). It then writes, for every decoded
position q of the held-out session, the top WIDTH experts per layer
predicted from position q-1, as STEM.pred.u16 in the layout
lily-cache-replay reads for a token-start prefetch (--prefetch 0:TOP
--pred-width 40).

Usage: .venv/bin/python tools/bench/expert_predictor.py <traces> <hidden> <out>
           --usage <ranking> --sessions STEM,STEM,... [--ids] [--cold W]
           [--epochs N] [--hidden-units H] [--batch B] [--decode-only]
           [--slots N] [--lru-share S] [--only STEM]

The hidden states are as sensitive as the sessions they come from (they
can largely be inverted back to tokens): keep <hidden> and <out> outside
the repository, as the traces themselves.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

# Qwen3.8-Flash-Next: MoE layers, experts per layer, routed per token,
# hidden size, vocabulary.
LAYERS, EXPERTS, TOPK, H, VOCAB = 48, 512, 10, 2560, 248320
WIDTH = 40


def read_json(path):
    with open(path) as f:
        return json.load(f)


def load(traces, hidden_dir, s):
    meta = read_json(os.path.join(traces, f"{s}.json"))
    n = meta["tokens"]
    routing = np.fromfile(os.path.join(traces, f"{s}.routing.u32"), dtype="<u4")
    routing = routing.reshape(LAYERS, n, TOPK)
    ids = np.fromfile(os.path.join(traces, f"{s}.u32"), dtype="<u4")
    hid = np.fromfile(os.path.join(hidden_dir, f"{s}.hidden.bf16"), dtype="<u2")
    hid = hid.reshape(n, H)
    assert len(ids) == n and int(ids.max()) < VOCAB, f"{s}: token ids"
    decode = np.concatenate(
        [np.arange(a[1], a[2]) for a in meta["requests"]] or [np.zeros(0, int)]
    ).astype(np.int64)
    return dict(name=s, n=n, routing=routing, ids=ids, hidden=hid, decode=decode)


def pinned_mask(usage_path, pinned):
    """True for the slices the loaded ranking pins at load (count
    descending, ties by layer then expert, as `UsageRanking::ranked`)."""
    u = read_json(usage_path)
    counts = np.array(u["counts"], dtype=np.float64).reshape(-1)
    assert counts.size == LAYERS * EXPERTS, f"{usage_path}: {counts.size} counts"
    order = np.lexsort((np.arange(counts.size), -counts))
    mask = np.zeros(counts.size, dtype=bool)
    mask[order[:pinned]] = True
    return mask


def bf16(a):
    return torch.from_numpy(a.view(np.int16).copy()).view(torch.bfloat16)


class Model(torch.nn.Module):
    def __init__(self, units, vocab):
        super().__init__()
        self.norm = torch.nn.LayerNorm(H)
        self.inp = torch.nn.Linear(H, units)
        # Token p+1's id through a small embedding: 248K ids x 64.
        self.ids = torch.nn.Embedding(vocab, 64) if vocab else None
        self.ids_proj = torch.nn.Linear(64, units) if vocab else None
        self.out = torch.nn.Linear(units, LAYERS * EXPERTS)

    def forward(self, h, tok=None):
        x = self.inp(self.norm(h))
        if self.ids is not None:
            x = x + self.ids_proj(self.ids(tok))
        return self.out(torch.nn.functional.gelu(x))


def targets(sess, pos):
    """Multi-hot routing of positions `pos`: [len(pos), LAYERS * EXPERTS]."""
    r = sess["routing"][:, pos, :]  # [L, B, K]
    t = np.zeros((len(pos), LAYERS * EXPERTS), dtype=np.float32)
    rows = np.repeat(np.arange(len(pos)), LAYERS * TOPK)
    cols = (np.arange(LAYERS)[None, :, None] * EXPERTS + r.transpose(1, 0, 2)).reshape(-1)
    t[rows, cols] = 1.0
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces")
    ap.add_argument("hidden")
    ap.add_argument("out")
    ap.add_argument("--usage", required=True)
    ap.add_argument("--ids", action="store_true")
    ap.add_argument("--cold", type=float, default=1.0,
                    help="loss weight of slices the shipped ranking does not pin")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--hidden-units", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--decode-only", action="store_true",
                    help="train on targets at decoded positions only")
    ap.add_argument("--slots", type=int, default=16246,
                    help="the cache's slots (the 64 GB plan's by default)")
    ap.add_argument("--lru-share", type=float, default=0.1,
                    help="the cache's LRU share; the rest is pinned")
    ap.add_argument("--only", default=None)
    ap.add_argument("--sessions", required=True,
                    help="comma-separated session stems, in serving order")
    args = ap.parse_args()
    sessions = args.sessions.split(",")
    os.makedirs(args.out, exist_ok=True)
    traces = os.path.abspath(args.traces)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data = {s: load(args.traces, args.hidden, s) for s in sessions}
    vocab = VOCAB if args.ids else 0
    pinned = round(args.slots * (1.0 - args.lru_share))
    weight = torch.ones(LAYERS * EXPERTS)
    weight[~torch.from_numpy(pinned_mask(args.usage, pinned))] = args.cold
    weight = weight.to(dev)
    log = {}
    for held in sessions:
        if args.only and held != args.only:
            continue
        t0 = time.time()
        torch.manual_seed(0)
        model = Model(args.hidden_units, vocab).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        # Training pairs (session, p): input p, target p + 1.
        pairs = []
        for s in sessions:
            if s == held:
                continue
            d = data[s]
            tgt = d["decode"] if args.decode_only else np.arange(1, d["n"])
            tgt = tgt[tgt >= 1]
            if len(tgt):
                pairs.append((s, tgt - 1))
        steps = 0
        for epoch in range(args.epochs):
            order = [(s, p) for s, ps in pairs for p in np.array_split(
                np.random.default_rng(epoch).permutation(ps),
                max(1, len(ps) // args.batch))]
            np.random.default_rng(100 + epoch).shuffle(order)
            total = 0.0
            for s, p in order:
                p = np.sort(p)
                d = data[s]
                h = bf16(d["hidden"][p]).to(dev).float()
                tok = torch.from_numpy(d["ids"][p + 1].astype(np.int64)).to(dev)
                y = torch.from_numpy(targets(d, p + 1)).to(dev)
                logit = model(h, tok)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logit, y, weight=weight)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
                steps += 1
            print(f"{held} epoch {epoch}: loss {total / max(1, len(order)):.5f}", flush=True)
        # Predict the held-out session's decoded positions from p - 1.
        d = data[held]
        q = d["decode"][d["decode"] >= 1]
        pred = np.zeros((2, LAYERS, d["n"], WIDTH), dtype="<u2")
        model.eval()
        hits = 0
        with torch.no_grad():
            for chunk in np.array_split(q, max(1, len(q) // 2048)):
                h = bf16(d["hidden"][chunk - 1]).to(dev).float()
                tok = torch.from_numpy(d["ids"][chunk].astype(np.int64)).to(dev)
                logit = model(h, tok).view(-1, LAYERS, EXPERTS)
                top = torch.topk(logit, WIDTH, dim=-1).indices.cpu().numpy()  # [B, L, W]
                pred[0][:, chunk, :] = top.transpose(1, 0, 2)
                actual = d["routing"][:, chunk, :].transpose(1, 0, 2)  # [B, L, K]
                hits += sum(len(set(a) & set(b)) for a, b in zip(
                    actual.reshape(-1, TOPK), top[:, :, :TOPK].reshape(-1, TOPK)))
        overlap = hits / (len(q) * LAYERS * TOPK)
        pred.tofile(os.path.join(args.out, f"{held}.pred.u16"))
        for ext in ("json", "routing.u32", "u32"):
            dst = os.path.join(args.out, f"{held}.{ext}")
            if not os.path.lexists(dst):
                os.symlink(os.path.join(traces, f"{held}.{ext}"), dst)
        log[held] = dict(top10_overlap=overlap, steps=steps, secs=time.time() - t0)
        print(f"{held}: top-10 overlap {overlap:.3f} on {len(q)} decoded positions "
              f"({time.time() - t0:.0f}s)", flush=True)
        with open(os.path.join(args.out, "train-log.json"), "w") as f:
            json.dump(dict(args=vars(args), sessions=log), f, indent=1)


if __name__ == "__main__":
    main()
