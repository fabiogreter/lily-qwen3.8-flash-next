#!/usr/bin/env python3
"""Puts the n-gram table's page-cache residency into a fixed, partly cold
state before an A/B session: every checkpoint file holding table shards is
mapped read-only, and a deterministic selection of 1 MB blocks (FRACTION of
them, chosen by a hash of file index and block index, the same selection every
time) is dropped from the page cache with msync(MS_INVALIDATE), the call
vmtouch uses to evict on macOS. The rest is read (touched) so it is resident.
Reports residency (mincore) before and after, for the table files only.

    python3 tools/bench/evict_table.py <checkpoint dir> [fraction=0.45]

With 0.6 the full checkpoint keeps about 17 GB of its table files resident,
close to the 18.2 GB the service kept after its background preload under
interactive use; a server started with `--ngram-preload false` then starts
from that state (the preload would read the rest back in seconds on an idle
machine).

Clean, read-only file pages only: nothing is written, and the pages dropped
are the table's own, which a request re-reads from the SSD on demand."""
import ctypes, ctypes.util, hashlib, json, os, sys, time

libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_longlong]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.msync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
PROT_READ, MAP_SHARED, MS_INVALIDATE = 1, 1, 0x0002
PAGE = os.sysconf("SC_PAGE_SIZE")
BLOCK = 1 << 20


def resident(addr, length):
    pages = (length + PAGE - 1) // PAGE
    vec = ctypes.create_string_buffer(pages)
    if libc.mincore(ctypes.c_void_p(addr), length, vec) != 0:
        raise OSError(ctypes.get_errno(), "mincore")
    return sum(b & 1 for b in vec.raw) * PAGE


def main():
    ckpt = sys.argv[1]
    fraction = float(sys.argv[2]) if len(sys.argv) > 2 else 0.45
    index = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
    files = sorted({v for k, v in index.items() if "ngram_embedding" in k})
    t = time.time()
    before = after = total = dropped = 0
    for fi, name in enumerate(files):
        path = os.path.join(ckpt, name)
        fd = os.open(path, os.O_RDONLY)
        length = os.fstat(fd).st_size
        addr = libc.mmap(None, length, PROT_READ, MAP_SHARED, fd, 0)
        if addr in (None, ctypes.c_void_p(-1).value):
            raise OSError(ctypes.get_errno(), f"mmap {path}")
        before += resident(addr, length)
        total += length
        keep = []
        for off in range(0, length, BLOCK):
            n = min(BLOCK, length - off)
            h = int.from_bytes(hashlib.blake2b(f"{fi}:{off // BLOCK}".encode(), digest_size=4).digest(), "little")
            if h / 2**32 < fraction:
                if libc.msync(ctypes.c_void_p(addr + off), n, MS_INVALIDATE) != 0:
                    raise OSError(ctypes.get_errno(), "msync")
                dropped += n
            else:
                keep.append((off, n))
        # The kept blocks are read so that every session starts from the same
        # state, not from whatever the previous one left.
        with open(path, "rb", buffering=0) as f:
            for off, n in keep:
                f.seek(off)
                f.read(n)
        after += resident(addr, length)
        libc.munmap(ctypes.c_void_p(addr), length)
        os.close(fd)
    gb = lambda b: b / 1e9
    print(f"table files: {len(files)}, {gb(total):.1f} GB; resident before {gb(before):.1f} GB, "
          f"after {gb(after):.1f} GB; dropped {gb(dropped):.1f} GB ({fraction:.0%} of the blocks) in {time.time() - t:.1f}s")


if __name__ == "__main__":
    main()
