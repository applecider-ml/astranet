#!/usr/bin/env python3
"""Download and unpack the paper's twelve checkpoints into models/.

    python tools/fetch_weights.py --url <release or Zenodo URL of astranet_v2_checkpoints.tar.gz>
    python tools/fetch_weights.py --file /path/to/astranet_v2_checkpoints.tar.gz

The URL can also be given as $ASTRANET_WEIGHTS_URL. The archive is verified against its MD5.
The novelty bundle of the service is a second, small asset:

    python tools/fetch_weights.py --novelty-url <URL of novelty_v1.joblib>      (or --novelty-file)
"""
import argparse, hashlib, os, sys, tarfile, urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MD5 = "329db18a3b75f6d90e4de77376f1736f"
NAME = "astranet_v2_checkpoints.tar.gz"
NOVELTY_MD5 = "4f1b14e832ce7aa1da299a63a4cbbedc"
NOVELTY_NAME = "novelty_v1.joblib"


def md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""): h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--url", default=os.environ.get("ASTRANET_WEIGHTS_URL")); ap.add_argument("--file")
    ap.add_argument("--novelty-url", default=os.environ.get("ASTRANET_NOVELTY_URL")); ap.add_argument("--novelty-file")
    ap.add_argument("--dest", default=str(REPO)); a = ap.parse_args()
    if a.novelty_url or a.novelty_file:
        import shutil
        out = Path(a.dest) / "models" / NOVELTY_NAME; out.parent.mkdir(parents=True, exist_ok=True)
        if a.novelty_file: shutil.copy(a.novelty_file, out)
        else: print("downloading", a.novelty_url); urllib.request.urlretrieve(a.novelty_url, out)
        if md5(out) != NOVELTY_MD5: sys.exit(f"MD5 mismatch for {out}")
        print("novelty bundle at", out)
        if not (a.file or a.url): return
    src = a.file
    if src is None:
        if not a.url: sys.exit("give --url, --file or $ASTRANET_WEIGHTS_URL")
        src = str(Path(a.dest) / NAME); print("downloading", a.url); urllib.request.urlretrieve(a.url, src)
    got = md5(src)
    if got != MD5: sys.exit(f"MD5 mismatch: {got} (expected {MD5})")
    with tarfile.open(src) as t:
        root = Path(a.dest).resolve()
        for m in t.getmembers():
            if not (root / m.name).resolve().is_relative_to(root) or m.issym() or m.islnk(): sys.exit(f"unsafe member {m.name}")
        t.extractall(root)
    n = len(list((root / "models").glob("trained_v7fix*/seed*/checkpoint_*.pt"))); print(f"{n} checkpoints under {root / 'models'}")


if __name__ == "__main__":
    main()
