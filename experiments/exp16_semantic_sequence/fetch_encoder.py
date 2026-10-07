"""Fetch the pinned encoder for offline experiments/packaging, never at job time."""

from __future__ import annotations

import hashlib
import argparse
import os
import urllib.request
from pathlib import Path

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "46605decb5369335a3847c9f41bb0b896c07dd1a"
WEIGHTS_SHA256 = (
    "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"
)
FILES = ("README.md", "config.json", "model.safetensors",
         "special_tokens_map.json", "tokenizer.json",
         "tokenizer_config.json", "vocab.txt")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(
        Path(__file__).resolve().parents[2] / ".cache" /
        "exp16_semantic_sequence" / "encoder"))
    args = parser.parse_args()
    dest = Path(args.output_dir).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        path = dest / name
        if name == "model.safetensors" and path.is_file() and \
                sha256(path) == WEIGHTS_SHA256:
            print("verified existing", name)
            continue
        url = "https://huggingface.co/%s/resolve/%s/%s" % (
            MODEL, REVISION, name)
        tmp = path.with_suffix(path.suffix + ".part")
        print("fetching", name, flush=True)
        urllib.request.urlretrieve(url, tmp)
        if name == "model.safetensors" and sha256(tmp) != WEIGHTS_SHA256:
            tmp.unlink(missing_ok=True)
            raise ValueError("downloaded model.safetensors SHA-256 mismatch")
        os.replace(tmp, path)
    print("offline encoder bundle ready:", dest)


if __name__ == "__main__":
    main()
