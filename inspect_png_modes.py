from __future__ import annotations

import argparse
import gzip
import json
import re
import tarfile
from collections import Counter
from pathlib import Path

from PIL import Image
from tqdm import tqdm


_SPLIT_RE = re.compile(r"^(.*?\.tar\.gz\.)([A-Za-z]+|\d+)$")


def split_key(path: Path):
    name = path.name
    m = _SPLIT_RE.match(name)
    if not m:
        return name
    suffix = m.group(2)
    if suffix.isdigit():
        return (m.group(1), 0, int(suffix))
    return (m.group(1), 1, suffix.lower())


class ConcatReader:
    def __init__(self, paths: list[Path]):
        self.paths = paths
        self.i = 0
        self.f = None

    def read(self, n: int = -1) -> bytes:
        chunks = []
        remaining = n

        while self.i < len(self.paths):
            if self.f is None:
                self.f = self.paths[self.i].open("rb")

            data = self.f.read(remaining if remaining >= 0 else -1)
            if data:
                chunks.append(data)
                if n >= 0:
                    remaining -= len(data)
                    if remaining <= 0:
                        break
            else:
                self.f.close()
                self.f = None
                self.i += 1

        return b"".join(chunks)

    def close(self):
        if self.f is not None:
            self.f.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", required=True, help="例: wplace_downloads/<tag>/*.tar.gz.*")
    ap.add_argument("--out", default="png_mode_report.json")
    ap.add_argument("--checkpoint-every", type=int, default=1000)
    args = ap.parse_args()

    paths = sorted(Path().glob(args.parts), key=split_key)
    if not paths:
        raise SystemExit(f"no parts matched: {args.parts}")

    out = Path(args.out)
    modes = Counter()
    samples: dict[str, list[str]] = {}
    total_png = 0

    concat = ConcatReader(paths)
    try:
        gz = gzip.GzipFile(fileobj=concat, mode="rb")
        with tarfile.open(fileobj=gz, mode="r|") as tar:
            for member in tqdm(tar, desc="scan PNG modes", unit="entry"):
                if not member.isfile() or not member.name.lower().endswith(".png"):
                    continue

                f = tar.extractfile(member)
                if f is None:
                    continue

                img = Image.open(f)
                mode = img.mode
                modes[mode] += 1
                total_png += 1
                samples.setdefault(mode, [])
                if len(samples[mode]) < 20:
                    samples[mode].append(member.name)

                if total_png % args.checkpoint_every == 0:
                    out.write_text(
                        json.dumps(
                            {
                                "total_png": total_png,
                                "modes": dict(modes),
                                "samples": samples,
                                "checkpoint": True,
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )

    finally:
        concat.close()

    result = {
        "total_png": total_png,
        "modes": dict(modes),
        "samples": samples,
        "checkpoint": False,
    }
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())