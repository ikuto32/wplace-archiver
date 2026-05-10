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
    m = _SPLIT_RE.match(path.name)
    if not m:
        return path.name
    suffix = m.group(2)
    return (m.group(1), 0, int(suffix)) if suffix.isdigit() else (m.group(1), 1, suffix.lower())


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


def black_palette_indices(img: Image.Image) -> list[int]:
    pal = img.getpalette()
    if not pal:
        return []
    out = []
    for i in range(0, len(pal) // 3):
        r, g, b = pal[i * 3 : i * 3 + 3]
        if (r, g, b) == (0, 0, 0):
            out.append(i)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", required=True)
    ap.add_argument("--out", default="p_mode_transparency_report.json")
    ap.add_argument("--checkpoint-every", type=int, default=1000)
    args = ap.parse_args()

    paths = sorted(Path().glob(args.parts), key=split_key)
    if not paths:
        raise SystemExit(f"no parts matched: {args.parts}")

    out = Path(args.out)
    total_p = 0
    has_trns = 0
    no_trns = 0
    black_index_counter = Counter()
    samples = {
        "has_trns": [],
        "no_trns_black_palette": [],
        "no_trns_no_black_palette": [],
    }

    concat = ConcatReader(paths)
    try:
        gz = gzip.GzipFile(fileobj=concat, mode="rb")
        with tarfile.open(fileobj=gz, mode="r|") as tar:
            for member in tqdm(tar, desc="inspect P transparency", unit="entry"):
                if not member.isfile() or not member.name.lower().endswith(".png"):
                    continue

                f = tar.extractfile(member)
                if f is None:
                    continue

                img = Image.open(f)
                if img.mode != "P":
                    continue

                total_p += 1
                black_indices = black_palette_indices(img)
                for idx in black_indices:
                    black_index_counter[idx] += 1

                if "transparency" in img.info:
                    has_trns += 1
                    if len(samples["has_trns"]) < 20:
                        samples["has_trns"].append(member.name)
                else:
                    no_trns += 1
                    if black_indices:
                        if len(samples["no_trns_black_palette"]) < 20:
                            samples["no_trns_black_palette"].append(
                                {"path": member.name, "black_indices": black_indices}
                            )
                    else:
                        if len(samples["no_trns_no_black_palette"]) < 20:
                            samples["no_trns_no_black_palette"].append(member.name)

                if total_p % args.checkpoint_every == 0:
                    out.write_text(
                        json.dumps(
                            {
                                "total_p": total_p,
                                "has_trns": has_trns,
                                "no_trns": no_trns,
                                "black_index_counter": dict(black_index_counter),
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
        "total_p": total_p,
        "has_trns": has_trns,
        "no_trns": no_trns,
        "black_index_counter": dict(black_index_counter),
        "samples": samples,
        "checkpoint": False,
    }
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())