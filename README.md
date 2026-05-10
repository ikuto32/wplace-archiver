# wplace Archiver v2

Rolling sparse palette-index pipeline for `murolem/wplace-archives`.

## Design

The normal path is intentionally reduced to:

```text
GitHub Release split assets
  -> download
  -> concat .tar.zst.* stream; legacy .tar.gz.* is supported
  -> ingest temporary tag overlay store
  -> apply to rolling state store
  -> delete tag store by default
  -> export XYZ PNG
```

`compose` and independent-tar asset modes are removed from the normal pipeline.

## Setup with uv

```powershell
uv init wplace-archiver
cd wplace-archiver
uv add aiohttp tqdm numpy Pillow
uv add --optional fast isal pyspng fpng-py
# 注: fpng-py は PyPI で 0.0.3 系が利用されるため、pyproject.toml では >=0.0.3 とする。
```

Or run this packaged project directly:

```powershell
cd wplace_archiver_v2
uv sync
uv run wplace_archiver.py self-test
```

## Commands

```powershell
uv run wplace_archiver.py self-test
uv run wplace_archiver.py run --no-export
uv run wplace_archiver.py export
uv run wplace_archiver.py validate-store
```

Benchmark local split parts:

```powershell
uv run wplace_archiver.py benchmark-decompress `
  --parts "wplace_downloads/world-2026-02-09T14-20-06.949Z/*.tar.zst.*" `
  --mode inflate `
  --backends zstd
```

Legacy gzip benchmark:

```powershell
uv run wplace_archiver.py benchmark-decompress `
  --parts "wplace_downloads/world-2026-02-09T14-20-06.949Z/*.tar.gz.*" `
  --mode inflate `
  --backends pigz isal python
```

RGB transparency diagnostics:

```powershell
uv run wplace_archiver.py diagnose-rgb-transparency `
  --parts "wplace_downloads/world-2026-02-09T14-20-06.949Z/*.tar.zst.*" `
  --sample 200 `
  --out wplace_sparse_store/diagnostics/rgb_transparency_samples
```

## Environment

Common environment variables:

```text
WPLACE_REPO=murolem/wplace-archives
WPLACE_DOWNLOAD_DIR=./wplace_downloads
WPLACE_STORE_ROOT=./wplace_sparse_store
WPLACE_XYZ_OUTPUT_DIR=./wplace_xyz
WPLACE_COMPRESSION_BACKEND=auto # auto | zstd | gzip
WPLACE_GZIP_BACKEND=auto       # gzip compatibility only: auto | pigz | isal | python
WPLACE_KEEP_TAG_STORES=0
WPLACE_KEEP_ARCHIVES=0
WPLACE_RGB_TRANSPARENCY_MODE=corners
WPLACE_RGB_TRANSPARENT_DOMINANT_MIN=0.90
WPLACE_VALIDATE_DOWNLOAD_DIGEST=0
```

## Checkpointing

All pipeline checkpoint state is stored in:

```text
wplace_sparse_store/pipeline_state.json
```

Re-running the same `run` command resumes at tag boundaries:

- applied tags are skipped
- ingested but unapplied tag stores are reused
- failed tags are retried
- state shards are atomically replaced

## Notes

- `7zip` backend is intentionally not implemented.
- `compose` is intentionally not part of the normal CLI.
- `self-test` uses tiny local synthetic archives and does not require network access.


## Compression

New split archives should use Zstandard: `*.tar.zst.*` or `*.tar.zstd.*`. Existing `*.tar.gz.*` releases remain readable for backward compatibility. Python 3.14+ uses the standard-library `compression.zstd`; Python <=3.13 uses `backports.zstd` via the project dependency marker.

## Intermediate store compression

New intermediate shard records are compressed with Zstandard by default:

```text
WPLACE_STORE_COMPRESSION=zstd  # zstd | none
WPLACE_STORE_ZSTD_LEVEL=3
```

Compression is recorded per tile record in each shard index using `compression` and `uncompressed_size` fields. Older stores that do not have these fields are treated as `compression="none"` and remain readable.

For debugging or compatibility, disable intermediate compression with:

```powershell
uv run wplace_archiver.py --store-compression none run --no-export
```


## Apply stability controls

The default apply executor is now `thread` to avoid long-running `ProcessPoolExecutor`
failures on Windows when native zstd/NumPy code is involved.

Recommended stable run:

```powershell
$env:WPLACE_APPLY_EXECUTOR="thread"
$env:WPLACE_APPLY_WORKERS="4"
uv run wplace_archiver.py run --no-export
```

If you need process-based execution, limit worker lifetime:

```powershell
$env:WPLACE_APPLY_EXECUTOR="process"
$env:WPLACE_APPLY_MAX_TASKS_PER_CHILD="1"
uv run wplace_archiver.py run --no-export
```

For maximum isolation:

```powershell
$env:WPLACE_APPLY_EXECUTOR="isolated-process"
uv run wplace_archiver.py run --no-export
```

Apply progress is checkpointed per shard under:

```text
wplace_sparse_store/.apply_shards/<tag>.json
```

If apply is interrupted, completed shards are skipped on the next run.

To validate a tag store and state store without modifying them:

```powershell
uv run python wplace_store_probe.py `
  --store-root wplace_sparse_store `
  --tag world-2026-01-20T10-37-37.596Z `
  --mode both `
  --exercise-compress
```
