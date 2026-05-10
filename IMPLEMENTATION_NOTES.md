# Implementation Notes

This package is a full reimplementation aligned to the v2 rolling pipeline specification.

## Removed from normal pipeline

- `compose`
- shard-name-only compose cache
- independent complete compressed-tar asset mode
- `7zip` backend
- mandatory external benchmark script
- split state files such as `.ingested_tags.json`, `.applied_tags.json`, `.composed_shards.json`
- standalone public `ingest` CLI

## Added / changed

- `pipeline_state.json` as the single checkpoint file
- strict `*.tar.zst.<letters-or-digits>` / `*.tar.zstd.<letters-or-digits>` / `*.tar.gz.<letters-or-digits>` asset filtering
- rolling-only `run` path
- ingested-but-unapplied tag store reuse
- pigz non-zero exit code treated as fatal
- palette snapshot/rollback around ingest
- RGB transparency diagnostics
- `validate-store`
- modular package layout
- offline `self-test` for regression coverage

## Implemented modules

```text
wplace_archiver/
  cli.py
  config.py
  github.py
  download.py
  split_assets.py
  decompress.py
  png_codec.py
  palette.py
  records.py
  shard_store.py
  ingest.py
  apply.py
  export.py
  state.py
  diagnostics.py
  validation.py
  benchmark.py
  self_test.py
  errors.py
  utils.py
```

## Validation performed in this environment

```bash
cd /mnt/data/wplace_archiver_v2
python -m compileall -q .
python wplace_archiver.py self-test --json
```

Result: passed.

The offline self-test validates:

- strict asset filtering excludes checksum-like files
- byte-split `.tar.zst.*` stream reconstruction with `.tar.gz.*` compatibility
- RGBA transparency preservation
- RGB corner-background transparency inference
- rolling apply overwrite semantics across two tags
- ingested tag-store checkpoint semantics
- palette rollback after ingest failure
- XYZ PNG export

### Spec 19 coverage mapping (self-test)

`wplace_archiver/self_test.py` defines `SPEC_19_COVERAGE` and emits `spec_19_coverage` in `self-test --json` output so maintenance can track each requirement directly.

| Spec 19 item | Self-test ID | Main assertion point |
| --- | --- | --- |
| 1 | `small_grid_synth_correctness` | tiny 4x4 fixtures ingested/applied/exported end-to-end |
| 2 | `rgba_transparency_preserved` | RGBA alpha remains transparent/visible in state and export |
| 3 | `rgb_black_transparency` | RGB black inferred transparent, non-black remains visible |
| 4 | `p_mode_trns_transparency` | P mode `tRNS` transparency survives decode |
| 5 | `sparse_record_roundtrip` | sparse tile encoding (`sparse-u32-u8-v1`) emitted and read |
| 6 | `dense_fallback_roundtrip` | dense fallback encoding (`dense-u8-v1`) emitted and read |
| 7 | `zstd_store_roundtrip` | zstd-compressed shard records round-trip (if zstd available) |
| 8 | `legacy_uncompressed_store_compat` | index without compression metadata reads as `none` |
| 9 | `rolling_apply_overwrite` | later tag overwrites earlier tag on same pixel |
| 10 | `apply_shard_checkpoint_resume` | `.apply_shards/<tag>.json` completion + shard list |
| 11 | `apply_worker_small_summary` | apply summary keeps only compact keys |
| 12 | `ingested_unapplied_tag_store_reuse` | `PipelineState` ingest checkpoint + manifest reuse |
| 13 | `asset_name_filter_excludes_checksums` | `.sha256` etc excluded from release asset selection |
| 14 | `palette_rollback_on_failure` | palette snapshot restored on ingest failure |
| 15 | `export_png_alpha_and_color` | exported PNG pixel alpha/color checked |

## Not validated here

- Live GitHub API download against `murolem/wplace-archives`
- Full-size 1000x1000 tile throughput
- Real archive RGB transparency visual audit
- pigz benchmark on real assets
- Digest validation against real GitHub asset metadata


## Zstandard

Zstandard is the preferred archive compression. The implementation reads `.tar.zst.*` and `.tar.zstd.*` split streams with `compression.zstd` on Python 3.14+, and falls back to `backports.zstd` on Python <=3.13. Legacy `.tar.gz.*` streams remain supported.

## Store compression update

Intermediate shard `.bin` files now store per-tile record payloads compressed with Zstandard by default. The shard index includes:

- `compression`: `zstd` or `none`
- `size`: stored byte length inside the shard `.bin`
- `uncompressed_size`: payload byte length after decompression

Records without `compression` metadata are interpreted as legacy uncompressed records (`none`). This preserves backward compatibility with stores generated before this change.


## Apply preventive changes

This build adds prevention and recovery controls for long-running apply jobs:

- Default `WPLACE_APPLY_EXECUTOR` is `thread` instead of `process`.
- `process` executor supports `WPLACE_APPLY_MAX_TASKS_PER_CHILD`.
- New `isolated-process` executor runs one shard per subprocess.
- Apply writes shard-level checkpoints to `wplace_sparse_store/.apply_shards/<tag>.json`.
- Worker return payloads are intentionally small summaries only.
- `wplace_store_probe.py` is included for read-only shard validation and dry-run merge checks.
