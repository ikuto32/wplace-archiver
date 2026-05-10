# wplace Archiver 仕様書 v2.2

## 1. 目的

本プログラムは、`murolem/wplace-archives` の GitHub Release に保存されている wplace アーカイブを取得し、タグを時系列に適用して、最終状態を XYZ タイル画像として出力する rolling archive pipeline である。

主目的は次の3点である。

1. GitHub Release 上の byte-split compressed tar archive を安定して取得・復元すること
2. PNGタイルを palette index ベースの低容量中間表現へ変換し、rolling state へ安全に適用すること
3. 最終状態を `z/x/y.png` 形式のXYZタイルとして出力すること

既定のタイル空間:

```text
z = 11
x = 0..2047
y = 0..2047
tile size = 1000x1000 px
```

内部表現は透明 + 最大64色の palette index とする。

---

## 2. 正規パイプライン

正規パイプラインは `run` である。

```text
GitHub Releases
  ↓ download
split compressed tar parts
  ↓ concat + decompress + tar scan
PNG tiles
  ↓ ingest
temporary sparse tag overlay store
  ↓ apply
rolling sparse state store
  ↓ export
XYZ PNG tiles: z/x/y.png
```

正規運用ではタグごとの中間storeを恒久保存しない。1タグを ingest し、rolling state に apply し、apply 成功後に一時 tag store を削除する。

```text
1 tag ingest
  ↓
stateへapply
  ↓
tag store削除
```

`KEEP_TAG_STORES=1` の場合のみ、検証目的で tag store を保持できる。

---

## 3. 入力データ

### 3.1 GitHub Release tag

対象タグは `world-YYYY-MM-DDTHH-MM-SS.sssZ` 形式を想定する。

例:

```text
world-2025-10-31T18-37-30.453Z
```

タグは timestamp に基づき昇順に処理する。

### 3.2 Release asset

正規の新規形式:

```text
*.tar.zst.<alpha-or-number>
*.tar.zstd.<alpha-or-number>
```

後方互換形式:

```text
*.tar.gz.<alpha-or-number>
```

例:

```text
world-2026-02-09T14-20-06.949Z.tar.zst.aa
world-2026-02-09T14-20-06.949Z.tar.zst.ab
world-2026-02-09T14-20-06.949Z.tar.gz.aa
```

処理対象外:

```text
*.sha256
*.txt
*.json
*.digest
*.md5
```

asset は同一prefixかつ同一圧縮形式の split parts のみを連結対象とする。suffix は split 順に安定ソートする。

同一tagに zstd と gzip が存在する場合、zstd を優先する。

### 3.3 tar内部PNGパス

PNG path は次の形式を想定する。

```text
<prefix>/<x>/<y>.png
```

`x`, `y` は `0 <= x < GRID_TILES`, `0 <= y < GRID_TILES` の範囲内でなければならない。

---

## 4. 圧縮stream / tar 展開

wplace archive asset は、1本の tar stream を圧縮したうえで byte split したものとして扱う。

対応する圧縮形式:

```text
Zstandard: *.tar.zst.* / *.tar.zstd.*
gzip:      *.tar.gz.*
```

正規の新規形式は Zstandard とする。gzip は後方互換のため読み取り対応を維持する。

正しい展開モデル:

```text
cat *.tar.zst.*  | zstd -dc | tar -x
cat *.tar.gz.*   | gzip -dc | tar -x
```

処理上は一時結合ファイルを作らず、split parts を正しい順序で stream として読み込む。

### 4.1 Zstandard backend

Zstandard backend は Python 3.14 以降の標準ライブラリ `compression.zstd` を使用する。Python 3.13以前で zstd asset を読む必要がある場合は、同一APIを提供する `backports.zstd` を任意依存として使用できる。

### 4.2 gzip backend

gzip backend は後方互換用である。

対応backend:

```text
pigz
isal
python gzip
```

`pigz` backend では、非ゼロ終了、CRC error、deflate error、unexpected EOF、stdin feed失敗を fatal error とする。

---

## 5. PNG decode と透明処理

### 5.1 RGBA PNG

RGBA PNG は alpha channel を正とする。

```text
alpha == 0 → transparent
alpha > 0  → visible palette color
```

RGBA の `#000000` は透明扱いしない。

### 5.2 RGB PNG

RGB PNG には alpha channel がない。RGB入力に限り、`#000000` を透明背景として扱う。

```text
RGB #000000   → transparent
RGB non-black → visible palette color
```

追加の透明色指定:

```text
RGB_TRANSPARENT_COLORS=0,0,0;255,255,255;#000000
```

この指定はRGB入力にのみ適用する。

### 5.3 P mode PNG

P mode PNG は palette-indexed PNG である。P mode PNG は `tRNS` 透明情報を保持して RGBA に復元する。

```text
P mode + tRNSあり → tRNSに従ってalphaを復元する
P mode + tRNSなし → palette colorを不透明として扱い、診断対象にする
```

P mode では、黒色そのものを透明条件にしない。透明は `tRNS` metadata を正とする。

fast PNG decoder を使用する場合でも、P mode PNG は Pillow 経路で処理する。P mode + tRNS を RGB として読み込んではならない。

### 5.4 decode backend 方針

```text
P mode   → PillowでRGBA復元
RGB      → RGBとして保持
RGBA     → fast path利用可
その他   → PillowでRGBA変換
```

fast path が透明情報を落とす可能性がある場合は使用しない。

---

## 6. palette index 仕様

内部表現では色を `uint8` palette index に変換する。

| index | 意味 |
| ----- | ---- |
| 0 | transparent |
| 1..64 | visible palette color |

palette は `palette.json` で管理する。

固定paletteが指定されている場合、未知色はエラーとする。固定paletteが指定されていない場合のみ、入力データから最大64色まで動的に登録できる。

palette 更新要件:

- ingest 失敗時に部分的なpalette更新を保存しない
- タグ単位で snapshot を作成し、失敗時は rollback する
- paletteが64色を超えた場合は、そのタグの ingest を失敗扱いにする
- apply 成功後にのみ palette を保存する

---

## 7. 中間データ形式

### 7.1 sparse tile record

透明ピクセルを保存せず、可視ピクセルのみ保存する。

```text
position: uint32
value:    uint8
```

未圧縮時は1可視ピクセルあたり5 bytesである。

### 7.2 dense tile record

可視ピクセルが多いタイルでは dense fallback を使用する。

```text
shape: (1000, 1000)
dtype: uint8
0: transparent
1..64: palette color
```

未圧縮時は1タイルあたり1,000,000 bytesである。

### 7.3 中間payload圧縮

中間 shard store の tile record payload は Zstandard で圧縮する。

対象:

```text
tags/<tag>/*.bin
state/*.bin
```

圧縮単位は tile record 単位とする。shard全体を1つのzstd streamにはしない。これにより、既存のoffset/size indexでtile単位のrandom accessを維持する。

index entry:

```json
{
  "x": 0,
  "y": 0,
  "offset": 0,
  "size": 12345,
  "uncompressed_size": 500000,
  "count": 100000,
  "encoding": "sparse-u32-u8-v1",
  "compression": "zstd"
}
```

`size` は `.bin` 内の保存byte数、`uncompressed_size` は復元後payload byte数である。

### 7.4 後方互換性

```text
compression == "zstd" → zstd decompressして読む
compression == "none" → そのまま読む
compression欠落       → 旧形式として none 扱い
```

既存の非圧縮 `.bin` は再生成なしで読み込める。

新規作成する中間storeの既定圧縮は zstd とする。

```text
WPLACE_STORE_COMPRESSION=zstd
WPLACE_STORE_ZSTD_LEVEL=3
```

---

## 8. apply 仕様

1タグ分の overlay store を rolling state store へ時系列順に適用する。

合成規則:

```text
new_pixel != transparent → state_pixel を new_pixel で上書き
new_pixel == transparent → state_pixel は変更しない
```

apply は shard 単位で実行する。

```text
overlay shard + state shard
  ↓
merge
  ↓
new state shard atomic write
```

state shard は一時ファイルへ書き込み、完了後に atomic replace する。

apply 成功後にのみ、そのタグを applied checkpoint に記録する。

---

## 9. apply 安定性・予防策

### 9.1 executor

対応 executor:

```text
thread
process
sequential
isolated-process
```

既定は `thread` とする。

理由:

- Windows環境では `ProcessPoolExecutor` の長時間worker再利用で `BrokenProcessPool` が発生しうる
- native zstd / NumPy allocator の状態がworker process内に蓄積しうる
- workerから親へのpickle/IPCで巨大データを返す設計は破綻しやすい
- `thread` はworker process突然死・pickle・process pool破損のリスクを避けられる

`process` executor は高速化目的の任意設定とする。

### 9.2 process executor の予防策

process executor 使用時は、worker寿命を制限できる。

```text
WPLACE_APPLY_MAX_TASKS_PER_CHILD=1
```

`1` の場合、1 shard 処理ごとに worker process を使い捨てる。速度は落ちるが、native crash、メモリ断片化、worker状態汚染を局所化できる。

### 9.3 isolated-process executor

`isolated-process` は安全性優先の executor である。shardごとに新しい subprocess を起動し、結果は小さいJSON summaryのみで受け取る。

用途:

- `BrokenProcessPool` の予防
- native crash の局所化
- 問題shardの特定
- 長時間applyの再開性確保

### 9.4 shard checkpoint

apply は shard 単位の checkpoint を持つ。

```text
wplace_sparse_store/.apply_shards/<tag>.json
```

構造:

```json
{
  "tag": "world-...",
  "completed_shards": ["s0000_0000"],
  "failed_shards": {
    "s0063_0062": {
      "stage": "merge",
      "error": "...",
      "failed_at": "..."
    }
  }
}
```

同一タグの apply が中断された場合、完了済み shard は再実行しない。全 shard が成功した後にのみ `pipeline_state.json` の applied tag を更新する。

### 9.5 worker戻り値

apply worker が親へ返す値は最小限にする。

許可する戻り値:

```text
shard name
tile_count
visible_pixels
stored_bytes
elapsed_sec
```

tile entries 全体、巨大なmanifest、payload bytes、NumPy配列を親へ返してはならない。

### 9.6 apply probe

読み取り専用の検証ツールを提供する。

```powershell
uv run python wplace_store_probe.py \
  --store-root wplace_sparse_store \
  --tag world-... \
  --mode both \
  --exercise-compress
```

probe は shardごとに独立subprocessを起動し、index/bin整合性、zstd展開、payload形式、dry-run merge、zstd圧縮を検証する。storeは変更しない。

---

## 10. export 仕様

rolling state store から XYZ PNG を生成する。

出力先:

```text
wplace_xyz/11/<x>/<y>.png
```

PNG encode は export 段階でのみ実行する。

palette index `0` は alpha `0` として出力する。palette index `1..64` は `palette.json` のRGB値に alpha `255` を付与して出力する。

---

## 11. 状態管理

状態は `pipeline_state.json` に集約する。

```text
wplace_sparse_store/pipeline_state.json
```

再開要件:

- download済みassetはsize/digest検証後に再利用する
- ingest済みで未applyのtag storeが存在する場合、ingestを再実行せず apply へ進む
- apply済みタグは skip する
- apply中断時は shard checkpoint から再開する
- エラーで中断したタグは次回再試行する
- state shard 更新は atomic write とする
- apply成功前に applied checkpoint を更新しない

---

## 12. 診断・統計

### 12.1 ingest stats

タグごとに ingest 統計を保存する。

```text
wplace_sparse_store/diagnostics/stats/ingest_<tag>.json
```

記録項目:

- `png_tiles_seen`
- `records_written`
- `visible_pixels`
- `rgb_tiles_seen`
- `rgb_transparency_diagnostics_sample`
- `p_tiles_seen`
- `p_has_trns`
- `p_no_trns`
- `rgba_black_warning_tiles`
- `rgba_black_warning_sample`

### 12.2 apply stats

タグごとに apply 統計を保存する。

```text
wplace_sparse_store/diagnostics/stats/apply_<tag>.json
```

記録項目:

- executor
- worker数
- completed shard数
- skipped checkpoint shard数
- failed shard情報
- elapsed秒
- total tile count
- total visible pixels
- total stored bytes

---

## 13. 進捗表示

長時間処理には `tqdm(total=...)` を使用する。

対象:

- download bytes
- ingest PNG tiles
- apply shards
- export tiles
- decompress benchmark bytes
- diagnostics sample generation

`ingest` の正確なtotalが必要な場合は pre-scan を有効化する。

```text
INGEST_PRESCAN=1
```

既定では pre-scan は無効とする。

---

## 14. CLI仕様

正規コマンド:

```text
self-test
run
export
benchmark-decompress
diagnose-rgb-transparency
validate-store
clean-temp
```

補助ツール:

```text
wplace_store_probe.py
```

### 14.1 run

```powershell
uv run wplace_archiver.py run --no-export
uv run wplace_archiver.py run --limit 10
uv run wplace_archiver.py run --from-tag world-2025-10-31T18-37-30.453Z
uv run wplace_archiver.py run --to-tag world-2026-02-09T14-20-06.949Z
```

### 14.2 apply executor 設定

安定性優先の既定:

```powershell
$env:WPLACE_APPLY_EXECUTOR="thread"
$env:WPLACE_APPLY_WORKERS="4"
uv run wplace_archiver.py run --no-export
```

process executorを使う場合:

```powershell
$env:WPLACE_APPLY_EXECUTOR="process"
$env:WPLACE_APPLY_MAX_TASKS_PER_CHILD="1"
uv run wplace_archiver.py run --no-export
```

isolated processを使う場合:

```powershell
$env:WPLACE_APPLY_EXECUTOR="isolated-process"
uv run wplace_archiver.py run --no-export
```

### 14.3 export

```powershell
uv run wplace_archiver.py export
```

### 14.4 validate-store

```powershell
uv run wplace_archiver.py validate-store
```

### 14.5 benchmark-decompress

```powershell
uv run wplace_archiver.py benchmark-decompress \
  --parts "wplace_downloads/<tag>/*.tar.zst.*" \
  --mode inflate \
  --backends zstd python
```

gzip互換入力:

```powershell
uv run wplace_archiver.py benchmark-decompress \
  --parts "wplace_downloads/<tag>/*.tar.gz.*" \
  --mode inflate \
  --backends pigz isal python
```

---

## 15. 設定

設定は `Config` dataclass に集約する。

主な環境変数:

```text
WPLACE_REPO
WPLACE_DOWNLOAD_DIR
WPLACE_STORE_ROOT
WPLACE_XYZ_OUTPUT_DIR
WPLACE_INTERVAL_DAYS
WPLACE_COMPRESSION_BACKEND
WPLACE_GZIP_BACKEND
WPLACE_PIGZ_PATH
WPLACE_VALIDATE_DOWNLOAD_DIGEST
WPLACE_KEEP_ARCHIVES
WPLACE_KEEP_TAG_STORES
WPLACE_STRICT_RGBA
WPLACE_STRICT_BINARY_ALPHA
WPLACE_RGB_TRANSPARENT_COLORS
WPLACE_BLACK_WARNING_RATIO
WPLACE_STORE_COMPRESSION
WPLACE_STORE_ZSTD_LEVEL
WPLACE_APPLY_WORKERS
WPLACE_APPLY_EXECUTOR
WPLACE_APPLY_MAX_TASKS_PER_CHILD
WPLACE_INGEST_PRESCAN
WPLACE_FIXED_PALETTE
```

---

## 16. 依存関係

必須依存:

```text
aiohttp
tqdm
numpy
Pillow
```

任意依存:

```text
isal
pyspng
fpng-py>=0.0.3
backports.zstd; python_version < "3.14"
```

Zstandard archive展開と中間store圧縮の両方で `compression.zstd` / `backports.zstd` を使用する。

---

## 17. エラー処理

処理段階ごとに例外型を分ける。

```text
DownloadError
AssetValidationError
DecompressError
TarScanError
PngDecodeError
PaletteError
IngestError
ApplyError
ExportError
StateConsistencyError
```

CLIでは例外を握りつぶさない。失敗stageと要約を `pipeline_state.json` または shard checkpoint に保存し、プロセスは非ゼロ終了する。

---

## 18. atomic write 要件

次のファイルは一時ファイル経由で atomic replace する。

- `pipeline_state.json`
- `palette.json`
- `manifest.json`
- `*.index.json`
- `*.bin`
- export PNG
- `.apply_shards/<tag>.json`

---

## 19. self-test

`self-test` は本番の正規経路を検証する。

必須検証:

1. 小サイズtile/gridで合成結果が正しい
2. RGBA透明が保持される
3. RGB黒透明化が動く
4. P mode + tRNS の透明情報が保持される
5. sparse record が正しく保存・復元される
6. dense fallback が正しく保存・復元される
7. 中間 shard store の zstd 圧縮読み書きが正しい
8. 旧形式の非圧縮 shard store を読める
9. 2タグ以上を rolling apply し、後続タグが前タグを上書きする
10. apply shard checkpoint から再開できる
11. apply worker戻り値が小さいsummaryのみである
12. ingest済み・未applyのtag storeを再利用する
13. asset名フィルタがchecksum等を除外する
14. palette失敗時にrollbackされる
15. export PNG の alpha と色が正しい

---

## 20. uv実行

```powershell
uv sync
uv run wplace_archiver.py self-test --json
uv run wplace_archiver.py run --no-export
uv run wplace_archiver.py export
uv run wplace_archiver.py validate-store
```

安定性優先のapply:

```powershell
$env:WPLACE_APPLY_EXECUTOR="thread"
$env:WPLACE_APPLY_WORKERS="4"
uv run wplace_archiver.py run --no-export
```

probe:

```powershell
uv run python wplace_store_probe.py \
  --store-root wplace_sparse_store \
  --tag world-... \
  --mode both \
  --exercise-compress
```

---

## 21. 成功条件

1. 対象タグを時系列順に取得できる
2. Release assetを厳密にフィルタし、不要assetを混入させない
3. 分割 `.tar.zst.*` / `.tar.zstd.*` / `.tar.gz.*` を正しい順序で復元・展開できる
4. gzip/zstd backend失敗を確実に検出できる
5. PNGをpalette index化できる
6. RGBA alpha を保持できる
7. RGB黒背景を透明化できる
8. P mode PNG の `tRNS` 透明情報を保持できる
9. 中間 shard store をzstd圧縮で保存し、旧none圧縮storeも読める
10. tag storeを保持せずrolling stateへ適用できる
11. ingest済み・未applyタグを再利用して再開できる
12. apply shard checkpoint から再開できる
13. process pool問題を避ける executor を選択できる
14. worker戻り値を最小化し、IPC負荷を抑えられる
15. state shard更新がatomicである
16. palette更新が失敗時にrollbackされる
17. 最終XYZ PNGを生成できる
18. PNGの透過が失われない
19. 主要処理の進捗が明確に表示される
20. self-test が本番経路を検証する

---

## 22. 非目的

本プログラムは以下を目的としない。

- 全タグの中間storeを恒久保存すること
- 全ピクセルをdense配列で保存すること
- タグごとにPNGを再エンコードして保存すること
- z=11以外のXYZピラミッドを生成すること
- PNGの見た目を補正・加工すること
- split partを独立tar.gzとして扱うこと
- 7zip backendを本体で維持すること
- 外部benchmark scriptを必須成果物にすること

