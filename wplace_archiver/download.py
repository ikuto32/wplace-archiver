from __future__ import annotations

import asyncio
from pathlib import Path

import aiohttp
from tqdm import tqdm

from .config import Config
from .errors import DownloadError
from .utils import digest_hex_from_github, sha256_file


def download_is_valid(path: Path, expected_size: int | None = None, expected_digest: str | None = None, cfg: Config | None = None) -> bool:
    if not path.exists():
        return False
    if expected_size is not None and expected_size >= 0 and path.stat().st_size != expected_size:
        return False
    if cfg is not None and cfg.validate_download_digest:
        digest_hex = digest_hex_from_github(expected_digest)
        if digest_hex and sha256_file(path).lower() != digest_hex:
            return False
    return True


async def _download_file(session: aiohttp.ClientSession, url: str, path: Path, expected_size: int | None, digest: str | None, cfg: Config, sem: asyncio.Semaphore, position: int) -> Path:
    async with sem:
        path.parent.mkdir(parents=True, exist_ok=True)
        if download_is_valid(path, expected_size, digest, cfg):
            return path
        partial = path.with_suffix(path.suffix + ".part")
        existing = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "wplace-archiver-v2"}
        mode = "wb"
        if existing and expected_size and existing < expected_size:
            headers["Range"] = f"bytes={existing}-"
            mode = "ab"
        async with session.get(url, headers=headers) as resp:
            if resp.status == 416:
                partial.unlink(missing_ok=True)
                existing = 0
                mode = "wb"
                async with session.get(url, headers={"User-Agent": "wplace-archiver-v2"}) as retry_resp:
                    retry_resp.raise_for_status()
                    await _stream_response(retry_resp, partial, mode, existing, expected_size, position)
            else:
                if resp.status >= 400:
                    text = await resp.text()
                    raise DownloadError(f"download failed {resp.status}: {url}: {text[:300]}")
                # If server ignored Range, restart.
                if existing and resp.status == 200:
                    existing = 0
                    mode = "wb"
                await _stream_response(resp, partial, mode, existing, expected_size, position)
        if expected_size is not None and partial.stat().st_size != expected_size:
            raise DownloadError(f"size mismatch after download: {path.name}: expected={expected_size}, got={partial.stat().st_size}")
        if cfg.validate_download_digest:
            digest_hex = digest_hex_from_github(digest)
            if digest_hex and sha256_file(partial).lower() != digest_hex:
                raise DownloadError(f"sha256 mismatch after download: {path.name}")
        partial.replace(path)
        return path


async def _stream_response(resp: aiohttp.ClientResponse, partial: Path, mode: str, existing: int, expected_size: int | None, position: int) -> None:
    total = expected_size if expected_size is not None else int(resp.headers.get("Content-Length", 0) or 0) + existing
    with tqdm(total=total or None, initial=existing, unit="B", unit_scale=True, desc=partial.name, position=position, leave=False) as pbar:
        with partial.open(mode) as f:
            async for chunk in resp.content.iter_chunked(1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


async def download_assets_async(cfg: Config, tag: str, assets: list[dict]) -> list[Path]:
    tag_dir = cfg.download_dir / tag
    sem = asyncio.Semaphore(cfg.max_concurrent_downloads)
    headers = {"Accept": "application/octet-stream", "User-Agent": "wplace-archiver-v2"}
    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = []
        for i, asset in enumerate(assets):
            name = str(asset["name"])
            url = str(asset.get("browser_download_url") or asset.get("url"))
            size = int(asset.get("size", -1)) if asset.get("size") is not None else None
            digest = asset.get("digest")
            tasks.append(_download_file(session, url, tag_dir / name, size, digest, cfg, sem, i))
        return await asyncio.gather(*tasks)


def download_assets(cfg: Config, tag: str, assets: list[dict]) -> list[Path]:
    return asyncio.run(download_assets_async(cfg, tag, assets))
