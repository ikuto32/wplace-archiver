from __future__ import annotations

import asyncio
import subprocess
import time
from datetime import datetime

import aiohttp
from tqdm import tqdm

from .config import Config
from .errors import DownloadError
from .utils import tag_datetime


def get_all_tags(cfg: Config) -> list[dict]:
    url = f"https://github.com/{cfg.repo}.git"
    result = subprocess.run(["git", "ls-remote", "--tags", url], capture_output=True, text=True, check=True)
    tags: dict[str, dict] = {}
    for line in result.stdout.splitlines():
        if "refs/tags/world-" not in line:
            continue
        tag = line.split("refs/tags/")[-1].replace("^{}", "")
        dt = tag_datetime(tag)
        if dt != datetime.max:
            tags[tag] = {"tag_name": tag, "published_at": dt}
    return sorted(tags.values(), key=lambda x: x["published_at"])


def select_target_tags(tags_info: list[dict], cfg: Config, *, limit: int | None = None, from_tag: str | None = None, to_tag: str | None = None) -> list[str]:
    target: list[str] = []
    last_date = None
    for info in tags_info:
        tag = info["tag_name"]
        if from_tag and tag_datetime(tag) < tag_datetime(from_tag):
            continue
        if to_tag and tag_datetime(tag) > tag_datetime(to_tag):
            continue
        pub = info["published_at"]
        if last_date is None or (pub - last_date).days >= cfg.interval_days:
            target.append(tag)
            last_date = pub
    if limit is not None:
        target = target[:limit]
    return target


async def fetch_api_with_retry(session: aiohttp.ClientSession, url: str):
    while True:
        async with session.get(url) as resp:
            if resp.status in (403, 429):
                reset = int(resp.headers.get("X-RateLimit-Reset", 0))
                wait_sec = max(reset - int(time.time()), 60) if reset else int(resp.headers.get("Retry-After", 60))
                tqdm.write(f"[GitHub API制限] {wait_sec}秒待機して再試行")
                await asyncio.sleep(wait_sec)
                continue
            if resp.status >= 400:
                text = await resp.text()
                raise DownloadError(f"GitHub API failed {resp.status}: {url}: {text[:300]}")
            return await resp.json()


async def get_release_assets_async(cfg: Config, tag: str) -> list[dict]:
    url = f"https://api.github.com/repos/{cfg.repo}/releases/tags/{tag}"
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "wplace-archiver-v2"}
    async with aiohttp.ClientSession(headers=headers) as session:
        data = await fetch_api_with_retry(session, url)
        return list(data.get("assets", []))


def get_release_assets(cfg: Config, tag: str) -> list[dict]:
    return asyncio.run(get_release_assets_async(cfg, tag))
