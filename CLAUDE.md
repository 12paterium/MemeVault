# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MemeVault — a semantic vector search system for meme images. Users type a natural language query and get the most relevant memes via cosine similarity on text embeddings.

## Architecture (3-layer decoupled)

- **Data Layer** — `data/metadata.json`: annotated meme metadata (path, text, tags, character, emotion, usage, background). The "source of truth."
- **Feature Layer** — `data/embeddings.npy`: cached text embeddings + `data/embeddings_meta.json` (mode marker). Never committed; can be rebuilt at any time.
- **Retrieval Layer** — numpy cosine similarity search over embeddings.

Key design rules: data !== features !== index. Embeddings are rebuildable caches, not assets. IDs are MD5 hashes of file content (not tied to filenames).

## Embedding Strategy

- **Text mode** (`build` / `build-text`): each semantic field (text, tags, emotion, usage, character, background, filename) is embedded separately, then average-pooled into one vector per entry. This prevents long text descriptions from drowning out short but important fields like tags or emotion.
- **Image mode** (`build-image`): embeds the image directly via the cross-modal model.
- A companion `embeddings_meta.json` file records which mode was used. Search auto-detects the mode and rebuilds stale embeddings on the fly if entry count changed.

## Code Structure

```
meme_vault/
  vault.py       — MemeVault class: primary public API
  client.py      — httpx async client for SiliconFlow API (vision + embedding)
  metadata.py    — Metadata dataclass, JSON load/save, dedup by MD5 hash
  embedding.py   — field-wise build_text / build_image, search, save/load
  cli.py         — CLI entry point (thin wrapper around MemeVault)
  config.py      — default models, constants (no paths)
  Tests.py       — simple test
main.py          — thin wrapper for `python main.py` compatibility
pyproject.toml   — package metadata, deps, console_scripts entry point
data/            — metadata.json + embeddings.npy + embeddings_meta.json (gitignored)
images/          — meme image files
```

## Python API

```python
from meme_vault import MemeVault
import asyncio

async def main():
    vault = MemeVault(
        data_dir="./my_vault",
        api_key="sk-xxx",                              # optional, defaults to env var
        embedding_model="Qwen/Qwen3-VL-Embedding-8B",   # optional
        vision_model="Qwen/Qwen3-VL-32B-Instruct",      # optional
    )

    await vault.parse("path/to/image.jpg")          # analyze + add single image
    await vault.parse_dir("./images")               # batch import all images in dir
    count = await vault.build_text()                # text → vectors (field-wise)
    count = await vault.build_image()               # image → vectors (cross-modal)
    results = await vault.search("无语", top_k=5)   # [(Metadata, score), ...]
    entries = vault.list()                          # list[Metadata]
    info = vault.info()                             # dict
    st = vault.status()                             # dict (see status() below)
    await vault.close()

asyncio.run(main())
```

### `status()` return dict

```python
{
    "version": "2.0.0",
    "data_dir": "/path/to/data",
    "api_key": True,
    "embedding_model": "Qwen/Qwen3-VL-Embedding-8B",
    "vision_model": "Qwen/Qwen3-VL-32B-Instruct",
    "embedding_mode": "text" | "image" | None,
    "metadata": {"total": 98, "analyzed": 98, "paths_exist": 98, "paths_missing": 0, "models": {...}},
    "embeddings": {"shape": [98, 4096], "dtype": "float32", "up_to_date": True} | None,
    "images": {"directory": "/path/to/images", "files": 120, "unparsed": 22},
}
```

## CLI

```bash
pip install -e .
meme-vault --version          # show version
meme-vault status             # show working directory status
meme-vault build              # build text embeddings (field-wise)
meme-vault build-image        # build image embeddings
meme-vault search <text>      # search memes (auto-rebuilds if stale)
meme-vault parse <path>       # analyze a single image
meme-vault parse -r <dir>     # recursively import all images from dir
meme-vault parse -r -f <dir>  # force re-parse even if already imported (e.g. after model change)
meme-vault list               # list all memes
meme-vault info               # show project info
```

Search auto-detects the previously used embedding mode (text or image) and auto-rebuilds if the metadata has changed since the last build.

## Configuration

- `SILICONFLOW_API_KEY` env var required for API calls
- Default models in `config.py`: embedding `Qwen/Qwen3-VL-Embedding-8B`, vision `Qwen/Qwen3-VL-32B-Instruct`
