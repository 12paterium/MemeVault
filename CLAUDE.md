# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MemeVault — a semantic vector search system for meme images. Users type a natural language query and get the most relevant memes via cosine similarity on text embeddings.

## Architecture (3-layer decoupled)

- **Data Layer** — `data/metadata.json`: annotated meme metadata (path, text, tags, character, emotion, usage, background). The "source of truth."
- **Feature Layer** — `data/embeddings.npy`: cached text embeddings. Never committed; can be rebuilt at any time.
- **Retrieval Layer** — numpy cosine similarity search over embeddings.

Key design rules: data !== features !== index. Embeddings are rebuildable caches, not assets. IDs are MD5 hashes of file content (not tied to filenames).

## Code Structure

```
meme_vault/
  vault.py       — MemeVault class: primary public API
  client.py      — httpx async client for SiliconFlow API (vision + embedding)
  metadata.py    — Metadata dataclass, JSON load/save, dedup by MD5 hash
  embedding.py   — build_text/build_image embeddings, search, save, load
  cli.py         — CLI entry point (thin wrapper around MemeVault)
  config.py      — default models, constants (no paths)
  Tests.py       — simple test
main.py          — thin wrapper for `python main.py` compatibility
pyproject.toml   — package metadata, deps, console_scripts entry point
data/            — metadata.json + embeddings.npy (gitignored)
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

    await vault.parse("path/to/image.jpg")          # analyze + add
    count = await vault.build_text()                # text → vectors
    count = await vault.build_image()               # image → vectors
    results = await vault.search("无语", top_k=5)   # [(Metadata, score), ...]
    entries = vault.list()                          # list[Metadata]
    info = vault.info()                             # dict
    await vault.close()

asyncio.run(main())
```

## CLI

```bash
pip install -e .
meme-vault build-text|build-image|search|parse|list|info
```

## Configuration

- `SILICONFLOW_API_KEY` env var required for API calls
- Default models in `config.py`: embedding `Qwen/Qwen3-VL-Embedding-8B`, vision `Qwen/Qwen3-VL-32B-Instruct`
