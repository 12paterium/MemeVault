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
  AIClient.py    — httpx-based async client for SiliconFlow API (vision + embedding)
  Metadata.py    — Metadata dataclass, JSON load/save, dedup by MD5 hash
  Embedding.py   — build/search embeddings, prepare_texts() concatenates fields for embedding
  config.py      — paths, model names, rate limits, batch size
  Tests.py       — simple test that loads and prints metadata
main.py          — CLI entry point (build, search, parse, batch, list, info)
data/            — metadata.json + embeddings.npy (gitignored)
images/          — meme image files
```

## Commands

```powershell
# Analyze a single image and add to vault
python main.py parse <image-path>

# Batch import all images from a directory
python main.py batch [dir]

# Build embeddings from metadata (requires SILICONFLOW_API_KEY)
python main.py build

# Search memes by natural language query
python main.py search <query text>

# List all memes
python main.py list

# Show project info (key status, meme count, models used)
python main.py info

# Run simple metadata test
python meme_vault/Tests.py
```

## Configuration

- `SILICONFLOW_API_KEY` env var is required for all API-dependent commands (parse, build, search, batch)
- Models are configured in `meme_vault/config.py`: embedding (`BAAI/bge-large-zh-v1.5`) and vision (`Qwen/Qwen3-VL-32B-Instruct`)
- Rate limiting: 1000 RPM / 80000 TPM via sliding window in `AIClient.RateLimiter`
