# CLAUDE.md

## Project Overview

MemeVault is a local semantic search system for meme images. Image analysis produces structured metadata; search uses three independently embedded channels and weighted reciprocal-rank fusion.

## Architecture

- Source data: `data/metadata.json` with text, tags, character, emotion, usage, and background.
- Rebuildable index: `data/search_index.npz` plus `data/search_index_meta.json`.
- Search channels: `character` (weight 3), `usage` (weight 3), and `content` (weight 1).
- `content` combines description, tags, emotion, background, and filename. Character and usage are not duplicated into it.
- Natural-language queries are decomposed with the configured vision/chat model. `SearchQuery` bypasses decomposition.
- Each active channel performs cosine ranking; weighted RRF produces the final Top N.

## Code Structure

```text
meme_vault/
  vault.py       - MemeVault orchestration and index freshness
  embedding.py   - query/result types, index build, cosine ranking, RRF
  client.py      - SiliconFlow chat, vision, and embedding client
  metadata.py    - metadata model and JSON persistence
  cli.py         - command-line interface
  config.py      - models, weights, and retrieval constants
tests/           - deterministic unit and integration tests
ResourceImages/  - optional local image test set
```

## Public API

```python
from meme_vault import MemeVault, SearchQuery

async with MemeVault() as vault:
    count = await vault.build()
    results = await vault.search("natural language", top_n=5)
    results = await vault.search(
        SearchQuery(character="初音未来", usage="吐槽", content="无语"),
        top_n=5,
        weights={"character": 3, "usage": 3, "content": 1},
    )
```

`SearchQuery` carries only channel content (character, usage, content); ranking weights are `search()` parameters. `search()` returns `list[SearchResult]`. Each result exposes `metadata`, normalized RRF `score`, and per-channel `dimensions` details. `MemeVault` is an async context manager and also exposes `parse` (returns `Metadata`), `parse_dir`, `prune`, and `status`.

## Commands

```bash
meme-vault build [--force]
meme-vault search <text> [--top-n 5] [--rerank]
meme-vault search --character <text> --usage <text> --content <text>
meme-vault parse <path>
meme-vault parse -r [-f] <directory>
meme-vault prune
meme-vault status
```

`--data-dir` (before the subcommand) selects the vault directory; default `./data`.

Run tests with `python -m unittest discover -s tests -v`. Tests must not require API credentials or network access.
