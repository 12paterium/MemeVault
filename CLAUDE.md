# CLAUDE.md

## Project Overview

MemeVault is a local semantic search system for meme images. Image analysis produces structured metadata; search uses three independently embedded channels and weighted reciprocal-rank fusion.

## Architecture

- Source data: one JSON per entry under `data/memes/` (`<image stem>.<id8>.json`) with text, tags, character, emotion, usage, and background; images are copied into `data/images/` and referenced by relative path, so the whole data dir is portable. A legacy single-file `data/metadata.json` is migrated on first load (kept as `metadata.json.bak`).
- Rebuildable index: `data/search_index.npz` plus `data/search_index_meta.json`.
- Scale (thousands of entries): each `MemeVault` instance caches entries (invalidated by `metadata.directory_stamp`: count + newest mtime + dir mtime) and the loaded index (invalidated by index file mtime/size). Saves write only the entries passed as `changed` (`save_metadata(..., changed=[...])`), and index rebuilds reuse rows whose (id, text-hash) match the previous index — only changed rows are embedded. `parse_dir` parses concurrently (`config.PARSE_CONCURRENCY`). Index format v2 stores per-row `ids` and per-channel text hashes in the npz.
- Search channels: `character` (weight 3), `usage` (weight 3), and `content` (weight 1).
- `content` combines description, tags, emotion, background, and filename. Character and usage are not duplicated into it.
- Image analysis (`AIClient.parse_image(image_path, filename)`) sends the original filename (id suffix stripped) as a hint alongside the image. The prompt scopes it to context the picture cannot show ("早八", "看手机") and forbids inferring characters or sources from it: measured against `gpt-5.6-sol`, an unscoped hint made the model name the wrong source (《搞笑一家人》/李顺才 for a 第五共和国 still) and stop describing appearance; with the scoped wording it identifies the source correctly and still uses the name's context.
- Natural-language queries are decomposed with the configured chat model (`chat_*` args; follows `vision_*` unless set). `SearchQuery` bypasses decomposition.
- Each active channel performs cosine ranking; weighted RRF produces the final Top N.
- `AIClient._post` retries statuses in `config.RETRY_STATUSES` **and network exceptions** (connect errors, timeouts) `config.RETRY_COUNT` times, honouring the `Retry-After` header, otherwise backing off 2/4/8s. The embedding client uses a 300s timeout (`config.EMBEDDING_TIMEOUT`) because a slow provider can take >90s for one 64-text batch; vision/chat/rerank keep 120s.
- Large collections are imported in batches with `tools/import_batches.py` (resumable: images already parsed by the same vision model are skipped; `--dry-run` counts pending files).
- Each role's client can be injected (`MemeVault(chat_client=..., vision_client=..., embed_client=..., rerank_client=...)`) so a host framework can supply its own model access. Injected clients must expose `.model` / `chat` / `parse_image` / `embed` / `rerank` / `close`; they belong to the caller and `close()` skips them. `client.parse_json_object(text)` parses JSON out of a reply that may be fenced or wrapped in prose, for clients that cannot request structured output.

## Code Structure

```text
meme_vault/
  vault.py       - MemeVault orchestration and index freshness
  embedding.py   - query/result types, index build, cosine ranking, RRF
  client.py      - SiliconFlow chat, vision, and embedding client
  metadata.py    - metadata model and JSON persistence
  cli.py         - command-line interface
  config.py      - models, weights, and retrieval constants
tools/
  import_batches.py - resumable batched import for large collections
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
