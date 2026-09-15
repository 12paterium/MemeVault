"""Import a large image collection in batches.

Splits <source> into N batches (sorted by filename) and parses one batch per run.
A multi-thousand-image import then survives rate limits, relay hiccups and reboots:
re-running a batch skips every image already parsed with the same vision model,
so batches are resumable and safe to repeat.

    python tools/import_batches.py --data-dir ./data --source D:/memes --batch 1
    python tools/import_batches.py --data-dir ./data --source D:/memes --batch 2 \
        --config <astrbot>/data/config/memeManager_config.json --summary batch2.json

Images are parsed with the vision model configured in `--config` (a plugin config
JSON, `sub_config` keys); without it the library defaults + SILICONFLOW_API_KEY are
used, exactly like the CLI. Pass `--build` to rebuild the search index right after the
batch (the plugin also rebuilds it incrementally on its next search).
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meme_vault import config as defaults
from meme_vault.metadata import Metadata
from meme_vault.vault import IMAGE_EXTS, MemeVault


def load_provider(config_path: Path | None) -> dict:
    if config_path is None:
        return {}
    with open(config_path, encoding="utf-8-sig") as file:
        config = json.load(file)
    sub = config.get("sub_config")
    if isinstance(sub, dict):
        return sub
    return next(
        (value for value in config.values() if isinstance(value, dict) and "visionModel" in value),
        {},
    )


def make_vault(data_dir: Path, sub: dict) -> MemeVault:
    return MemeVault(
        data_dir=str(data_dir),
        api_key=sub.get("apiKey"),
        embedding_model=sub.get("embeddingModel"),
        embedding_base_url=sub.get("embeddingApiBase"),
        vision_model=sub.get("visionModel"),
        vision_api_key=sub.get("visionApiKey") or sub.get("apiKey"),
        vision_base_url=sub.get("visionApiBase"),
    )


def batch_files(source: Path, batch: int, batches: int) -> tuple[int, list[str]]:
    files = sorted(
        name for name in os.listdir(source)
        if os.path.splitext(name)[1].lower() in IMAGE_EXTS
    )
    per = (len(files) + batches - 1) // batches
    start = (batch - 1) * per
    return len(files), files[start:start + per]


async def run(args) -> int:
    source = Path(args.source)
    if not source.is_dir():
        raise SystemExit(f"Source directory not found: {source}")
    total_files, chunk = batch_files(source, args.batch, args.batches)
    sub = load_provider(args.config)
    vault = make_vault(args.data_dir, sub)

    entries = vault._entries()
    id_index = {entry.id: index for index, entry in enumerate(entries) if entry.id}
    old_models = {entry.id: entry.analyzed_by for entry in entries if entry.id}

    pending = []
    for name in chunk:
        entry = Metadata(str(source / name))
        if not entry.id:
            print(f"  UNREADABLE: {name}")
            continue
        if entry.id in id_index and old_models.get(entry.id) == vault.vision_model:
            continue
        pending.append(entry)

    print(f"vault     : {vault.data_dir} ({len(entries)} entries)")
    print(f"vision    : {vault.vision_model} @ {vault.vision_base_url}")
    print(f"collection: {total_files} images, batch {args.batch}/{args.batches} = {len(chunk)} files")
    print(f"pending   : {len(pending)} "
          f"(already parsed by {vault.vision_model}: {len(chunk) - len(pending)})", flush=True)
    if args.dry_run:
        print("dry run: nothing parsed")
        return 0
    if not pending and not args.build:
        print("nothing to parse")
        return 0

    concurrency = max(1, args.concurrency or defaults.PARSE_CONCURRENCY)
    semaphore = asyncio.Semaphore(concurrency)
    client = vault._get_vision_client()
    started = time.time()
    done = 0

    async def analyze(entry):
        nonlocal done
        async with semaphore:
            result = await entry.analyze(client)
            done += 1
            if done % 10 == 0 or done == len(pending):
                rate = done / max(time.time() - started, 1e-6)
                print(f"  [{done}/{len(pending)}] {rate * 60:.1f}/min, "
                      f"~{(len(pending) - done) / rate / 60:.0f} min left", flush=True)
            return entry, result

    added = updated = failed = 0
    changed, failures = [], []
    for entry, error in await asyncio.gather(*(analyze(item) for item in pending)):
        if error:
            failed += 1
            failures.append({
                "file": os.path.basename(entry.path),
                "error": str(error.get("error", "?"))[:200],
            })
            continue
        if entry.id in id_index:
            entries[id_index[entry.id]] = entry
            updated += 1
        else:
            entries.append(entry)
            id_index[entry.id] = len(entries) - 1
            added += 1
        changed.append(entry)
    vault._save(entries, changed=changed)
    elapsed = time.time() - started
    indexed = None
    if args.build:
        # reload so the index is built in the canonical (filename-sorted) order the
        # vault uses when it reads the files back; otherwise its fingerprint would
        # mismatch and the host would rebuild on the next search
        vault._entries_cache = None
        print("rebuilding the index...", flush=True)
        started = time.time()
        indexed = await vault.build()
        print(f"index: {indexed} entries in {(time.time() - started) / 60:.1f} min", flush=True)
    await vault.close()
    summary = {
        "batch": args.batch,
        "batches": args.batches,
        "pending": len(pending),
        "added": added,
        "updated": updated,
        "failed": failed,
        "seconds": round(elapsed),
        "vault_total": len(entries),
        "indexed": indexed,
        "failures": failures,
    }
    print(f"\nadded={added} updated={updated} failed={failed} in {elapsed / 60:.1f} min")
    for item in failures:
        print(f"  FAILED: {item['file']} - {item['error']}")
    print(f"vault now holds {len(entries)} entries"
          + ("; index rebuilt" if indexed else "; run again with --build to refresh the index"))
    if args.summary:
        with open(args.summary, "w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)
        print(f"summary written to {args.summary}")
    return 0 if not failed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True, help="vault directory")
    parser.add_argument("--source", type=Path, required=True, help="directory with the images")
    parser.add_argument("--batch", type=int, required=True, help="which batch to parse (1-based)")
    parser.add_argument("--batches", type=int, default=4, help="how many batches to split into")
    parser.add_argument("--config", type=Path, default=None,
                        help="plugin config JSON; without it the library defaults + env are used")
    parser.add_argument("--concurrency", type=int, default=None,
                        help=f"parallel vision calls (default {defaults.PARSE_CONCURRENCY})")
    parser.add_argument("--summary", type=Path, default=None, help="write a JSON summary here")
    parser.add_argument("--build", action="store_true",
                        help="rebuild the search index right after the batch")
    parser.add_argument("--dry-run", action="store_true", help="report pending files and stop")
    args = parser.parse_args()
    if args.batch < 1 or args.batches < 1 or args.batch > args.batches:
        raise SystemExit("--batch must be within 1..--batches")
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
