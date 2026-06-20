import asyncio, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from meme_vault import Metadata, config, load_metadata, save_metadata
from meme_vault.Embedding import build_embeddings, save_embeddings, load_embeddings, search, embedding_client

HELP = """Commands:
  build              Build embeddings from metadata
  search <text>      Search memes
  parse <path>       Analyze image and add to vault
  batch [dir]        Batch import all images from directory (default: public)
  list               List all memes
  info               Show project info"""

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}


async def cmd_build():
    entries = load_metadata()
    if not entries:
        print("No entries. Add some with 'parse' first."); return
    if not config.API_KEY:
        print("Set SILICONFLOW_API_KEY first."); return
    print(f"Building {len(entries)} embeddings...")
    embeddings = await build_embeddings(embedding_client, entries)
    save_embeddings(embeddings)
    print(f"Done. {embeddings.shape[0]} vectors ready.")


async def cmd_search(query: str):
    if not query.strip():
        print("Empty query."); return
    entries = load_metadata()
    if not entries:
        print("No memes in vault."); return
    try:
        embeddings = load_embeddings()
    except (FileNotFoundError, RuntimeError) as e:
        print(e); return
    if not config.API_KEY:
        print("Set SILICONFLOW_API_KEY first."); return
    results = await search(query, entries, embeddings)
    if not results:
        print("No results."); return
    print(f"\nTop {len(results)} for: {query}\n")
    for i, (m, score) in enumerate(results, 1):
        print(f"  {i}. [{score:.3f}] {m.text}")
        if m.tags: print(f"     tags: {', '.join(m.tags)}")
        if m.emotion: print(f"     mood: {', '.join(m.emotion)}")
        if m.background: print(f"     bg: {m.background}")
        print()


async def cmd_parse(image_path: str):
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}"); return
    if not config.API_KEY:
        print("Set SILICONFLOW_API_KEY first."); return
    entry = Metadata(image_path)
    if not entry.id:
        print(f"Cannot read file: {image_path}"); return
    error = await entry.analyze()
    if error:
        print(f"Vision API error: {error.get('error', '?')}"); return
    entries = load_metadata()
    # dedup by MD5 hash — if same image (moved/renamed), update in place; discard the old entry
    for i, e in enumerate(entries):
        if e.id == entry.id:
            entries[i] = entry
            print(f"Updated: {entry.text}")
            save_metadata(entries)
            return
    entries.append(entry)
    save_metadata(entries)
    print(f"Added: {entry.text}")


async def cmd_batch(root_dir: str):
    if not os.path.isdir(root_dir):
        print(f"Directory not found: {root_dir}"); return
    if not config.API_KEY:
        print("Set SILICONFLOW_API_KEY first."); return

    entries = load_metadata()
    existing = {e.id: i for i, e in enumerate(entries) if e.id}

    # collect image files
    files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                files.append(os.path.join(dirpath, f))
    files.sort()
    print(f"Found {len(files)} images in {root_dir}")

    # pre-filter: update paths for existing (file may have been moved/renamed), collect truly new for analysis
    to_analyze = []
    path_updates = 0
    for fp in files:
        m = Metadata(fp)
        if not m.id:
            continue
        idx = existing.get(m.id)
        if idx is not None:
            # same content, possibly different path — keep metadata, update path only (no API call)
            if entries[idx].path != m.path:
                entries[idx].path = m.path
                path_updates += 1
        else:
            to_analyze.append(fp)
    if path_updates:
        save_metadata(entries)
    print(f"New: {len(to_analyze)}, path-updated: {path_updates}, skipped: {len(files) - len(to_analyze) - path_updates}")
    if not to_analyze:
        print("Nothing to import."); return

    sem = asyncio.Semaphore(5)  # limit concurrent API calls to avoid flooding
    added, errors, save_counter = 0, 0, 0

    async def process_one(path: str):
        nonlocal added, errors, save_counter
        async with sem:
            entry = Metadata(path)
            if not entry.id:
                errors += 1; return
            err = await entry.analyze()
            if err:
                print(f"  ERROR: {path} — {err.get('error', '?')}")
                errors += 1; return
            entries.append(entry)
            added += 1
            save_counter += 1
            print(f"  [{added}/{len(to_analyze)}] {entry.text}")
            if save_counter % 50 == 0:
                save_metadata(entries)
                print(f"  --- auto-saved at {added} ---")

    await asyncio.gather(*[process_one(fp) for fp in to_analyze])
    save_metadata(entries)
    print(f"\nDone. +{added} added, {path_updates} path-updated, {errors} errors. Total: {len(entries)} memes.")


def cmd_list():
    entries = load_metadata()
    if not entries:
        print("No memes in vault."); return
    for i, e in enumerate(entries, 1):
        print(f"  {i}. {e.text}  ({e.path})")


async def main():
    if len(sys.argv) < 2:
        print(HELP); return
    cmd = sys.argv[1].lower()
    try:
        if cmd == "build": await cmd_build()
        elif cmd == "search" and len(sys.argv) >= 3: await cmd_search(" ".join(sys.argv[2:]))
        elif cmd == "parse" and len(sys.argv) >= 3: await cmd_parse(sys.argv[2])
        elif cmd == "batch": await cmd_batch(sys.argv[2] if len(sys.argv) >= 3 else config.PUBLIC_DIR)
        elif cmd == "list": cmd_list()
        elif cmd == "info":
            entries = load_metadata()
            models = {}
            for e in entries:
                m = e.analyzed_by or "unknown"
                models[m] = models.get(m, 0) + 1
            print(f"Data: {config.DATA_DIR}")
            print(f"Key:  {'OK' if config.API_KEY else 'Missing'}")
            print(f"Memes: {len(entries)}")
            if models:
                print("Models:")
                for m, n in sorted(models.items(), key=lambda x: -x[1]):
                    print(f"  {m}: {n}")
        else: print(HELP)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    asyncio.run(main())
