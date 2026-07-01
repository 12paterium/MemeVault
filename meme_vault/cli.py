import asyncio, sys, os
from .vault import MemeVault

HELP = """Commands:
  build              Build embeddings from text metadata (default)
  build-text         Same as build
  build-image       Build embeddings from images (cross-modal)
  search <text>     Search memes
  parse <path>      Analyze image and add to vault
  parse -r <dir>    Recursively import all images from directory
  parse -r -f <dir> Force re-parse even if already imported
  list              List all memes
  info              Show project info
  status            Show working directory status
  --version, -v     Show version number"""


async def main():
    if len(sys.argv) < 2:
        print(HELP); return
    if sys.argv[1] in ("--version", "-v"):
        from .vault import __version__
        print(f"meme-vault {__version__}")
        return
    cmd = sys.argv[1].lower()
    vault = MemeVault()
    try:
        if cmd in ("build", "build-text"):
            n = await vault.build_text()
            print(f"Done. {n} vectors ready.")
        elif cmd == "build-image":
            n = await vault.build_image()
            print(f"Done. {n} vectors ready.")
        elif cmd == "search" and len(sys.argv) >= 3:
            query = " ".join(sys.argv[2:])
            results = await vault.search(query)
            if not results:
                print("No results."); return
            print(f"\nTop {len(results)} for: {query}\n")
            for i, (m, score) in enumerate(results, 1):
                print(f"  {i}. [{score:.3f}] {m.text}")
                if m.tags: print(f"     tags: {', '.join(m.tags)}")
                if m.emotion: print(f"     mood: {', '.join(m.emotion)}")
                if m.background: print(f"     bg: {m.background}")
                print()
        elif cmd == "parse" and len(sys.argv) >= 3:
            if sys.argv[2] in ("-r", "--recursive") and len(sys.argv) >= 4:
                force = "-f" in sys.argv or "--force" in sys.argv
                n = await vault.parse_dir(sys.argv[3], force=force)
                print(f"Done. {n} new images imported.")
            else:
                text = await vault.parse(sys.argv[2])
                print(f"Added: {text}")
        elif cmd == "list":
            entries = vault.list()
            if not entries:
                print("No memes in vault.")
            else:
                for i, e in enumerate(entries, 1):
                    print(f"  {i}. {e.text}  ({e.path})")
        elif cmd == "info":
            info = vault.info()
            print(f"Data: {info['data_dir']}")
            print(f"Key:  {'OK' if info['has_key'] else 'Missing'}")
            print(f"Memes: {info['memes']}")
            if info["models"]:
                print("Models:")
                for m, n in sorted(info["models"].items(), key=lambda x: -x[1]):
                    print(f"  {m}: {n}")
        elif cmd == "status":
            s = vault.status()
            print(f"Version:         {s['version']}")
            print(f"Data directory:  {s['data_dir']}")
            print(f"API key:         {'OK' if s['api_key'] else 'Missing'}")
            print(f"Embedding model: {s['embedding_model']}")
            print(f"Vision model:    {s['vision_model']}")
            print(f"Embedding mode:  {s['embedding_mode'] or 'none (run build first)'}")
            print()
            m = s["metadata"]
            print(f"Metadata: {m['total']} entries ({m['analyzed']} analyzed)")
            print(f"  Paths:   {m['paths_exist']} exist, {m['paths_missing']} missing")
            if m["models"]:
                print(f"  Models:  {', '.join(f'{k}: {v}' for k, v in m['models'].items())}")
            print()
            e = s["embeddings"]
            if e:
                print(f"Embeddings: shape={e.get('shape')}, dtype={e.get('dtype')}")
                if e.get("up_to_date"):
                    print(f"  Status:   up to date")
                elif "error" in e:
                    print(f"  Status:   {e['error']}")
                else:
                    print(f"  Status:   needs rebuild ({m['total']} entries, {e['shape'][0]} vectors)")
            else:
                print("Embeddings: none")
            print()
            img = s["images"]
            if img["directory"]:
                print(f"Images directory: {img['directory']}")
                print(f"  Files:    {img['files']}")
                print(f"  Unparsed: {img['unparsed']}")
            else:
                print("Images directory: not found")
        else:
            print(HELP)
    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        pass
    finally:
        await vault.close()


if __name__ == "__main__":
    asyncio.run(main())


def cli():
    asyncio.run(main())