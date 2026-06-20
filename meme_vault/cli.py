import asyncio, sys, os
from .vault import MemeVault

HELP = """Commands:
  build              Build embeddings from text metadata (default)
  build-text         Same as build
  build-image       Build embeddings from images (cross-modal)
  search <text>     Search memes
  parse <path>      Analyze image and add to vault
  list              List all memes
  info              Show project info"""


async def main():
    if len(sys.argv) < 2:
        print(HELP); return
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