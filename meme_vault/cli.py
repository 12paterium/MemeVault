import argparse
import asyncio

from .embedding import SearchQuery
from .vault import MemeVault, __version__


DEFAULT_DATA_DIR = "./data"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="meme-vault",
        description="Semantic meme search with character, usage, and content ranking.",
    )
    parser.add_argument("--version", "-v", action="version", version=f"meme-vault {__version__}")
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=f"Vault data directory (default: {DEFAULT_DATA_DIR})",
    )
    subparsers = parser.add_subparsers(dest="command")

    build = subparsers.add_parser("build", help="Build the three-dimension search index")
    build.add_argument("--force", action="store_true", help="Rebuild even when the index is current")

    search = subparsers.add_parser("search", help="Search memes")
    search.add_argument("query", nargs="*", help="Natural-language query")
    search.add_argument("--character", default="", help="Character or visual identity")
    search.add_argument("--usage", default="", help="Intended usage or chat situation")
    search.add_argument("--content", default="", help="Description, emotion, tags, or background")
    search.add_argument("--top-n", type=int, default=5, help="Number of results")
    search.add_argument("--rerank", action="store_true", help="Rerank candidates with the rerank model")
    search.add_argument("--character-weight", type=float)
    search.add_argument("--usage-weight", type=float)
    search.add_argument("--content-weight", type=float)

    parse = subparsers.add_parser("parse", help="Analyze images and update metadata")
    parse.add_argument("path")
    parse.add_argument("-r", "--recursive", action="store_true")
    parse.add_argument("-f", "--force", action="store_true")

    subparsers.add_parser("prune", help="Remove entries whose image files are missing")

    subparsers.add_parser("status", help="Show metadata, index, and image status")
    return parser


def _collect_weights(args) -> dict[str, float]:
    weights = {}
    for dimension in ("character", "usage", "content"):
        value = getattr(args, f"{dimension}_weight")
        if value is not None:
            weights[dimension] = value
    return weights


def _structured_query(args) -> SearchQuery | None:
    structured = bool(args.character or args.usage or args.content)
    if not structured:
        if _collect_weights(args):
            raise ValueError("Weight options require a structured query field.")
        return None
    if args.query:
        raise ValueError("Use either a natural-language query or structured query options, not both.")
    return SearchQuery(
        character=args.character,
        usage=args.usage,
        content=args.content,
    )


def _print_results(results):
    if not results:
        print("No results.")
        return
    for number, result in enumerate(results, 1):
        score = f"{result.score:.4f}"
        if result.rerank_score is not None:
            score += f" rerank={result.rerank_score:.4f}"
        print(f"  {number}. [{score}] {result.metadata.text}")
        for dimension in ("character", "usage", "content"):
            detail = result.dimensions.get(dimension)
            if detail is None:
                continue
            print(
                f"     {dimension}: similarity={detail.similarity:.4f}, "
                f"rank={detail.rank}, weight={detail.weight:g}, "
                f"contribution={detail.contribution:.4f}"
            )
        print(f"     path: {result.metadata.path}")
        print()


async def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0

    vault = MemeVault(data_dir=args.data_dir)
    try:
        if args.command == "build":
            count = await vault.build(force=args.force)
            print(f"Done. {count} entries indexed across 3 dimensions.")
        elif args.command == "search":
            query = _structured_query(args)
            if query is None:
                query = " ".join(args.query).strip()
            results = await vault.search(
                query,
                top_n=args.top_n,
                weights=_collect_weights(args) or None,
                rerank=args.rerank,
            )
            _print_results(results)
        elif args.command == "parse":
            if args.recursive:
                count = await vault.parse_dir(args.path, force=args.force)
                print(f"Done. {count} new images imported.")
            else:
                if args.force:
                    raise ValueError("--force is only valid with --recursive.")
                entry = await vault.parse(args.path)
                print(f"Added: {entry.text}")
        elif args.command == "prune":
            removed = vault.prune()
            print(f"Done. {len(removed)} entries removed.")
            for entry in removed:
                print(f"  - {entry.text}  ({entry.path})")
        elif args.command == "status":
            status = vault.status()
            metadata = status["metadata"]
            index = status["index"]
            images = status["images"]
            print(f"Version:         {status['version']}")
            print(f"Data directory:  {status['data_dir']}")
            print(f"API key:         {'OK' if status['api_key'] else 'Missing'}")
            print(f"Embedding model: {status['embedding_model']}")
            print(f"Vision model:    {status['vision_model']}")
            print(f"Chat model:      {status['chat_model']}")
            print(f"Rerank model:    {status['rerank_model']}")
            print()
            print(f"Metadata: {metadata['total']} entries ({metadata['analyzed']} analyzed)")
            print(f"  Paths:   {metadata['paths_exist']} exist, {metadata['paths_missing']} missing")
            if index:
                state = "up to date" if index["up_to_date"] else "needs rebuild"
                print(
                    f"Index: {index['count']} entries x {index['dimension_size']} dimensions "
                    f"({state})"
                )
                print(f"  Channels: {', '.join(index['dimensions'])}")
            else:
                print("Index: none (run build)")
            if images["directory"]:
                print(f"Images: {images['files']} files, {images['unparsed']} unparsed")
                print(f"  Directory: {images['directory']}")
            else:
                print("Images: directory not found")
        return 0
    except (TypeError, ValueError, RuntimeError) as error:
        print(f"Error: {error}")
        return 1
    finally:
        await vault.close()


def cli():
    raise SystemExit(asyncio.run(main()))


if __name__ == "__main__":
    cli()
