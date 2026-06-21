import asyncio, os, numpy as np
from . import config as defaults
from .client import AIClient
from .metadata import Metadata, load_metadata, save_metadata
from .embedding import (
    build_text_embeddings, build_image_embeddings,
    save_embeddings, load_embeddings, search,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}


def _count_images(root: str) -> int:
    """Count image files under a directory."""
    if not os.path.isdir(root):
        return 0
    total = 0
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                total += 1
    return total


class MemeVault:
    def __init__(self, data_dir="./data", api_key=None,
                 embedding_model=None, vision_model=None):
        self.data_dir = os.path.abspath(data_dir)
        self.metadata_path = os.path.join(self.data_dir, "metadata.json")
        self.embeddings_path = os.path.join(self.data_dir, "embeddings.npy")
        self.api_key = api_key or defaults.API_KEY
        self.embedding_model = embedding_model or defaults.EMBEDDING_MODEL
        self.vision_model = vision_model or defaults.VISION_MODEL
        self._embed_client = None
        self._vision_client = None

    def _get_embed_client(self):
        if self._embed_client is None:
            self._embed_client = AIClient(
                model=self.embedding_model, api_key=self.api_key)
        return self._embed_client

    def _get_vision_client(self):
        if self._vision_client is None:
            self._vision_client = AIClient(
                model=self.vision_model, api_key=self.api_key)
        return self._vision_client

    async def parse(self, image_path: str) -> str:
        entry = Metadata(image_path)
        if not entry.id:
            raise ValueError(f"Cannot read file: {image_path}")
        client = self._get_vision_client()
        error = await entry.analyze(client)
        if error:
            raise RuntimeError(f"Vision API error: {error.get('error', '?')}")
        entries = load_metadata(self.metadata_path)
        for i, e in enumerate(entries):
            if e.id == entry.id:
                entries[i] = entry
                save_metadata(entries, self.metadata_path)
                return entry.text
        entries.append(entry)
        save_metadata(entries, self.metadata_path)
        return entry.text

    async def parse_dir(self, root_dir: str) -> int:
        if not os.path.isdir(root_dir):
            raise ValueError(f"Directory not found: {root_dir}")
        entries = load_metadata(self.metadata_path)
        existing = {e.id for e in entries if e.id}
        added = 0
        client = self._get_vision_client()
        for dirpath, _, filenames in os.walk(root_dir):
            for f in filenames:
                if os.path.splitext(f)[1].lower() not in IMAGE_EXTS:
                    continue
                fp = os.path.join(dirpath, f)
                entry = Metadata(fp)
                if not entry.id or entry.id in existing:
                    continue
                error = await entry.analyze(client)
                if error:
                    print(f"  SKIP: {fp} — {error.get('error', '?')}")
                    continue
                entries.append(entry)
                existing.add(entry.id)
                added += 1
                print(f"  [{added}] {entry.text}")
        save_metadata(entries, self.metadata_path)
        return added

    async def build(self) -> int:
        return await self.build_text()

    async def build_text(self) -> int:
        entries = load_metadata(self.metadata_path)
        if not entries:
            raise ValueError("No entries. Add some with parse() first.")
        client = self._get_embed_client()
        old = self._load_existing_embeddings()
        if old is not None and old.shape[0] == len(entries):
            return old.shape[0]  # nothing new
        if old is not None and old.shape[0] < len(entries):
            new_entries = entries[old.shape[0]:]
            new_vecs = await build_text_embeddings(client, new_entries)
            embeddings = np.concatenate([old, new_vecs])
        else:
            embeddings = await build_text_embeddings(client, entries)
        save_embeddings(embeddings, self.embeddings_path)
        return embeddings.shape[0]

    async def build_image(self) -> int:
        entries = load_metadata(self.metadata_path)
        if not entries:
            raise ValueError("No entries. Add some with parse() first.")
        client = self._get_embed_client()
        old = self._load_existing_embeddings()
        if old is not None and old.shape[0] == len(entries):
            return old.shape[0]
        if old is not None and old.shape[0] < len(entries):
            new_entries = entries[old.shape[0]:]
            new_vecs = await build_image_embeddings(client, new_entries)
            embeddings = np.concatenate([old, new_vecs])
        else:
            embeddings = await build_image_embeddings(client, entries)
        save_embeddings(embeddings, self.embeddings_path)
        return embeddings.shape[0]

    def _load_existing_embeddings(self):
        try:
            return load_embeddings(self.embeddings_path)
        except (FileNotFoundError, RuntimeError):
            return None

    async def search(self, query: str, top_k: int = 5) -> list:
        entries = load_metadata(self.metadata_path)
        if not entries:
            return []
        try:
            embeddings = load_embeddings(self.embeddings_path)
        except (FileNotFoundError, RuntimeError):
            raise RuntimeError("No embeddings. Run build_text() or build_image() first.")
        if len(entries) != embeddings.shape[0]:
            raise RuntimeError(
                f"Entry/embedding mismatch: {len(entries)} entries vs "
                f"{embeddings.shape[0]} embeddings. Rebuild.")
        client = self._get_embed_client()
        return await search(client, query, entries, embeddings, top_k)

    def list(self) -> list:
        return load_metadata(self.metadata_path)

    def info(self) -> dict:
        entries = load_metadata(self.metadata_path)
        models = {}
        for e in entries:
            m = e.analyzed_by or "unknown"
            models[m] = models.get(m, 0) + 1
        return {
            "data_dir": self.data_dir,
            "memes": len(entries),
            "has_key": bool(self.api_key),
            "embedding_model": self.embedding_model,
            "vision_model": self.vision_model,
            "models": models,
        }

    def status(self) -> dict:
        entries = load_metadata(self.metadata_path)
        analyzed = sum(1 for e in entries if e.analyzed_at)
        paths_exist = sum(1 for e in entries if e.path and os.path.exists(e.path))
        paths_missing = len(entries) - paths_exist
        models = {}
        for e in entries:
            m = e.analyzed_by or "unknown"
            models[m] = models.get(m, 0) + 1

        emb_info = None
        emb_path = self.embeddings_path
        if os.path.exists(emb_path):
            try:
                emb = load_embeddings(emb_path)
                emb_info = {"shape": list(emb.shape), "dtype": str(emb.dtype)}
                emb_info["up_to_date"] = emb.shape[0] == len(entries)
            except Exception:
                emb_info = {"error": "corrupt or unreadable"}

        image_root = os.path.join(os.path.dirname(self.data_dir), "images")
        image_count = _count_images(image_root)

        # find unparsed images
        import hashlib
        parsed_ids = {e.id for e in entries if e.id}
        unparsed = 0
        if os.path.isdir(image_root):
            for dirpath, _, filenames in os.walk(image_root):
                for f in filenames:
                    if os.path.splitext(f)[1].lower() not in IMAGE_EXTS:
                        continue
                    fp = os.path.join(dirpath, f)
                    try:
                        with open(fp, "rb") as fh:
                            h = hashlib.md5(fh.read()).hexdigest()
                        if h not in parsed_ids:
                            unparsed += 1
                    except OSError:
                        pass

        return {
            "data_dir": self.data_dir,
            "api_key": bool(self.api_key),
            "embedding_model": self.embedding_model,
            "vision_model": self.vision_model,
            "metadata": {
                "total": len(entries),
                "analyzed": analyzed,
                "paths_exist": paths_exist,
                "paths_missing": paths_missing,
                "models": models,
            },
            "embeddings": emb_info,
            "images": {
                "directory": image_root if os.path.isdir(image_root) else None,
                "files": image_count,
                "unparsed": unparsed,
            },
        }

    async def close(self):
        for c in (self._embed_client, self._vision_client):
            if c:
                await c.close()
        self._embed_client = None
        self._vision_client = None