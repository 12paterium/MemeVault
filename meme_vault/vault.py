from __future__ import annotations

import hashlib
import os
from dataclasses import replace

from . import config as defaults
from .client import AIClient
from .embedding import (
    DIMENSIONS,
    INDEX_VERSION,
    SearchQuery,
    SearchResult,
    build_search_index,
    index_manifest_path,
    load_search_index,
    metadata_fingerprint,
    parse_search_query,
    rerank_text,
    save_search_index,
    search as rank_search,
)
from .metadata import Metadata, load_metadata, save_metadata


__version__ = "3.1.0"


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}


def _iter_images(root: str):
    if not os.path.isdir(root):
        return
    for directory, _, filenames in os.walk(root):
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() in IMAGE_EXTS:
                yield os.path.abspath(os.path.join(directory, filename))


def _file_md5(path: str) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class MemeVault:
    def __init__(
        self,
        data_dir="./data",
        api_key=None,
        embedding_model=None,
        vision_model=None,
        embedding_base_url=None,
        vision_api_key=None,
        vision_base_url=None,
        rerank_model=None,
        rerank_api_key=None,
        rerank_api_base=None,
    ):
        self.data_dir = os.path.abspath(data_dir)
        self.metadata_path = os.path.join(self.data_dir, "metadata.json")
        self.index_path = os.path.join(self.data_dir, "search_index.npz")
        self.api_key = api_key or defaults.API_KEY
        self.embedding_model = embedding_model or defaults.EMBEDDING_MODEL
        self.vision_model = vision_model or defaults.VISION_MODEL
        self.embedding_base_url = embedding_base_url or defaults.API_BASE
        self.vision_api_key = vision_api_key or self.api_key
        self.vision_base_url = vision_base_url or defaults.API_BASE
        self.rerank_model = rerank_model or defaults.RERANK_MODEL
        self.rerank_api_key = rerank_api_key or self.api_key
        self.rerank_api_base = rerank_api_base or defaults.API_BASE
        self._embed_client = None
        self._vision_client = None
        self._rerank_client = None

    async def __aenter__(self) -> "MemeVault":
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    def _get_embed_client(self):
        if self._embed_client is None:
            self._embed_client = AIClient(
                model=self.embedding_model,
                api_key=self.api_key,
                base_url=self.embedding_base_url,
            )
        return self._embed_client

    def _get_vision_client(self):
        if self._vision_client is None:
            self._vision_client = AIClient(
                model=self.vision_model,
                api_key=self.vision_api_key,
                base_url=self.vision_base_url,
            )
        return self._vision_client

    def _get_rerank_client(self):
        if self._rerank_client is None:
            self._rerank_client = AIClient(
                model=self.rerank_model,
                api_key=self.rerank_api_key,
                base_url=self.rerank_api_base,
            )
        return self._rerank_client

    @staticmethod
    def _count_models(entries: list[Metadata]) -> dict[str, int]:
        models = {}
        for entry in entries:
            model = entry.analyzed_by or "unknown"
            models[model] = models.get(model, 0) + 1
        return models

    async def parse(self, image_path: str) -> Metadata:
        """Analyze one image and store or replace its metadata entry."""
        entry = Metadata(image_path)
        if not entry.id:
            raise ValueError(f"Cannot read file: {image_path}")
        error = await entry.analyze(self._get_vision_client())
        if error:
            raise RuntimeError(f"Vision API error: {error.get('error', '?')}")

        entries = load_metadata(self.metadata_path)
        for index, existing in enumerate(entries):
            if existing.id == entry.id:
                entries[index] = entry
                save_metadata(entries, self.metadata_path)
                return entry
        entries.append(entry)
        save_metadata(entries, self.metadata_path)
        return entry

    async def parse_dir(self, root_dir: str, force: bool = False) -> int:
        if not os.path.isdir(root_dir):
            raise ValueError(f"Directory not found: {root_dir}")
        entries = load_metadata(self.metadata_path)
        id_index = {entry.id: index for index, entry in enumerate(entries) if entry.id}
        old_models = {entry.id: entry.analyzed_by for entry in entries if entry.id}
        added = 0
        client = self._get_vision_client()

        for image_path in _iter_images(root_dir):
            entry = Metadata(image_path)
            if not entry.id:
                continue
            if (
                not force
                and entry.id in id_index
                and old_models.get(entry.id) == self.vision_model
            ):
                continue
            error = await entry.analyze(client)
            if error:
                print(f"  SKIP: {image_path} - {error.get('error', '?')}")
                continue
            if entry.id in id_index:
                entries[id_index[entry.id]] = entry
                print(f"  [UPDATE] {entry.text}")
            else:
                entries.append(entry)
                id_index[entry.id] = len(entries) - 1
                added += 1
                print(f"  [{added}] {entry.text}")
        save_metadata(entries, self.metadata_path)
        return added

    def _load_current_index(self, entries: list[Metadata]):
        try:
            index = load_search_index(self.index_path)
        except RuntimeError:
            return None
        manifest = index.manifest
        if (
            manifest.get("version") != INDEX_VERSION
            or manifest.get("model") != self.embedding_model
            or manifest.get("count") != len(entries)
            or manifest.get("dimensions") != list(DIMENSIONS)
            or manifest.get("fingerprint") != metadata_fingerprint(entries)
        ):
            return None
        return index

    async def build(self, force: bool = False) -> int:
        entries = load_metadata(self.metadata_path)
        if not entries:
            raise ValueError("No entries. Add some with parse() first.")
        if not force and self._load_current_index(entries) is not None:
            return len(entries)

        index = await build_search_index(self._get_embed_client(), entries)
        save_search_index(
            index,
            self.index_path,
            model=self.embedding_model,
            fingerprint=metadata_fingerprint(entries),
        )
        return index.count

    async def _ensure_index(self, entries: list[Metadata]):
        index = self._load_current_index(entries)
        if index is not None:
            return index
        await self.build(force=True)
        return load_search_index(self.index_path)

    async def search(
        self,
        query: str | SearchQuery,
        top_n: int = defaults.TOP_N,
        weights: dict[str, float] | None = None,
        rerank: bool = False,
    ) -> list[SearchResult]:
        if not isinstance(query, (str, SearchQuery)):
            raise TypeError("query must be a string or SearchQuery.")
        if isinstance(query, str) and not query.strip():
            raise ValueError("Query cannot be empty.")
        if weights is not None and not isinstance(weights, dict):
            raise TypeError("weights must be a dict mapping dimensions to numbers.")

        entries = load_metadata(self.metadata_path)
        if not entries:
            return []
        index = await self._ensure_index(entries)
        if isinstance(query, str):
            parsed = await parse_search_query(self._get_vision_client(), query)
        else:
            parsed = query
        candidates = max(top_n, defaults.RERANK_CANDIDATES) if rerank else top_n
        results = await rank_search(
            self._get_embed_client(),
            parsed,
            entries,
            index,
            candidates,
            weights=weights,
        )
        if rerank and results:
            results = await self._apply_rerank(query, results)
        return results[:min(top_n, len(results))]

    async def _apply_rerank(
        self,
        query: str | SearchQuery,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        """Reorder candidates with the cross-encoder, keeping RRF order if reranking fails."""
        text = query if isinstance(query, str) else " ".join(
            part for part in (query.character, query.usage, query.content) if part
        )
        if not text.strip():
            return results
        documents = [rerank_text(result.metadata) for result in results]
        try:
            ranked = await self._get_rerank_client().rerank(text, documents, top_n=len(documents))
        except Exception as error:
            print(f"  RERANK SKIP: {error}")
            return results
        scores = {index: score for index, score in ranked}
        order = sorted(
            range(len(results)),
            key=lambda position: (-scores.get(position, float("-inf")), position),
        )
        return [
            replace(results[position], rerank_score=scores.get(position))
            for position in order
        ]

    def prune(self) -> list[Metadata]:
        """Remove entries whose image files no longer exist on disk."""
        entries = load_metadata(self.metadata_path)
        kept, removed = [], []
        for entry in entries:
            if entry.path and os.path.exists(entry.path):
                kept.append(entry)
            else:
                removed.append(entry)
        if removed:
            save_metadata(kept, self.metadata_path)
        return removed

    def _find_image_root(self) -> str | None:
        project_root = os.path.dirname(self.data_dir)
        candidates = (
            os.path.join(project_root, "ResourceImages"),
            os.path.join(project_root, "images"),
        )
        return next((path for path in candidates if os.path.isdir(path)), None)

    def status(self) -> dict:
        entries = load_metadata(self.metadata_path)
        analyzed = sum(1 for entry in entries if entry.analyzed_at)
        paths_exist = sum(1 for entry in entries if entry.path and os.path.exists(entry.path))

        index_info = None
        try:
            index = load_search_index(self.index_path)
            manifest = index.manifest
            index_info = {
                "version": manifest.get("version"),
                "model": manifest.get("model"),
                "count": index.count,
                "dimension_size": manifest.get("dimension_size"),
                "dimensions": list(DIMENSIONS),
                "up_to_date": self._load_current_index(entries) is not None,
            }
        except RuntimeError:
            pass

        image_root = self._find_image_root()
        image_paths = list(_iter_images(image_root)) if image_root else []
        parsed_ids = {entry.id for entry in entries if entry.id}
        unparsed = 0
        for image_path in image_paths:
            try:
                if _file_md5(image_path) not in parsed_ids:
                    unparsed += 1
            except OSError:
                unparsed += 1

        return {
            "version": __version__,
            "data_dir": self.data_dir,
            "api_key": bool(self.api_key),
            "embedding_model": self.embedding_model,
            "vision_model": self.vision_model,
            "rerank_model": self.rerank_model,
            "metadata": {
                "total": len(entries),
                "analyzed": analyzed,
                "paths_exist": paths_exist,
                "paths_missing": len(entries) - paths_exist,
                "models": self._count_models(entries),
            },
            "index": index_info,
            "images": {
                "directory": image_root,
                "files": len(image_paths),
                "unparsed": unparsed,
            },
        }

    async def close(self):
        clients = {
            id(client): client
            for client in (self._embed_client, self._vision_client, self._rerank_client)
            if client
        }
        for client in clients.values():
            await client.close()
        self._embed_client = None
        self._vision_client = None
        self._rerank_client = None
