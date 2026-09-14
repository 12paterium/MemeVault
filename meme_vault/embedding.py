import asyncio
import hashlib
import json
import math
import os
import zipfile
from dataclasses import dataclass, field

import numpy as np

from . import config
from .client import AIClient
from .metadata import Metadata


DIMENSIONS = ("character", "usage", "content")
DEFAULT_WEIGHTS = {
    "character": config.CHARACTER_WEIGHT,
    "usage": config.USAGE_WEIGHT,
    "content": config.CONTENT_WEIGHT,
}
INDEX_VERSION = 1

QUERY_SCHEMA = {
    "type": "object",
    "properties": {
        "content": {"type": "string"},
        "character": {"type": "string"},
        "usage": {"type": "string"},
    },
    "required": list(DIMENSIONS),
    "additionalProperties": False,
}
QUERY_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "meme_search_query", "schema": QUERY_SCHEMA},
}
QUERY_PROMPT = """将表情包搜索请求拆成三个字段，只输出 JSON：
- character: 用户想找的人物、角色或外观；没有则为空字符串
- usage: 表情包的使用目的或聊天场景，如吐槽、怼人、回应、庆祝；没有则为空字符串
- content: 除人物和用途外的画面、文字、情绪、标签或梗背景；没有则为空字符串
不要在多个字段中重复同一个条件，不要补充用户没有表达的限制。"""


@dataclass(slots=True)
class SearchQuery:
    character: str = ""
    usage: str = ""
    content: str = ""

    def __post_init__(self):
        self.character = _clean_query_value(self.character)
        self.usage = _clean_query_value(self.usage)
        self.content = _clean_query_value(self.content)


@dataclass(frozen=True, slots=True)
class DimensionScore:
    similarity: float
    rank: int
    weight: float
    contribution: float


@dataclass(frozen=True, slots=True)
class SearchResult:
    metadata: Metadata
    score: float
    dimensions: dict[str, DimensionScore] = field(default_factory=dict)
    rerank_score: float | None = None


@dataclass(slots=True)
class SearchIndex:
    vectors: dict[str, np.ndarray]
    masks: dict[str, np.ndarray]
    manifest: dict = field(default_factory=dict)

    @property
    def count(self) -> int:
        if not self.vectors:
            return 0
        return next(iter(self.vectors.values())).shape[0]


def _clean_query_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return " ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def _join_values(values) -> str:
    if isinstance(values, str):
        return values.strip()
    return " ".join(str(value).strip() for value in values or [] if str(value).strip())


def _content_text(entry: Metadata) -> str:
    parts = []
    filename = os.path.splitext(os.path.basename(entry.path))[0].strip()
    if filename and filename != entry.text:
        parts.append(f"filename: {filename}")
    if entry.text:
        parts.append(f"description: {entry.text.strip()}")
    tags = _join_values(entry.tags)
    if tags:
        parts.append(f"tags: {tags}")
    emotion = _join_values(entry.emotion)
    if emotion:
        parts.append(f"emotion: {emotion}")
    if entry.background and entry.background.strip() != "无":
        parts.append(f"background: {entry.background.strip()}")
    return "\n".join(parts)


def prepare_dimension_texts(entries: list[Metadata]) -> dict[str, list[str]]:
    return {
        "character": [_join_values(entry.character) for entry in entries],
        "usage": [_join_values(entry.usage) for entry in entries],
        "content": [_content_text(entry) for entry in entries],
    }


def rerank_text(entry: Metadata) -> str:
    """Full text of one entry for cross-encoder reranking: content plus character and usage."""
    parts = []
    filename = os.path.splitext(os.path.basename(entry.path))[0].strip()
    if filename:
        parts.append(f"filename: {filename}")
    if entry.text:
        parts.append(f"description: {entry.text.strip()}")
    for label, values in (
        ("tags", entry.tags),
        ("character", entry.character),
        ("emotion", entry.emotion),
        ("usage", entry.usage),
    ):
        text = _join_values(values)
        if text:
            parts.append(f"{label}: {text}")
    if entry.background and entry.background.strip() != "无":
        parts.append(f"background: {entry.background.strip()}")
    return "\n".join(parts)


async def _embed_with_retry(
    client: AIClient,
    inputs: list[str],
    retries: int = config.RETRY_COUNT,
) -> list[list[float]]:
    for attempt in range(retries):
        try:
            vectors = await client.embed(inputs)
            if len(vectors) != len(inputs):
                raise RuntimeError(
                    f"Embedding API returned {len(vectors)} vectors for {len(inputs)} inputs."
                )
            return vectors
        except RuntimeError:
            if attempt >= retries - 1:
                raise
            await asyncio.sleep(1.5 ** attempt)
    raise RuntimeError("Embedding request failed.")


async def _embed_batches(client: AIClient, inputs: list[str]) -> list[list[float]]:
    result = []
    batch_size = max(1, config.EMBEDDING_BATCH_SIZE)
    for start in range(0, len(inputs), batch_size):
        result.extend(await _embed_with_retry(client, inputs[start:start + batch_size]))
    return result


def _normalize_rows(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise RuntimeError(f"Expected a 2D embedding matrix, got shape {vectors.shape}.")
    norms = np.linalg.norm(vectors, axis=1)
    mask = np.isfinite(norms) & (norms > 1e-10)
    normalized = np.zeros_like(vectors, dtype=np.float32)
    normalized[mask] = vectors[mask] / norms[mask, None]
    return normalized, mask


async def build_search_index(client: AIClient, entries: list[Metadata]) -> SearchIndex:
    if not entries:
        raise ValueError("Cannot build an index without metadata entries.")

    dimension_texts = prepare_dimension_texts(entries)
    positions = []
    inputs = []
    masks = {}
    for dimension in DIMENSIONS:
        texts = dimension_texts[dimension]
        mask = np.array([bool(text.strip()) for text in texts], dtype=np.bool_)
        masks[dimension] = mask
        for row in np.flatnonzero(mask):
            positions.append((dimension, int(row)))
            inputs.append(texts[row])

    if not inputs:
        raise ValueError("Metadata entries do not contain any searchable fields.")

    embedded = np.asarray(await _embed_batches(client, inputs), dtype=np.float32)
    if embedded.ndim != 2 or embedded.shape[0] != len(inputs):
        raise RuntimeError(f"Invalid embedding response shape: {embedded.shape}.")
    dimension_size = embedded.shape[1]
    vectors = {
        dimension: np.zeros((len(entries), dimension_size), dtype=np.float32)
        for dimension in DIMENSIONS
    }
    for (dimension, row), vector in zip(positions, embedded):
        vectors[dimension][row] = vector

    for dimension in DIMENSIONS:
        vectors[dimension], valid_vectors = _normalize_rows(vectors[dimension])
        masks[dimension] &= valid_vectors
    return SearchIndex(vectors=vectors, masks=masks)


def metadata_fingerprint(entries: list[Metadata]) -> str:
    indexed_data = []
    for entry in entries:
        indexed_data.append({
            "id": entry.id,
            "path": entry.path,
            "text": entry.text,
            "tags": entry.tags,
            "character": entry.character,
            "emotion": entry.emotion,
            "usage": entry.usage,
            "background": entry.background,
        })
    payload = json.dumps(
        indexed_data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def index_manifest_path(index_path: str) -> str:
    base, _ = os.path.splitext(index_path)
    return base + "_meta.json"


def save_search_index(
    index: SearchIndex,
    index_path: str,
    *,
    model: str,
    fingerprint: str,
):
    os.makedirs(os.path.dirname(os.path.abspath(index_path)), exist_ok=True)
    temp_index = index_path + ".tmp.npz"
    temp_manifest = index_manifest_path(index_path) + ".tmp"
    arrays = {}
    for dimension in DIMENSIONS:
        arrays[dimension] = index.vectors[dimension]
        arrays[f"{dimension}_mask"] = index.masks[dimension]
    np.savez_compressed(temp_index, **arrays)

    manifest = {
        "version": INDEX_VERSION,
        "model": model,
        "count": index.count,
        "dimensions": list(DIMENSIONS),
        "dimension_size": next(iter(index.vectors.values())).shape[1],
        "fingerprint": fingerprint,
    }
    with open(temp_manifest, "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    os.replace(temp_index, index_path)
    os.replace(temp_manifest, index_manifest_path(index_path))
    index.manifest = manifest


def load_search_index(index_path: str) -> SearchIndex:
    try:
        with open(index_manifest_path(index_path), "r", encoding="utf-8") as file:
            manifest = json.load(file)
        with np.load(index_path, allow_pickle=False) as data:
            vectors = {
                dimension: np.asarray(data[dimension], dtype=np.float32)
                for dimension in DIMENSIONS
            }
            masks = {
                dimension: np.asarray(data[f"{dimension}_mask"], dtype=np.bool_)
                for dimension in DIMENSIONS
            }
    except (
        EOFError,
        FileNotFoundError,
        KeyError,
        ValueError,
        OSError,
        zipfile.BadZipFile,
        json.JSONDecodeError,
    ) as error:
        raise RuntimeError(f"Search index is missing or invalid: {error}") from error

    if not isinstance(manifest, dict):
        raise RuntimeError("Search index manifest must be a JSON object.")
    if any(matrix.ndim != 2 for matrix in vectors.values()):
        raise RuntimeError("Search index vectors must all be two-dimensional.")
    counts = {matrix.shape[0] for matrix in vectors.values()}
    widths = {matrix.shape[1] for matrix in vectors.values()}
    if len(counts) != 1 or len(widths) != 1:
        raise RuntimeError("Search index vector shapes are inconsistent.")
    count = next(iter(counts))
    if any(mask.shape != (count,) for mask in masks.values()):
        raise RuntimeError("Search index masks are inconsistent with its vectors.")
    if manifest.get("count") != count:
        raise RuntimeError("Search index manifest count does not match its vectors.")
    if manifest.get("version") != INDEX_VERSION:
        raise RuntimeError(f"Unsupported search index version: {manifest.get('version')}.")
    if manifest.get("dimensions") != list(DIMENSIONS):
        raise RuntimeError("Search index manifest dimensions are invalid.")
    if manifest.get("dimension_size") != next(iter(widths)):
        raise RuntimeError("Search index manifest dimension size is invalid.")
    return SearchIndex(vectors=vectors, masks=masks, manifest=manifest)


async def parse_search_query(client: AIClient, query: str) -> SearchQuery:
    raw_query = query.strip()
    if not raw_query:
        raise ValueError("Query cannot be empty.")
    try:
        response = await client.chat(
            messages=[
                {"role": "system", "content": QUERY_PROMPT},
                {"role": "user", "content": raw_query},
            ],
            temperature=0,
            response_format=QUERY_FORMAT,
        )
        data = json.loads(response)
        parsed = SearchQuery(
            content=data.get("content", ""),
            character=data.get("character", ""),
            usage=data.get("usage", ""),
        )
        if parsed.content or parsed.character or parsed.usage:
            return parsed
        print("  QUERY PARSE SKIP: model returned an empty query, using the raw text.")
    except Exception as error:
        print(f"  QUERY PARSE SKIP: {error}")
    return SearchQuery(content=raw_query)


def _resolved_weights(weights: dict[str, float] | None) -> dict[str, float]:
    resolved = dict(DEFAULT_WEIGHTS)
    for dimension, value in (weights or {}).items():
        if dimension not in DIMENSIONS:
            raise ValueError(f"Unknown search dimension: {dimension}.")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Weight for {dimension} must be a non-negative number.")
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Weight for {dimension} must be a non-negative number.")
        resolved[dimension] = value
    return resolved


async def search(
    client: AIClient,
    query: SearchQuery,
    entries: list[Metadata],
    index: SearchIndex,
    top_n: int = config.TOP_N,
    weights: dict[str, float] | None = None,
) -> list[SearchResult]:
    if not isinstance(query, SearchQuery):
        raise TypeError("query must be a SearchQuery.")
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n <= 0:
        raise ValueError("top_n must be a positive integer.")
    if len(entries) != index.count:
        raise RuntimeError(
            f"Metadata/index mismatch: {len(entries)} entries vs {index.count} vectors."
        )
    if not entries:
        return []

    weights = _resolved_weights(weights)
    active = [
        dimension for dimension in DIMENSIONS
        if getattr(query, dimension).strip() and weights[dimension] > 0
    ]
    if not active:
        raise ValueError("Query cannot be empty.")

    query_vectors = np.asarray(
        await _embed_with_retry(client, [getattr(query, dimension) for dimension in active]),
        dtype=np.float32,
    )
    query_vectors, valid_queries = _normalize_rows(query_vectors)
    dimension_texts = prepare_dimension_texts(entries)

    rankings = {}
    similarities = {}
    usable = []
    for position, dimension in enumerate(active):
        valid_documents = index.masks[dimension]
        if not valid_queries[position] or not valid_documents.any():
            continue
        scores = np.clip(index.vectors[dimension] @ query_vectors[position], -1, 1)
        document_indices = np.flatnonzero(valid_documents)
        query_text = getattr(query, dimension).casefold()
        literal_matches = np.array([
            query_text in text.casefold()
            for text in dimension_texts[dimension]
        ], dtype=np.bool_)
        order = document_indices[np.lexsort((
            -scores[document_indices], -literal_matches[document_indices].astype(np.int8)
        ))]
        rankings[dimension] = {int(row): rank for rank, row in enumerate(order, 1)}
        similarities[dimension] = scores
        usable.append(dimension)

    if not usable:
        return []

    max_rrf = sum(weights[dimension] for dimension in usable) / (config.RRF_K + 1)
    active_weight = sum(weights[dimension] for dimension in usable)
    ranked_results = []
    for row, metadata in enumerate(entries):
        raw_score = 0.0
        weighted_similarity = 0.0
        dimension_scores = {}
        for dimension in usable:
            rank = rankings[dimension].get(row)
            if rank is None:
                continue
            weight = weights[dimension]
            raw_contribution = weight / (config.RRF_K + rank)
            contribution = raw_contribution / max_rrf
            similarity = float(similarities[dimension][row])
            raw_score += raw_contribution
            weighted_similarity += weight * similarity
            dimension_scores[dimension] = DimensionScore(
                similarity=similarity,
                rank=rank,
                weight=weight,
                contribution=contribution,
            )
        if not dimension_scores:
            continue
        result = SearchResult(
            metadata=metadata,
            score=raw_score / max_rrf,
            dimensions=dimension_scores,
        )
        ranked_results.append((result, weighted_similarity / active_weight, row))

    ranked_results.sort(key=lambda item: (-item[0].score, -item[1], item[2]))
    return [item[0] for item in ranked_results[:min(top_n, len(ranked_results))]]
