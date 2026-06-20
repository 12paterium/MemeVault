import asyncio, base64, os, numpy as np
from .client import AIClient
from .metadata import Metadata
from . import config

_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
         ".gif": "image/gif", ".bmp": "image/bmp", ".webp": "image/webp"}


def _read_image_b64(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    mime = _MIME.get(ext, "image/jpeg")
    with open(path, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"


def prepare_texts(entries: list[Metadata]) -> list[str]:
    texts = []
    for e in entries:
        parts = []
        fname = os.path.splitext(os.path.basename(e.path))[0]
        if fname and fname != e.text:
            parts.append(fname)
        if e.text:
            parts.append(e.text)
        if e.tags:
            parts.append(" ".join(e.tags))
        if e.emotion:
            parts.append(" ".join(e.emotion))
        if e.usage:
            parts.append(" ".join(e.usage))
        if e.character:
            parts.append(" ".join(e.character))
        if e.background and e.background != "无":
            parts.append(e.background)
        texts.append(" ".join(parts).strip() or "(empty)")
    return texts


async def build_text_embeddings(client: AIClient, entries: list[Metadata]) -> np.ndarray:
    if not entries:
        return np.empty((0, 0), dtype=np.float32)
    texts = prepare_texts(entries)
    vectors = await _embed_with_retry(client, texts)
    return np.array(vectors, dtype=np.float32)


async def build_image_embeddings(client: AIClient, entries: list[Metadata]) -> np.ndarray:
    if not entries:
        return np.empty((0, 0), dtype=np.float32)
    inputs = [{"image": _read_image_b64(e.path)} for e in entries]
    vectors = await _embed_with_retry(client, inputs)
    return np.array(vectors, dtype=np.float32)


async def _embed_with_retry(client: AIClient, inputs: list, retries: int = config.RETRY_COUNT) -> list[list[float]]:
    for attempt in range(retries):
        try:
            return await client.embed(inputs)
        except RuntimeError as e:
            if attempt < retries - 1:
                await asyncio.sleep(1.5 ** attempt)
            else:
                raise


def save_embeddings(embeddings: np.ndarray, path: str):
    if embeddings.size == 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, embeddings)


def load_embeddings(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"embeddings.npy not found. Run 'build' first.")
    arr = np.load(path)
    if arr.ndim != 2:
        raise RuntimeError(f"Expected 2D array, got shape {arr.shape}")
    return arr


async def search(client: AIClient, query: str, entries: list[Metadata],
                 embeddings: np.ndarray, top_k: int = config.TOP_K) -> list:
    if not query.strip():
        raise ValueError("Query cannot be empty.")
    if not entries or embeddings.size == 0:
        return []
    if len(entries) != embeddings.shape[0]:
        raise RuntimeError(
            f"Entry/embedding mismatch: {len(entries)} entries vs {embeddings.shape[0]} embeddings. "
            f"Run 'build' first."
        )

    top_k = min(top_k, len(entries))
    query_vec = np.array((await client.embed([query]))[0], dtype=np.float32)

    query_norm = np.linalg.norm(query_vec)
    if query_norm < 1e-10:
        return []

    norms = np.linalg.norm(embeddings, axis=1)
    valid = norms > 1e-10
    if not valid.any():
        return []

    similarities = np.clip(embeddings[valid] @ query_vec / (norms[valid] * query_norm), -1, 1)
    indices = np.argsort(similarities)[-top_k:][::-1]
    original_indices = np.where(valid)[0][indices]
    return [(entries[i], float(similarities[j])) for j, i in enumerate(original_indices)]
