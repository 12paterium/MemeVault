import os, asyncio, numpy as np
from .AIClient import AIClient
from .Metadata import Metadata
from . import config

embedding_client = AIClient(model=config.EMBEDDING_MODEL)


def prepare_texts(entries: list[Metadata]) -> list[str]:
    texts = []
    for e in entries:
        parts = []
        fname = os.path.splitext(os.path.basename(e.path))[0]
        # repeat filename to weigh it heavier — meaningful names often summarize the meme
        if fname and fname != e.text:
            parts.append(" ".join([fname] * 3))
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


async def _embed_with_retry(client: AIClient, texts: list[str], retries: int = config.RETRY_COUNT) -> list[list[float]]:
    for attempt in range(retries):
        try:
            return await client.embed(texts)
        except RuntimeError as e:
            if attempt < retries - 1:
                await asyncio.sleep(1.5 ** attempt)
            else:
                raise


async def build_embeddings(client: AIClient, entries: list[Metadata]) -> np.ndarray:
    if not entries:
        return np.empty((0, 0), dtype=np.float32)
    texts = prepare_texts(entries)
    all_vectors = []
    for i in range(0, len(texts), config.BATCH_SIZE):
        batch = texts[i:i + config.BATCH_SIZE]
        all_vectors.extend(await _embed_with_retry(client, batch))
    return np.array(all_vectors, dtype=np.float32) if all_vectors else np.empty((0, 0), dtype=np.float32)


def save_embeddings(embeddings: np.ndarray, path: str = config.EMBEDDINGS_PATH):
    if embeddings.size == 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, embeddings)


def load_embeddings(path: str = config.EMBEDDINGS_PATH) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"embeddings.npy not found. Run 'build' first.")
    arr = np.load(path)
    if arr.ndim != 2:
        raise RuntimeError(f"Expected 2D array, got shape {arr.shape}")
    return arr


async def search(query: str, entries: list[Metadata], embeddings: np.ndarray, top_k: int = config.TOP_K) -> list:
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
    query_vec = np.array((await embedding_client.embed([query]))[0], dtype=np.float32)
    query_norm = np.linalg.norm(query_vec)
    if query_norm < 1e-10:
        return []

    norms = np.linalg.norm(embeddings, axis=1)
    valid = norms > 1e-10
    if not valid.any():
        return []

    similarities = np.clip(embeddings[valid] @ query_vec / (norms[valid] * query_norm), -1, 1)
    indices = np.argsort(similarities)[-top_k:][::-1]
    # map back from valid-subset indices to positions in the original entries list
    original_indices = np.where(valid)[0][indices]
    return [(entries[i], float(similarities[j])) for j, i in enumerate(original_indices)]
