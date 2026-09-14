import json
import os
import tempfile
import unittest

import numpy as np

from meme_vault.embedding import (
    SearchIndex,
    SearchQuery,
    build_search_index,
    load_search_index,
    metadata_fingerprint,
    parse_search_query,
    save_search_index,
    search,
)
from meme_vault.metadata import Metadata


def _vector_for(text: str) -> list[float]:
    text = str(text)
    values = np.array([
        float("初音未来" in text),
        float("哆啦A梦" in text),
        float("吐槽" in text),
        float("庆祝" in text),
        float("开心" in text),
        float("悲伤" in text),
        float("无语" in text),
    ], dtype=np.float32)
    if not values.any():
        values[-1] = 0.25
    return values.tolist()


class FakeClient:
    def __init__(self, chat_response=None, chat_error=None):
        self.model = "fake-model"
        self.chat_response = chat_response
        self.chat_error = chat_error
        self.embed_calls = []
        self.chat_calls = []

    async def embed(self, inputs):
        self.embed_calls.append(list(inputs))
        return [_vector_for(value) for value in inputs]

    async def chat(self, **kwargs):
        self.chat_calls.append(kwargs)
        if self.chat_error:
            raise self.chat_error
        return self.chat_response

    async def close(self):
        pass


def _entry(identifier, *, text, character, usage, emotion=None):
    return Metadata.from_dict({
        "id": identifier,
        "path": f"{identifier}.jpg",
        "text": text,
        "tags": [],
        "character": character,
        "emotion": emotion or [],
        "usage": usage,
        "background": "无",
    })


class SearchIndexTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.entries = [
            _entry(
                "primary",
                text="悲伤的反应",
                character=["初音未来"],
                usage=["吐槽"],
                emotion=["悲伤"],
            ),
            _entry(
                "content",
                text="开心的反应",
                character=["哆啦A梦"],
                usage=["庆祝"],
                emotion=["开心"],
            ),
        ]

    async def test_primary_dimensions_outweigh_content(self):
        client = FakeClient()
        index = await build_search_index(client, self.entries)

        results = await search(
            client,
            SearchQuery(character="初音未来", usage="吐槽", content="开心"),
            self.entries,
            index,
            top_n=2,
        )

        self.assertEqual([result.metadata.id for result in results], ["primary", "content"])
        winner = results[0]
        self.assertEqual(winner.dimensions["character"].rank, 1)
        self.assertEqual(winner.dimensions["usage"].rank, 1)
        self.assertEqual(winner.dimensions["content"].rank, 2)
        self.assertEqual(winner.dimensions["character"].weight, 3.0)
        self.assertEqual(winner.dimensions["usage"].weight, 3.0)
        self.assertEqual(winner.dimensions["content"].weight, 1.0)
        self.assertGreater(winner.score, results[1].score)

    async def test_weight_override_can_change_the_winner(self):
        client = FakeClient()
        index = await build_search_index(client, self.entries)

        results = await search(
            client,
            SearchQuery(
                character="初音未来",
                usage="吐槽",
                content="开心",
            ),
            self.entries,
            index,
            top_n=2,
            weights={"content": 10},
        )

        self.assertEqual(results[0].metadata.id, "content")
        self.assertEqual(results[0].dimensions["content"].weight, 10.0)

    async def test_literal_match_precedes_higher_similarity_non_match(self):
        entries = [
            _entry(
                "literal",
                text="一只蓝色卡通猫",
                character=[],
                usage=[],
            ),
            _entry(
                "semantic-only",
                text="一只粉色卡通猪",
                character=[],
                usage=[],
            ),
        ]
        index = SearchIndex(
            vectors={
                "character": np.zeros((2, 7), dtype=np.float32),
                "usage": np.zeros((2, 7), dtype=np.float32),
                "content": np.array(
                    [[0.6, 0, 0, 0, 0, 0, 0.8], [0, 0, 0, 0, 0, 0, 1.0]],
                    dtype=np.float32,
                ),
            },
            masks={
                "character": np.zeros(2, dtype=np.bool_),
                "usage": np.zeros(2, dtype=np.bool_),
                "content": np.ones(2, dtype=np.bool_),
            },
        )

        results = await search(
            FakeClient(),
            SearchQuery(content="蓝色"),
            entries,
            index,
            top_n=2,
        )

        self.assertEqual(
            [result.metadata.id for result in results],
            ["literal", "semantic-only"],
        )

    async def test_index_round_trip_preserves_vectors_and_manifest(self):
        client = FakeClient()
        index = await build_search_index(client, self.entries)
        fingerprint = metadata_fingerprint(self.entries)

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "search_index.npz")
            save_search_index(
                index,
                path,
                model=client.model,
                fingerprint=fingerprint,
            )
            loaded = load_search_index(path)

        self.assertEqual(loaded.count, len(self.entries))
        self.assertEqual(loaded.manifest["model"], client.model)
        self.assertEqual(loaded.manifest["fingerprint"], fingerprint)
        for dimension in ("character", "usage", "content"):
            np.testing.assert_allclose(loaded.vectors[dimension], index.vectors[dimension])
            np.testing.assert_array_equal(loaded.masks[dimension], index.masks[dimension])

    async def test_invalid_manifest_is_rejected(self):
        client = FakeClient()
        index = await build_search_index(client, self.entries)

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "search_index.npz")
            save_search_index(
                index,
                path,
                model=client.model,
                fingerprint=metadata_fingerprint(self.entries),
            )
            manifest_path = os.path.join(directory, "search_index_meta.json")
            with open(manifest_path, "w", encoding="utf-8") as file:
                json.dump({"version": 999, "count": len(self.entries)}, file)

            with self.assertRaisesRegex(RuntimeError, "Unsupported search index version"):
                load_search_index(path)

    async def test_missing_document_dimension_is_not_ranked(self):
        entries = [
            self.entries[0],
            _entry(
                "missing-character",
                text="开心的反应",
                character=[],
                usage=["吐槽"],
                emotion=["开心"],
            ),
        ]
        client = FakeClient()
        index = await build_search_index(client, entries)

        results = await search(
            client,
            SearchQuery(character="初音未来", usage="吐槽"),
            entries,
            index,
            top_n=2,
        )

        missing = next(result for result in results if result.metadata.id == "missing-character")
        self.assertNotIn("character", missing.dimensions)
        self.assertIn("usage", missing.dimensions)

    async def test_rejects_invalid_query_and_weights(self):
        client = FakeClient()
        index = await build_search_index(client, self.entries)

        with self.assertRaisesRegex(ValueError, "Query cannot be empty"):
            await search(client, SearchQuery(), self.entries, index)
        with self.assertRaisesRegex(ValueError, "Unknown search dimension"):
            await search(
                client,
                SearchQuery(content="开心"),
                self.entries,
                index,
                weights={"emotion": 2},
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            await search(
                client,
                SearchQuery(content="开心"),
                self.entries,
                index,
                weights={"content": -1},
            )


class QueryParsingTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_query_decomposition(self):
        response = json.dumps({
            "content": "无语",
            "character": "初音未来",
            "usage": "吐槽群友",
        }, ensure_ascii=False)
        client = FakeClient(chat_response=response)

        query = await parse_search_query(client, "找初音未来吐槽群友的无语表情")

        self.assertEqual(query.character, "初音未来")
        self.assertEqual(query.usage, "吐槽群友")
        self.assertEqual(query.content, "无语")
        self.assertEqual(len(client.chat_calls), 1)

    async def test_query_decomposition_failure_falls_back_to_content(self):
        client = FakeClient(chat_error=RuntimeError("offline"))

        query = await parse_search_query(client, "随便找一张无语图")

        self.assertEqual(query.content, "随便找一张无语图")
        self.assertEqual(query.character, "")
        self.assertEqual(query.usage, "")

    async def test_non_runtime_query_failure_also_falls_back(self):
        client = FakeClient(chat_error=OSError("network unavailable"))

        query = await parse_search_query(client, "找一张开心的图")

        self.assertEqual(query.content, "找一张开心的图")

    async def test_empty_query_is_rejected_before_chat_call(self):
        client = FakeClient(chat_response="{}")

        with self.assertRaisesRegex(ValueError, "Query cannot be empty"):
            await parse_search_query(client, "  ")
        self.assertEqual(client.chat_calls, [])
