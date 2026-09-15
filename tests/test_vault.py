import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout

import numpy as np

from meme_vault.embedding import SearchQuery, load_search_index
from meme_vault.metadata import Metadata, load_metadata, save_metadata
from meme_vault.vault import IMAGE_EXTS, MemeVault

from tests.test_embedding import FakeClient


class FakeVisionClient:
    model = "fake-vision"

    def __init__(self):
        self.parse_calls = []

    async def parse_image(self, path, filename=None):
        self.parse_calls.append((path, filename))
        filename = filename or os.path.splitext(os.path.basename(path))[0]
        character = ["初音未来"] if "初音未来" in filename else ["未知角色"]
        return {
            "text": filename,
            "tags": ["测试资源"],
            "character": character,
            "emotion": ["无语"],
            "usage": ["聊天回应"],
            "background": "无",
        }

    async def close(self):
        pass


class FakeRerankClient:
    model = "fake-rerank"

    def __init__(self, order=None, error=None):
        self.order = order
        self.error = error
        self.calls = []

    async def rerank(self, query, documents, top_n=None):
        self.calls.append({"query": query, "documents": list(documents), "top_n": top_n})
        if self.error:
            raise self.error
        if self.order is not None:
            return self.order
        return [(index, 1.0 - index) for index in range(len(documents))]

    async def close(self):
        pass


class MemeVaultTests(unittest.IsolatedAsyncioTestCase):
    async def test_build_search_and_stale_index_rebuild(self):
        entries = [
            Metadata.from_dict({
                "id": "miku",
                "path": "miku.jpg",
                "text": "悲伤",
                "character": ["初音未来"],
                "usage": ["吐槽"],
                "emotion": ["悲伤"],
            }),
            Metadata.from_dict({
                "id": "doraemon",
                "path": "doraemon.jpg",
                "text": "开心",
                "character": ["哆啦A梦"],
                "usage": ["庆祝"],
                "emotion": ["开心"],
            }),
        ]
        with tempfile.TemporaryDirectory() as directory:
            save_metadata(entries, directory)
            vault = MemeVault(
                data_dir=directory,
                api_key="test",
                embedding_model="fake-model",
                vision_model="fake-vision",
            )
            embed_client = FakeClient()
            vision_client = FakeClient(chat_response=json.dumps({
                "content": "开心",
                "character": "初音未来",
                "usage": "吐槽",
            }, ensure_ascii=False))
            chat_client = FakeClient(chat_response=json.dumps({
                "content": "开心",
                "character": "初音未来",
                "usage": "吐槽",
            }, ensure_ascii=False))
            vault._embed_client = embed_client
            vault._vision_client = vision_client
            vault._chat_client = chat_client

            count = await vault.build()
            self.assertEqual(count, 2)
            first_build_calls = len(embed_client.embed_calls)
            self.assertTrue(os.path.exists(vault.index_path))

            self.assertEqual(await vault.build(), 2)
            self.assertEqual(len(embed_client.embed_calls), first_build_calls)

            results = await vault.search(
                SearchQuery(character="初音未来", usage="吐槽", content="开心"),
                top_n=2,
            )
            self.assertEqual(results[0].metadata.id, "miku")
            self.assertEqual(vision_client.chat_calls, [])

            parsed_results = await vault.search("找初音未来吐槽群友的图", top_n=1)
            self.assertEqual(parsed_results[0].metadata.id, "miku")
            self.assertEqual(len(chat_client.chat_calls), 1)
            self.assertEqual(vision_client.chat_calls, [])

            entries[0].usage = ["庆祝"]
            save_metadata(entries, directory)
            await vault.search(SearchQuery(content="开心"), top_n=1)
            self.assertGreater(len(embed_client.embed_calls), first_build_calls)
            await vault.close()

    async def test_empty_vault_search_returns_no_results(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = MemeVault(data_dir=directory, api_key="test")
            self.assertEqual(await vault.search(SearchQuery(content="无语")), [])

    async def test_search_weights_and_context_manager(self):
        entries = [
            Metadata.from_dict({
                "id": "miku",
                "path": "miku.jpg",
                "text": "悲伤",
                "character": ["初音未来"],
                "usage": ["吐槽"],
                "emotion": ["悲伤"],
            }),
            Metadata.from_dict({
                "id": "doraemon",
                "path": "doraemon.jpg",
                "text": "开心",
                "character": ["哆啦A梦"],
                "usage": ["庆祝"],
                "emotion": ["开心"],
            }),
        ]
        with tempfile.TemporaryDirectory() as directory:
            save_metadata(entries, directory)
            async with MemeVault(
                data_dir=directory,
                api_key="test",
                embedding_model="fake-model",
                vision_model="fake-vision",
            ) as vault:
                vault._embed_client = FakeClient()

                await vault.build()
                balanced = await vault.search(
                    SearchQuery(character="初音未来", usage="吐槽", content="开心"),
                    top_n=1,
                )
                self.assertEqual(balanced[0].metadata.id, "miku")

                content_only = await vault.search(
                    SearchQuery(character="初音未来", usage="吐槽", content="开心"),
                    top_n=1,
                    weights={"character": 0, "usage": 0, "content": 10},
                )
                self.assertEqual(content_only[0].metadata.id, "doraemon")

                with self.assertRaises(TypeError):
                    await vault.search(SearchQuery(content="开心"), weights="usage")


class RerankTests(unittest.IsolatedAsyncioTestCase):
    def _entries(self):
        return [
            Metadata.from_dict({
                "id": "miku",
                "path": "miku.jpg",
                "text": "悲伤的反应",
                "character": ["初音未来"],
                "usage": ["吐槽"],
                "emotion": ["悲伤"],
            }),
            Metadata.from_dict({
                "id": "doraemon",
                "path": "doraemon.jpg",
                "text": "开心的反应",
                "character": ["哆啦A梦"],
                "usage": ["庆祝"],
                "emotion": ["开心"],
            }),
        ]

    async def _vault(self, directory):
        vault = MemeVault(
            data_dir=directory,
            api_key="test",
            embedding_model="fake-model",
            vision_model="fake-vision",
        )
        vault._embed_client = FakeClient()
        await vault.build()
        return vault

    async def test_rerank_reorders_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            save_metadata(self._entries(), directory)
            vault = await self._vault(directory)

            baseline = await vault.search(SearchQuery(character="初音未来"), top_n=2)
            self.assertEqual(baseline[0].metadata.id, "miku")
            self.assertIsNone(baseline[0].rerank_score)

            rerank_client = FakeRerankClient(order=[(1, 0.9), (0, 0.1)])
            vault._rerank_client = rerank_client
            ranked = await vault.search(SearchQuery(character="初音未来"), top_n=2, rerank=True)

            self.assertEqual([result.metadata.id for result in ranked], ["doraemon", "miku"])
            self.assertEqual(ranked[0].rerank_score, 0.9)
            self.assertEqual(ranked[1].rerank_score, 0.1)
            self.assertEqual(len(rerank_client.calls), 1)
            self.assertIn("初音未来", rerank_client.calls[0]["query"])
            self.assertIn("初音未来", rerank_client.calls[0]["documents"][0])
            self.assertEqual(rerank_client.calls[0]["top_n"], len(rerank_client.calls[0]["documents"]))
            await vault.close()

    async def test_rerank_failure_keeps_rrf_order(self):
        with tempfile.TemporaryDirectory() as directory:
            save_metadata(self._entries(), directory)
            vault = await self._vault(directory)
            vault._rerank_client = FakeRerankClient(error=RuntimeError("rerank down"))

            with redirect_stdout(io.StringIO()) as output:
                ranked = await vault.search(SearchQuery(character="初音未来"), top_n=2, rerank=True)

            self.assertEqual([result.metadata.id for result in ranked], ["miku", "doraemon"])
            self.assertIn("RERANK SKIP", output.getvalue())
            await vault.close()


class PruneTests(unittest.TestCase):
    def test_prune_removes_only_missing_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            existing = os.path.join(directory, "exists.jpg")
            open(existing, "w").close()
            entries = [
                Metadata.from_dict({"id": "keep", "path": existing, "text": "保留"}),
                Metadata.from_dict({"id": "drop", "path": os.path.join(directory, "gone.jpg"), "text": "清理"}),
            ]
            save_metadata(entries, directory)
            vault = MemeVault(data_dir=directory, api_key="test")
            removed = vault.prune()
            self.assertEqual([entry.id for entry in removed], ["drop"])
            self.assertEqual([entry.id for entry in load_metadata(directory)], ["keep"])
            self.assertEqual(vault.prune(), [])


class ResourceImagesTests(unittest.IsolatedAsyncioTestCase):
    def _resource_root(self):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        resource_root = os.path.join(project_root, "ResourceImages")
        if not os.path.isdir(resource_root):
            self.skipTest("ResourceImages test set is not available")
        return project_root, resource_root

    def test_status_discovers_resource_images(self):
        project_root, resource_root = self._resource_root()

        expected = 0
        for directory, _, filenames in os.walk(resource_root):
            expected += sum(
                1 for filename in filenames
                if os.path.splitext(filename)[1].lower() in IMAGE_EXTS
            )
        vault = MemeVault(data_dir=os.path.join(project_root, "data"))
        status = vault.status()

        self.assertEqual(status["images"]["directory"], resource_root)
        self.assertEqual(status["images"]["files"], expected)
        self.assertGreater(expected, 0)

    async def test_resource_images_complete_fake_pipeline(self):
        _, resource_root = self._resource_root()
        image_count = sum(
            1
            for directory, _, filenames in os.walk(resource_root)
            for filename in filenames
            if os.path.splitext(filename)[1].lower() in IMAGE_EXTS
        )

        with tempfile.TemporaryDirectory() as directory:
            vault = MemeVault(
                data_dir=directory,
                api_key="test",
                embedding_model="fake-model",
                vision_model="fake-vision",
            )
            embed_client = FakeClient()
            vision_client = FakeVisionClient()
            vault._embed_client = embed_client
            vault._vision_client = vision_client

            with redirect_stdout(io.StringIO()):
                added = await vault.parse_dir(resource_root)
            self.assertEqual(added, image_count)
            self.assertEqual(len(vision_client.parse_calls), image_count)
            for path, filename in vision_client.parse_calls:
                self.assertEqual(filename, os.path.splitext(os.path.basename(path))[0].strip())
            self.assertEqual(len(load_metadata(vault.data_dir)), image_count)

            with redirect_stdout(io.StringIO()):
                self.assertEqual(await vault.parse_dir(resource_root), 0)
            self.assertEqual(len(vision_client.parse_calls), image_count)

            self.assertEqual(await vault.build(), image_count)
            results = await vault.search(
                SearchQuery(character="初音未来", content="初音未来"),
                top_n=5,
            )
            self.assertTrue(results)
            self.assertTrue(
                any("初音未来" in result.metadata.path for result in results),
                "ResourceImages entries containing 初音未来 should rank in the top results",
            )
            await vault.close()


class ProviderConfigTests(unittest.TestCase):
    def test_chat_follows_vision_unless_configured(self):
        vault = MemeVault(
            data_dir=".",
            api_key="main-key",
            vision_model="vision-model",
            vision_api_key="vision-key",
            vision_base_url="http://vision",
        )
        self.assertEqual(vault.chat_model, "vision-model")
        self.assertEqual(vault.chat_api_key, "vision-key")
        self.assertEqual(vault.chat_base_url, "http://vision")

        split = MemeVault(
            data_dir=".",
            api_key="main-key",
            vision_api_key="vision-key",
            vision_base_url="http://vision",
            chat_model="chat-model",
            chat_api_key="chat-key",
            chat_base_url="http://chat",
        )
        self.assertEqual(split.chat_model, "chat-model")
        self.assertEqual(split.chat_api_key, "chat-key")
        self.assertEqual(split.chat_base_url, "http://chat")
        self.assertEqual(split.vision_model, "Qwen/Qwen3-VL-32B-Instruct")


class RecordingClient:
    """Minimal stand-in for AIClient, recording what the vault asks of it."""

    def __init__(self, model):
        self.model = model
        self.parsed = []
        self.closed = False

    async def chat(self, messages, temperature=0.2, response_format=None):
        return '{"content": "无语"}'

    async def parse_image(self, path, filename=None):
        self.parsed.append((path, filename))
        return {"text": "图", "tags": [], "character": [], "emotion": [], "usage": [], "background": "无"}

    async def embed(self, inputs):
        return [[1.0, 0.0] for _ in inputs]

    async def rerank(self, query, documents, top_n=None):
        return [(index, 1.0) for index in range(len(documents))]

    async def close(self):
        self.closed = True


class InjectedClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_injected_clients_are_used_and_kept_alive(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "猫 疑惑.jpg")
            with open(source, "wb") as file:
                file.write(b"fake-image-bytes")
            vision = RecordingClient("injected-vision")
            chat = RecordingClient("injected-chat")
            vault = MemeVault(
                data_dir=os.path.join(directory, "vault"),
                chat_client=chat,
                vision_client=vision,
            )
            self.assertEqual(vault.vision_model, "injected-vision")
            self.assertEqual(vault.chat_model, "injected-chat")

            entry = await vault.parse(source)
            await vault.close()

            self.assertEqual(vision.parsed, [(source, "猫 疑惑")])
            self.assertEqual(entry.analyzed_by, "injected-vision")
            self.assertFalse(vision.closed, "injected clients belong to the caller")
            self.assertFalse(chat.closed)
            self.assertIs(vault._vision_client, vision)

    async def test_explicit_names_override_injected_clients(self):
        vault = MemeVault(
            data_dir=".",
            chat_model="explicit-chat",
            chat_client=RecordingClient("injected-chat"),
        )
        self.assertEqual(vault.chat_model, "explicit-chat")

    async def test_created_clients_are_closed(self):
        vault = MemeVault(data_dir=".", api_key="key", embedding_model="m")
        created = vault._get_embed_client()
        await vault.close()

        self.assertIsNone(vault._embed_client)
        self.assertIsNot(vault._get_embed_client(), created)


class CacheTests(unittest.TestCase):
    def _entry(self, identifier, path):
        return Metadata.from_dict({"id": identifier, "path": path, "text": "图"})

    def test_entries_cache_reuses_and_refreshes(self):
        with tempfile.TemporaryDirectory() as directory:
            save_metadata([self._entry("a", "a.jpg")], directory)
            vault = MemeVault(data_dir=directory, api_key="test")

            self.assertEqual([entry.id for entry in vault._entries()], ["a"])
            self.assertIs(vault._entries(), vault._entries(), "an unchanged vault must reuse its cache")

            save_metadata([self._entry("a", "a.jpg"), self._entry("b", "b.jpg")], directory)
            self.assertEqual([entry.id for entry in vault._entries()], ["a", "b"])


class IndexReuseTests(unittest.IsolatedAsyncioTestCase):
    def _entries(self):
        return [
            Metadata.from_dict({
                "id": "miku",
                "path": "miku.jpg",
                "text": "悲伤的反应",
                "character": ["初音未来"],
                "usage": ["吐槽"],
                "emotion": ["悲伤"],
            }),
            Metadata.from_dict({
                "id": "doraemon",
                "path": "doraemon.jpg",
                "text": "开心的反应",
                "character": ["哆啦A梦"],
                "usage": ["庆祝"],
                "emotion": ["开心"],
            }),
        ]

    def _vault(self, directory, model):
        vault = MemeVault(
            data_dir=directory,
            api_key="test",
            embedding_model=model,
            vision_model="fake-vision",
        )
        vault._embed_client = FakeClient()
        vault._embed_client.model = model
        return vault

    async def test_rebuild_embeds_only_changed_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            entries = self._entries()
            save_metadata(entries, directory)
            vault = self._vault(directory, "fake-model")

            await vault.build()
            full_texts = sum(len(batch) for batch in vault._embed_client.embed_calls)
            before = load_search_index(vault.index_path)

            entries[0].text = "全新的描述"
            save_metadata(entries, directory, changed=[entries[0]])
            await vault.build(force=True)

            incremental_texts = sum(len(batch) for batch in vault._embed_client.embed_calls) - full_texts
            self.assertEqual(incremental_texts, 1, "only the changed content row should be re-embedded")

            after = load_search_index(vault.index_path)
            for dimension in ("character", "usage", "content"):
                np.testing.assert_array_equal(before.vectors[dimension][1], after.vectors[dimension][1])
            await vault.close()

    async def test_model_change_reembeds_everything(self):
        with tempfile.TemporaryDirectory() as directory:
            entries = self._entries()
            save_metadata(entries, directory)
            first = self._vault(directory, "fake-model")
            await first.build()
            full_texts = sum(len(batch) for batch in first._embed_client.embed_calls)

            second = self._vault(directory, "other-model")
            await second.build(force=True)

            self.assertEqual(sum(len(batch) for batch in second._embed_client.embed_calls), full_texts)
            await first.close()
            await second.close()
