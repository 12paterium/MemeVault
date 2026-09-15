import unittest
from unittest import mock

import httpx

from meme_vault.client import AIClient, _retry_delay, parse_json_object


def _client(handler):
    client = AIClient(model="test-model", api_key="test-key", base_url="http://test")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


class RetryDelayTests(unittest.TestCase):
    def test_server_hint_wins_over_backoff(self):
        self.assertEqual(_retry_delay(httpx.Response(429, headers={"Retry-After": "7"}), {}, 0), 7.0)
        self.assertEqual(_retry_delay(httpx.Response(429), {"error": {"retry_after": 3}}, 0), 3.0)

    def test_backoff_without_hints(self):
        self.assertEqual(_retry_delay(httpx.Response(500), None, 0), 2.0)
        self.assertEqual(_retry_delay(None, None, 2), 8.0)


class ParseJsonObjectTests(unittest.TestCase):
    def test_plain_object(self):
        self.assertEqual(parse_json_object('{"content": "无语"}'), {"content": "无语"})

    def test_code_fence(self):
        text = '```json\n{"text": "猫", "tags": ["猫"]}\n```'
        self.assertEqual(parse_json_object(text), {"text": "猫", "tags": ["猫"]})

    def test_fence_with_trailing_prose(self):
        text = '```json\n{"content": "早八"}\n```\n以上是拆解结果。'
        self.assertEqual(parse_json_object(text), {"content": "早八"})

    def test_prose_around_object(self):
        text = '好的，结果是：\n{"content": "无语", "usage": "吐槽"}\n希望有帮助。'
        self.assertEqual(parse_json_object(text), {"content": "无语", "usage": "吐槽"})

    def test_nested_object_is_kept_whole(self):
        text = '{"a": {"b": 1}, "c": "}"}'
        self.assertEqual(parse_json_object(text), {"a": {"b": 1}, "c": "}"})

    def test_object_inside_an_array_is_still_found(self):
        self.assertEqual(parse_json_object('[{"a": 1}]'), {"a": 1})

    def test_array_without_an_object_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_json_object('[1, 2]')

    def test_garbage_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            parse_json_object("我不确定该怎么回答这个问题")
        self.assertIn("我不确定", str(caught.exception))

    def test_empty_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_json_object("   ")


class PostRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_network_error_is_retried(self):
        calls = []

        async def handler(request):
            calls.append(request)
            if len(calls) < 3:
                raise httpx.ConnectError("connection reset", request=request)
            return httpx.Response(200, json={"ok": True})

        client = _client(handler)
        with mock.patch("meme_vault.client.asyncio.sleep", new=mock.AsyncMock()):
            result = await client._post("http://test/chat/completions", {})
        await client.close()

        self.assertEqual(result, {"ok": True})
        self.assertEqual(len(calls), 3)

    async def test_network_error_gives_up_with_its_message(self):
        async def handler(request):
            raise httpx.ReadTimeout("slow provider", request=request)

        client = _client(handler)
        with mock.patch("meme_vault.client.asyncio.sleep", new=mock.AsyncMock()):
            with self.assertRaises(RuntimeError) as caught:
                await client._post("http://test/embeddings", {})
        await client.close()

        self.assertIn("slow provider", str(caught.exception))

    async def test_retryable_status_then_success(self):
        responses = [httpx.Response(503), httpx.Response(200, json={"ok": True})]

        async def handler(request):
            return responses.pop(0)

        client = _client(handler)
        with mock.patch("meme_vault.client.asyncio.sleep", new=mock.AsyncMock()):
            result = await client._post("http://test/chat/completions", {})
        await client.close()

        self.assertEqual(result, {"ok": True})
