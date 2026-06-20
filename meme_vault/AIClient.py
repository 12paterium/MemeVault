import asyncio, base64, json, time
import httpx

from . import config


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~1 token per char for CJK, ~4 chars per token for English."""
    cjk = sum(1 for c in text if '一' <= c <= '鿿')
    other = len(text) - cjk
    return cjk + other // 4 + 1


class RateLimiter:
    def __init__(self, rpm: int, tpm: int):
        self.max_rpm = rpm
        self.max_tpm = tpm
        self._req_log: list[float] = []
        self._tokens_used: list[tuple[float, int]] = []
        self._lock = asyncio.Lock()

    def _slide(self):
        now = time.monotonic()
        cutoff = now - 60
        self._req_log = [t for t in self._req_log if t > cutoff]
        self._tokens_used = [(t, n) for t, n in self._tokens_used if t > cutoff]

    async def acquire(self, estimated_tokens: int = 0):
        async with self._lock:
            self._slide()
            while (
                len(self._req_log) >= self.max_rpm
                or (sum(n for _, n in self._tokens_used) + estimated_tokens > self.max_tpm)
            ):
                self._slide()
                if not self._req_log and not self._tokens_used:
                    break
                # wait until the oldest request slides out of the 60s window
                oldest_req = self._req_log[0] if self._req_log else 0
                oldest_tok = self._tokens_used[0][0] if self._tokens_used else 0
                wait = max(oldest_req, oldest_tok) + 60 - time.monotonic()
                if wait > 0:
                    await asyncio.sleep(wait)
                self._slide()
            self._req_log.append(time.monotonic())
            self._tokens_used.append((time.monotonic(), estimated_tokens))


_rl = RateLimiter(config.RATE_LIMIT_RPM, config.RATE_LIMIT_TPM)

SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "character": {"type": "array", "items": {"type": "string"}},
        "emotion": {"type": "array", "items": {"type": "string"}},
        "usage": {"type": "array", "items": {"type": "string"}},
        "background": {"type": "string"},
    },
    "required": ["text", "tags", "character", "emotion", "usage", "background"],
}

VISION_PROMPT = """分析这张表情包/梗图。必须输出 JSON 格式，包含：
- text: 一句中文描述（画面内容 + 梗的含义，直白说清楚）
- tags: 关键词标签（3-8个）
- character: 画面中的角色/人物。知名角色写名字（如"博丽灵梦""特朗普""哆啦A梦"），不知名的写外观特征（如"粉发双马尾少女""戴眼镜的中年男性""橘猫"）
- emotion: 情绪/表情（如"无语""开心""震惊"）
- usage: 使用场景/用途（如"吐槽""怼人""表达无语"）
- background: 理解这张梗图需要的背景知识。比如涉及什么文化梗、游戏梗、时事梗、社区内部梗。只写需要知道什么，不要重复描述画面。如果没有特殊背景知识就写"无"。
只输出 JSON。"""


class AIClient:
    def __init__(self, model="", api_key=None, base_url=None):
        self.model = model
        self.api_key = api_key or config.API_KEY
        self.base_url = (base_url or config.API_BASE).rstrip("/")
        self._client = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _post(self, url, body, token_estimate=0):
        await _rl.acquire(token_estimate)
        client = await self._get_client()
        headers = {"Authorization": f"Bearer {self.api_key}"}
        for attempt in range(config.RETRY_COUNT):
            resp = await client.post(url, json=body, headers=headers)
            if resp.status_code == 200:
                return resp.json()
            data = resp.json()
            if resp.status_code == 429 and attempt < config.RETRY_COUNT - 1:
                retry_after = int(data.get("error", {}).get("retry_after", 2 ** attempt))
                await asyncio.sleep(retry_after)
                continue
            # non-429 or last retry — surface the error immediately
            raise RuntimeError(data.get("error", {}).get("message", str(data)))
        return resp.json()

    async def parse_image(self, image_path: str) -> dict:
        try:
            with open(image_path, "rb") as f:
                data = f.read()
                b64 = base64.b64encode(data).decode()
        except Exception as e:
            return {"error": str(e)}

        if not self.api_key:
            return {"error": "API key not set. Set SILICONFLOW_API_KEY."}
        if not self.model:
            return {"error": "Vision model not configured."}

        # rough token estimate: image ~1k tokens, messages ~200 tokens
        token_est = max(len(data) // 512, 200) + _estimate_tokens(VISION_PROMPT)
        try:
            data = await self._post(f"{self.base_url}/chat/completions", {
                "model": self.model,
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": VISION_PROMPT},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": "Analyze this meme."},
                    ]},
                ],
                "response_format": {"type": "json_schema", "json_schema": {"name": "meme_analysis", "schema": SCHEMA}},
            }, token_estimate=token_est)
            return json.loads(data["choices"][0]["message"]["content"])
        except Exception as e:
            return {"error": str(e)}

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
            raise RuntimeError("API key not set. Set SILICONFLOW_API_KEY.")
        if not self.model:
            raise RuntimeError("Embedding model not configured.")
        if not texts:
            return []

        token_est = sum(_estimate_tokens(t) for t in texts)
        data = await self._post(f"{self.base_url}/embeddings", {
            "model": self.model,
            "input": texts,
        }, token_estimate=token_est)
        # API may return embeddings in arbitrary order; sort by index to align with input order
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_data]
