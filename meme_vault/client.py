import asyncio, base64, json
import httpx

from . import config

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

    async def _post(self, url, body):
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
            })
            return json.loads(data["choices"][0]["message"]["content"])
        except Exception as e:
            return {"error": str(e)}

    async def embed(self, inputs: list) -> list[list[float]]:
        if not self.api_key:
            raise RuntimeError("API key not set. Set SILICONFLOW_API_KEY.")
        if not self.model:
            raise RuntimeError("Embedding model not configured.")
        if not inputs:
            return []
        data = await self._post(f"{self.base_url}/embeddings", {
            "model": self.model,
            "input": inputs,
        })
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_data]
