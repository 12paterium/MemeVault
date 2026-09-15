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
    "additionalProperties": False,
}

MEME_FORMAT = {"type": "json_schema", "json_schema": {"name": "meme_analysis", "schema": SCHEMA}}


def _retry_delay(resp, data, attempt: int) -> float:
    """Seconds to wait before a retry: the server's hint first, then exponential backoff."""
    headers = resp.headers if resp is not None else {}
    for value in (headers.get("Retry-After"), (data or {}).get("error", {}).get("retry_after")):
        try:
            if value is not None:
                return max(1.0, float(value))
        except (TypeError, ValueError):
            continue
    return float(2 ** (attempt + 1))


def _strip_code_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    text = text.split("\n", 1)[1] if "\n" in text else text.strip("`")
    text = text.rstrip()
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def parse_json_object(text: str) -> dict:
    """Dig the first JSON object out of a model reply.

    Clients that cannot request structured output (e.g. AstrBot providers) rely on the
    prompt saying "only JSON"; this tolerates code fences and prose around the object.
    """
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Empty response.")
    body = _strip_code_fence(raw)
    candidates = [body]
    for source in (body, raw):
        start, end = source.find("{"), source.rfind("}")
        if start != -1 and end > start:
            candidates.append(source[start:end + 1])
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    raise ValueError(f"Response is not a JSON object: {raw[:200]}")

VISION_PROMPT = """分析这张表情包/梗图。必须输出 JSON 格式，包含：
- text: 一句中文描述（画面内容 + 梗的含义，直白说清楚）
- tags: 关键词标签（3-8个）
- character: 画面中的角色/人物。知名角色写名字（如"博丽灵梦""特朗普""哆啦A梦"），不知名的写外观特征（如"粉发双马尾少女""戴眼镜的中年男性""橘猫"）
- emotion: 情绪/表情（如"无语""开心""震惊"）
- usage: 使用场景/用途（如"吐槽""怼人""表达无语"）
- background: 理解这张梗图需要的背景知识。比如涉及什么文化梗、游戏梗、时事梗、社区内部梗。认得出出处就写出具体名字（作品名、角色名、事件名），不要停在"某电视剧""某网络片段"这类模糊说法；只写需要知道什么，不要重复描述画面。如果没有特殊背景知识就写"无"。
用户消息里可能附上图片的原始文件名，它只是上传者的描述，可能不准确、也可能与画面无关。文件名只能用来补充画面里看不出的使用语境（例如"早八""看手机"这类场合）；绝对不要用文件名去推断角色、人物或出处——画面认不出就按外观描述，宁可写"不知名"也不要编造；画面与文件名冲突时以画面为准，也不要把文件名原样抄进 text。
只输出 JSON。"""


class AIClient:
    def __init__(self, model="", api_key=None, base_url=None, timeout=120.0):
        self.model = model
        self.api_key = api_key or config.API_KEY
        self.base_url = (base_url or config.API_BASE).rstrip("/")
        self.timeout = timeout
        self._client = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout))
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _post(self, url, body):
        client = await self._get_client()
        headers = {"Authorization": f"Bearer {self.api_key}"}
        last_attempt = config.RETRY_COUNT - 1
        for attempt in range(config.RETRY_COUNT):
            try:
                resp = await client.post(url, json=body, headers=headers)
            except httpx.HTTPError as error:
                # network errors are transient too: retry them like a 5xx
                if attempt < last_attempt:
                    await asyncio.sleep(_retry_delay(None, None, attempt))
                    continue
                raise RuntimeError(str(error) or error.__class__.__name__) from error
            if resp.status_code == 200:
                return resp.json()
            try:
                data = resp.json()
            except ValueError:
                data = {"error": {"message": resp.text[:200]}}
            if resp.status_code in config.RETRY_STATUSES and attempt < last_attempt:
                await asyncio.sleep(_retry_delay(resp, data, attempt))
                continue
            message = data.get("error", {}).get("message") or resp.text[:200].strip()
            raise RuntimeError(f"{message or 'request failed'} (HTTP {resp.status_code})")
        return resp.json()

    async def chat(self, messages, temperature=0.2, response_format=None,
                  **kwargs) -> str:
        if not self.api_key:
            raise RuntimeError("API key not set. Set SILICONFLOW_API_KEY.")
        if not self.model:
            raise RuntimeError("Model not configured.")
        body = {"model": self.model, "messages": messages,
                "temperature": temperature, **kwargs}
        if response_format is not None:
            body["response_format"] = response_format
        data = await self._post(f"{self.base_url}/chat/completions", body)
        return data["choices"][0]["message"]["content"]

    async def parse_image(self, image_path: str, filename: str = "") -> dict:
        try:
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
        except Exception as e:
            return {"error": str(e)}

        if not self.api_key:
            return {"error": "API key not set. Set SILICONFLOW_API_KEY."}
        if not self.model:
            return {"error": "Vision model not configured."}

        hint = "分析这张表情包。"
        if filename:
            hint += f"\n文件名：{filename}"

        try:
            text = await self.chat(
                messages=[
                    {"role": "system", "content": VISION_PROMPT},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": hint},
                    ]},
                ],
                response_format=MEME_FORMAT,
            )
            return json.loads(text)
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

    async def rerank(self, query: str, documents: list, top_n: int = None) -> list[tuple[int, float]]:
        """Score documents against a query. Returns (original index, score) sorted best-first."""
        if not self.api_key:
            raise RuntimeError("API key not set. Set SILICONFLOW_API_KEY.")
        if not self.model:
            raise RuntimeError("Rerank model not configured.")
        if not documents:
            return []
        body = {"model": self.model, "query": query, "documents": documents}
        if top_n is not None:
            body["top_n"] = top_n
        data = await self._post(f"{self.base_url}/rerank", body)
        return [(item["index"], item["relevance_score"]) for item in data["results"]]
