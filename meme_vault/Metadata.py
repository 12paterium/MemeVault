import datetime, hashlib, json, os
from . import config



class Metadata:
    def __init__(self, path=""):
        self.path = path
        self.id = self.text = self.background = ""
        self.tags = self.character = self.emotion = self.usage = []
        self.analyzed_at = ""
        self.analyzed_by = ""
        if path:
            # content-based ID: same file (even renamed) gets same hash → dedup-friendly
            try:
                with open(path, "rb") as f:
                    self.id = hashlib.md5(f.read()).hexdigest()
            except OSError:
                pass

    @classmethod
    def from_dict(cls, data: dict) -> "Metadata":
        m = cls()
        for field in ["path", "id", "text", "background"]:
            setattr(m, field, data.get(field, "") or "")
        for field in ["tags", "character", "emotion", "usage"]:
            val = data.get(field, [])
            setattr(m, field, val if isinstance(val, list) else [str(val)] if val else [])
        m.analyzed_at = data.get("analyzed_at", "") or ""
        m.analyzed_by = data.get("analyzed_by", "") or ""
        return m

    async def analyze(self, client=None) -> dict | None:
        from .client import AIClient
        if client is None:
            client = AIClient(model=config.VISION_MODEL)
        result = await client.parse_image(self.path)
        if "error" in result:
            return result
        self.text = result.get("text", "") or ""
        self.tags = result.get("tags", [])
        self.character = result.get("character", [])
        self.emotion = result.get("emotion", [])
        self.usage = result.get("usage", [])
        self.background = result.get("background", "") or ""
        self.analyzed_at = datetime.datetime.now().isoformat(timespec="seconds")
        self.analyzed_by = config.VISION_MODEL

    def to_dict(self) -> dict:
        d = {
            "id": self.id or "",
            "path": self.path or "",
            "text": self.text or "",
            "tags": self.tags or [],
            "character": self.character or [],
            "emotion": self.emotion or [],
            "usage": self.usage or [],
        }
        if self.background:
            d["background"] = self.background
        if self.analyzed_at:
            d["analyzed_at"] = self.analyzed_at
        if self.analyzed_by:
            d["analyzed_by"] = self.analyzed_by
        return d

def load_metadata(path: str) -> list[Metadata]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(data, list):
        return []
    return [Metadata.from_dict(item) for item in data if isinstance(item, dict)]

def save_metadata(entries: list[Metadata], path: str):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in entries], f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"Cannot save metadata: {e}")
