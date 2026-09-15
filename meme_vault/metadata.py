import datetime
import hashlib
import json
import os
import shutil
import uuid

from . import config


MEMES_DIR = "memes"
IMAGES_DIR = "images"
LEGACY_METADATA_NAME = "metadata.json"
REVISION_NAME = ".revision"


class Metadata:
    def __init__(self, path=""):
        self.path = path
        self.id = self.text = self.background = ""
        self.tags = []
        self.character = []
        self.emotion = []
        self.usage = []
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
        result = await client.parse_image(self.path, filename=image_stem(self))
        if "error" in result:
            return result
        self.text = result.get("text", "") or ""
        self.tags = result.get("tags", [])
        self.character = result.get("character", [])
        self.emotion = result.get("emotion", [])
        self.usage = result.get("usage", [])
        self.background = result.get("background", "") or ""
        self.analyzed_at = datetime.datetime.now().isoformat(timespec="seconds")
        self.analyzed_by = client.model or config.VISION_MODEL

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


def memes_dir(data_dir: str) -> str:
    return os.path.join(data_dir, MEMES_DIR)


def images_dir(data_dir: str) -> str:
    return os.path.join(data_dir, IMAGES_DIR)


def _is_inside(path: str, directory: str) -> bool:
    try:
        return os.path.commonpath([os.path.abspath(path), os.path.abspath(directory)]) == os.path.abspath(directory)
    except ValueError:
        return False


def _file_suffix(entry: Metadata) -> str:
    return entry.id[:8] if entry.id else "unknown"


def image_stem(entry: Metadata) -> str:
    """Original image name without extension; ignores the vault id suffix when present."""
    stem = os.path.splitext(os.path.basename(entry.path))[0].strip() or "meme"
    suffix = _file_suffix(entry)
    if stem.endswith(f".{suffix}"):
        stem = stem[: -(len(suffix) + 1)]
    return stem


def _entry_filename(entry: Metadata) -> str:
    """<image stem>.<id8>.json; idempotent, so re-saving keeps the same name."""
    return f"{image_stem(entry)}.{_file_suffix(entry)}.json"


def _vault_image_path(entry: Metadata, data_dir: str) -> str | None:
    """Copy the entry image into <data_dir>/images and return its data-dir-relative path."""
    if not entry.path:
        return None
    absolute = os.path.abspath(entry.path)
    if _is_inside(absolute, data_dir):
        return os.path.relpath(absolute, data_dir).replace(os.sep, "/")
    if not os.path.isfile(absolute):
        return None
    stem = os.path.splitext(os.path.basename(absolute))[0].strip() or "meme"
    extension = os.path.splitext(absolute)[1].lower()
    relative = f"{IMAGES_DIR}/{stem}.{_file_suffix(entry)}{extension}"
    target = os.path.join(data_dir, *relative.split("/"))
    os.makedirs(os.path.dirname(target), exist_ok=True)
    if not os.path.exists(target):
        shutil.copy2(absolute, target)
    return relative


def _resolve_path(path: str, data_dir: str) -> str:
    if not path or os.path.isabs(path):
        return path
    return os.path.join(data_dir, *path.split("/"))


def _write_json(path: str, payload: dict):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def _revision_path(data_dir: str) -> str:
    return os.path.join(memes_dir(data_dir), REVISION_NAME)


def _bump_revision(data_dir: str):
    """Mark the vault as changed; the value is unique so file clocks cannot hide a write."""
    try:
        with open(_revision_path(data_dir), "w", encoding="utf-8") as file:
            file.write(uuid.uuid4().hex)
    except OSError:
        pass


def _read_revision(data_dir: str) -> str:
    try:
        with open(_revision_path(data_dir), encoding="utf-8") as file:
            return file.read().strip()
    except OSError:
        return ""


def directory_stamp(data_dir: str) -> tuple:
    """Cheap change detector for <data_dir>/memes: revision marker, file count, newest mtime."""
    target = memes_dir(data_dir)
    count = 0
    newest = 0
    try:
        with os.scandir(target) as items:
            for item in items:
                if item.name.endswith(".json"):
                    count += 1
                    mtime = item.stat().st_mtime_ns
                    if mtime > newest:
                        newest = mtime
        directory_mtime = os.stat(target).st_mtime_ns
    except OSError:
        return (0, 0, 0, "")
    return (count, newest, directory_mtime, _read_revision(data_dir))


def save_metadata(entries: list[Metadata], data_dir: str, changed: list[Metadata] | None = None):
    """One JSON file per entry under <data_dir>/memes; images are copied into <data_dir>/images.

    changed=None rewrites every entry file; pass the touched entries to write only those.
    """
    try:
        os.makedirs(memes_dir(data_dir), exist_ok=True)
        expected = {_entry_filename(entry) for entry in entries}
        for entry in entries if changed is None else changed:
            relative = _vault_image_path(entry, data_dir)
            payload = entry.to_dict()
            if relative:
                entry.path = os.path.join(data_dir, *relative.split("/"))
                payload["path"] = relative
            _write_json(os.path.join(memes_dir(data_dir), _entry_filename(entry)), payload)
        for filename in os.listdir(memes_dir(data_dir)):
            if filename.endswith(".json") and filename not in expected:
                os.remove(os.path.join(memes_dir(data_dir), filename))
        _bump_revision(data_dir)
    except OSError as error:
        print(f"Cannot save metadata: {error}")


def _migrate_legacy(data_dir: str):
    """One-time migration: a legacy metadata.json becomes memes/ (+ images/)."""
    legacy = os.path.join(data_dir, LEGACY_METADATA_NAME)
    if not os.path.isfile(legacy):
        return
    try:
        with open(legacy, "r", encoding="utf-8-sig") as file:
            data = json.load(file)
    except (json.JSONDecodeError, OSError):
        return
    entries = [Metadata.from_dict(item) for item in data if isinstance(item, dict)] if isinstance(data, list) else []
    save_metadata(entries, data_dir)
    os.replace(legacy, legacy + ".bak")
    print(f"  MIGRATED: {len(entries)} entries -> {MEMES_DIR}/ and {IMAGES_DIR}/ (legacy file kept as {LEGACY_METADATA_NAME}.bak)")


def load_metadata(data_dir: str) -> list[Metadata]:
    """Load every entry from <data_dir>/memes, migrating a legacy metadata.json on first use."""
    if not os.path.isdir(data_dir):
        return []
    _migrate_legacy(data_dir)
    target = memes_dir(data_dir)
    if not os.path.isdir(target):
        return []
    entries = []
    seen = set()
    for filename in sorted(os.listdir(target)):
        if not filename.endswith(".json"):
            continue
        try:
            with open(os.path.join(target, filename), "r", encoding="utf-8-sig") as file:
                data = json.load(file)
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        entry = Metadata.from_dict(data)
        if entry.id and entry.id in seen:
            print(f"  SKIP duplicate entry: {filename}")
            continue
        seen.add(entry.id)
        entry.path = _resolve_path(entry.path, data_dir)
        entries.append(entry)
    return entries
