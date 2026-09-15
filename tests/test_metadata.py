import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout

from meme_vault.metadata import (
    IMAGES_DIR,
    LEGACY_METADATA_NAME,
    MEMES_DIR,
    Metadata,
    directory_stamp,
    load_metadata,
    save_metadata,
)


def _entry(identifier, path, text="图"):
    return Metadata.from_dict({"id": identifier, "path": path, "text": text})


def _entry_files(directory):
    return sorted(
        name for name in os.listdir(os.path.join(directory, MEMES_DIR)) if name.endswith(".json")
    )


class MetadataStorageTests(unittest.TestCase):
    def test_save_writes_one_file_per_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            save_metadata([_entry("a", "a.jpg"), _entry("b", "b.jpg")], directory)

            self.assertEqual(_entry_files(directory), ["a.a.json", "b.b.json"])
            self.assertEqual([entry.id for entry in load_metadata(directory)], ["a", "b"])

    def test_external_image_is_copied_into_vault(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "outside", "猫 图.jpg")
            os.makedirs(os.path.dirname(source))
            with open(source, "wb") as file:
                file.write(b"fake-image-bytes")
            vault_dir = os.path.join(directory, "vault")

            save_metadata([_entry("cat", source)], vault_dir)

            with open(os.path.join(vault_dir, MEMES_DIR, "猫 图.cat.json"), encoding="utf-8") as file:
                stored = json.load(file)
            self.assertTrue(stored["path"].startswith(f"{IMAGES_DIR}/"))
            self.assertTrue(os.path.exists(os.path.join(vault_dir, *stored["path"].split("/"))))
            self.assertTrue(os.path.exists(source), "the original image must stay in place")

            loaded = load_metadata(vault_dir)
            self.assertEqual(loaded[0].path, os.path.join(vault_dir, *stored["path"].split("/")))
            self.assertTrue(os.path.exists(loaded[0].path))

    def test_load_migrates_legacy_metadata_json(self):
        with tempfile.TemporaryDirectory() as directory:
            image = os.path.join(directory, "a.jpg")
            with open(image, "wb") as file:
                file.write(b"img")
            with open(os.path.join(directory, LEGACY_METADATA_NAME), "w", encoding="utf-8") as file:
                json.dump([{"id": "a", "path": image, "text": "旧条目"}], file, ensure_ascii=False)

            with redirect_stdout(io.StringIO()) as output:
                entries = load_metadata(directory)

            self.assertEqual([entry.id for entry in entries], ["a"])
            self.assertEqual(entries[0].text, "旧条目")
            self.assertTrue(os.path.isdir(os.path.join(directory, MEMES_DIR)))
            self.assertTrue(os.path.exists(os.path.join(directory, LEGACY_METADATA_NAME + ".bak")))
            self.assertFalse(os.path.exists(os.path.join(directory, LEGACY_METADATA_NAME)))
            self.assertIn("MIGRATED", output.getvalue())

    def test_save_removes_stale_entry_files(self):
        with tempfile.TemporaryDirectory() as directory:
            save_metadata([_entry("a", "a.jpg"), _entry("b", "b.jpg")], directory)
            save_metadata([_entry("a", "a.jpg")], directory)

            self.assertEqual(_entry_files(directory), ["a.a.json"])

    def test_missing_directory_loads_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(load_metadata(os.path.join(directory, "nope")), [])


class IncrementalSaveTests(unittest.TestCase):
    def test_changed_entries_only_write_their_own_files(self):
        with tempfile.TemporaryDirectory() as directory:
            entries = [_entry("a", "a.jpg"), _entry("b", "b.jpg")]
            save_metadata(entries, directory)
            untouched = os.path.join(directory, MEMES_DIR, "b.b.json")
            before = os.stat(untouched).st_mtime_ns

            entries[0].text = "改过的描述"
            save_metadata(entries, directory, changed=[entries[0]])

            self.assertEqual(os.stat(untouched).st_mtime_ns, before)
            self.assertEqual(load_metadata(directory)[0].text, "改过的描述")

    def test_subset_save_still_removes_stale_files(self):
        with tempfile.TemporaryDirectory() as directory:
            entries = [_entry("a", "a.jpg"), _entry("b", "b.jpg")]
            save_metadata(entries, directory)

            save_metadata(entries[:1], directory, changed=[])

            self.assertEqual(_entry_files(directory), ["a.a.json"])

    def test_directory_stamp_changes_with_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            save_metadata([_entry("a", "a.jpg")], directory)
            first = directory_stamp(directory)
            save_metadata([_entry("a", "a.jpg"), _entry("b", "b.jpg")], directory)
            self.assertNotEqual(directory_stamp(directory), first)


class _RecordingVisionClient:
    model = "hint-vision"

    def __init__(self):
        self.filenames = []

    async def parse_image(self, path, filename=None):
        self.filenames.append(filename)
        return {
            "text": "图",
            "tags": [],
            "character": [],
            "emotion": [],
            "usage": [],
            "background": "无",
        }


class AnalyzeHintTests(unittest.IsolatedAsyncioTestCase):
    async def test_analyze_passes_the_original_filename_without_the_vault_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "doro 疑惑 看手机.jpg")
            with open(source, "wb") as file:
                file.write(b"fake-image-bytes")
            client = _RecordingVisionClient()

            entry = Metadata(source)
            await entry.analyze(client)

            vault_dir = os.path.join(directory, "vault")
            save_metadata([entry], vault_dir)
            stored = load_metadata(vault_dir)[0]  # now lives in images/ under <stem>.<id8>.jpg
            await stored.analyze(client)

            self.assertEqual(client.filenames, ["doro 疑惑 看手机", "doro 疑惑 看手机"])
            self.assertEqual(stored.analyzed_by, "hint-vision")
