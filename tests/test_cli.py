import unittest

from meme_vault.cli import (
    DEFAULT_DATA_DIR,
    _collect_weights,
    _parser,
    _structured_query,
)


class CliQueryTests(unittest.TestCase):
    def test_weight_without_structured_field_is_rejected(self):
        args = _parser().parse_args([
            "search",
            "自然语言查询",
            "--content-weight",
            "5",
        ])

        with self.assertRaisesRegex(ValueError, "structured query field"):
            _structured_query(args)

    def test_structured_query_keeps_weight_overrides(self):
        args = _parser().parse_args([
            "search",
            "--character",
            "初音未来",
            "--usage",
            "吐槽",
            "--usage-weight",
            "4",
        ])

        query = _structured_query(args)

        self.assertEqual(query.character, "初音未来")
        self.assertEqual(query.usage, "吐槽")
        self.assertEqual(_collect_weights(args), {"usage": 4.0})

    def test_new_command_parsers(self):
        parser = _parser()
        self.assertEqual(parser.parse_args(["prune"]).command, "prune")
        self.assertEqual(parser.parse_args(["status"]).command, "status")


class DataDirTests(unittest.TestCase):
    def test_data_dir_flag_and_default(self):
        parser = _parser()
        self.assertEqual(parser.parse_args(["--data-dir", "custom", "status"]).data_dir, "custom")
        self.assertEqual(parser.parse_args(["status"]).data_dir, DEFAULT_DATA_DIR)
