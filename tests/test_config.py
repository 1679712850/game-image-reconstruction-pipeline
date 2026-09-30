"""Configuration validation and strict key handling."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pydantic import ValidationError

from app.config import PipelineConfig, load_config


class ConfigTests(unittest.TestCase):
    def test_yaml_values_are_loaded(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "pipeline.yaml"
            path.write_text("mock: true\nmax_retry: 2\ncrop:\n  padding: 3\n", encoding="utf-8")
            config = load_config(path)
            self.assertEqual(config.max_retry, 2)
            self.assertEqual(config.crop.padding, 3)

    def test_unknown_keys_and_negative_retries_are_rejected(self) -> None:
        for value in ({"max_retries": 2}, {"max_retry": -1}):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                PipelineConfig.model_validate(value)

    def test_failure_injection_requires_mock(self) -> None:
        with self.assertRaises(ValidationError):
            PipelineConfig(mock=False, exercise_retry=True)
