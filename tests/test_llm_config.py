import json
import os
import tempfile
import unittest

from drone_app.llm_clients import (
    DummyClient,
    create_llm_client,
    get_model_choices,
    load_llm_config,
    resolve_llm_settings,
)


class TestLLMConfig(unittest.TestCase):
    def test_loads_flat_llm_config(self):
        path = self._write_config({"service": "dummy", "model": "dummy-from-json"})

        config = load_llm_config(path)

        self.assertEqual(config["service"], "dummy")
        self.assertEqual(config["model"], "dummy-from-json")

    def test_resolve_prefers_explicit_arguments_over_config(self):
        path = self._write_config({"service": "gemini", "model": "gemini-test"})

        settings = resolve_llm_settings(
            service="dummy",
            model_name="dummy-explicit",
            config_path=path,
        )

        self.assertEqual(settings["service"], "dummy")
        self.assertEqual(settings["model"], "dummy-explicit")

    def test_explicit_service_does_not_inherit_different_config_model(self):
        path = self._write_config({"service": "gemini", "model": "gemini-test"})

        settings = resolve_llm_settings(service="dummy", config_path=path)

        self.assertEqual(settings["service"], "dummy")
        self.assertIsNone(settings["model"])

    def test_create_dummy_client_from_config(self):
        path = self._write_config({"llm": {"service": "dummy", "model": "dummy-json"}})

        client = create_llm_client(config_path=path)

        self.assertIsInstance(client, DummyClient)
        self.assertEqual(client.model_name, "dummy-json")

    def test_missing_optional_config_uses_default_service(self):
        settings = resolve_llm_settings(config_path="missing_llm_config_for_test.json")

        self.assertEqual(settings["service"], "gemini")
        self.assertIsNone(settings["model"])

    def test_model_choices_include_configured_custom_model(self):
        choices = get_model_choices("gemini", "custom-gemini-model")

        self.assertEqual(choices[0], "custom-gemini-model")
        self.assertIn("gemini-2.5-flash", choices)

    def test_model_choices_for_dummy(self):
        self.assertEqual(get_model_choices("dummy"), ["dummy-model"])

    def _write_config(self, payload):
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as config_file:
            json.dump(payload, config_file)
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        return path


if __name__ == "__main__":
    unittest.main()
