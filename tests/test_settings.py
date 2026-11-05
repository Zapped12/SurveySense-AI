"""Live API access must require an explicit server-side opt-in."""

import unittest
from unittest.mock import patch

from streamlit.errors import StreamlitSecretNotFoundError

from settings import get_setting, live_api_enabled


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict("os.environ", {"ENABLE_LIVE_API": ""}))
        self.secrets = {}
        self.enterContext(patch("settings.st.secrets", self.secrets))

    def test_default_is_disabled_with_or_without_an_api_key(self):
        self.assertFalse(live_api_enabled())
        self.secrets["OPENAI_API_KEY"] = "unused-test-key"
        self.assertFalse(live_api_enabled())

    def test_only_explicit_true_values_enable_live_api(self):
        for value in [False, "false", "", "0", 0, 1, "1", "yes", "invalid", None, [], {}]:
            with self.subTest(value=value):
                self.secrets["ENABLE_LIVE_API"] = value
                self.assertFalse(live_api_enabled())
        for value in [True, "true", "TRUE", " true "]:
            with self.subTest(value=value):
                self.secrets["ENABLE_LIVE_API"] = value
                self.assertTrue(live_api_enabled())

    def test_environment_overrides_secrets_in_both_directions(self):
        self.secrets["ENABLE_LIVE_API"] = True
        with patch.dict("os.environ", {"ENABLE_LIVE_API": "false"}):
            self.assertFalse(live_api_enabled())
        self.secrets["ENABLE_LIVE_API"] = False
        with patch.dict("os.environ", {"ENABLE_LIVE_API": "true"}):
            self.assertTrue(live_api_enabled())

    def test_missing_secrets_file_defaults_to_disabled(self):
        for error in [FileNotFoundError(), StreamlitSecretNotFoundError("No secrets")]:
            with self.subTest(error=type(error).__name__), patch("settings.st.secrets") as secrets:
                secrets.get.side_effect = error
                self.assertFalse(live_api_enabled())

    def test_text_configuration_keeps_existing_precedence_and_validation(self):
        self.secrets["OPENAI_MODEL"] = "from-secrets"
        with patch.dict("os.environ", {"OPENAI_MODEL": " from-environment "}):
            self.assertEqual(get_setting("OPENAI_MODEL"), "from-environment")
        with patch.dict("os.environ", {"OPENAI_MODEL": ""}):
            self.assertEqual(get_setting("OPENAI_MODEL"), "from-secrets")
            self.secrets["OPENAI_MODEL"] = True
            with self.assertRaises(ValueError):
                get_setting("OPENAI_MODEL")


if __name__ == "__main__":
    unittest.main()
