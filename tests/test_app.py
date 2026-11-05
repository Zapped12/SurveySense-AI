"""Exercise the actual Streamlit interface without using API credentials."""

import json
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from analysis import AnalysisResult

ROOT = Path(__file__).resolve().parents[1]


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict("os.environ", {"ENABLE_LIVE_API": "false"}))

    def app(self):
        return AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()

    @patch("analysis.OpenAI")
    def test_demo_dashboard_filter_and_stale_results(self, client_class):
        app = self.app()
        self.assertFalse(app.exception)
        app.checkbox[0].check().run()
        self.assertEqual(app.radio[0].options, ["Demo (no API)"])
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual([metric.value for metric in app.metric], ["12", "6", "6", "6"])
        self.assertEqual(len(app.dataframe[1].value), 6)
        app.multiselect[0].set_value(["incomplete_answer"]).run()
        self.assertEqual(len(app.dataframe[1].value), 2)
        app.text_area[0].set_value("Changed question context").run()
        self.assertEqual(len(app.metric), 0)
        self.assertTrue(app.button[0].disabled)
        client_class.assert_not_called()

    @patch("analysis.OpenAI")
    def test_missing_key_shows_setup_message(self, client_class):
        with patch.dict("os.environ", {
            "ENABLE_LIVE_API": "true", "OPENAI_API_KEY": "", "OPENAI_MODEL": "",
        }):
            app = self.app()
            app.secrets["OPENAI_API_KEY"] = ""
            app.secrets["OPENAI_MODEL"] = "gpt-4.1-mini"
            app.checkbox[0].check().run()
            app.radio[0].set_value("OpenAI API").run()
            app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertIn("OPENAI_API_KEY", app.error[0].value)
        client_class.assert_not_called()

    @patch("analysis.OpenAI")
    def test_disabled_api_option_stays_hidden_even_with_configured_key(self, client_class):
        app = self.app()
        app.secrets["OPENAI_API_KEY"] = "unused-test-key"
        app.secrets["ENABLE_LIVE_API"] = True
        app.checkbox[0].check().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.radio[0].options, ["Demo (no API)"])
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual([metric.value for metric in app.metric], ["12", "6", "6", "6"])
        client_class.assert_not_called()

    @patch("analysis.OpenAI")
    def test_manipulated_mode_cannot_reach_live_api_when_disabled(self, client_class):
        app = self.app()
        app.secrets["OPENAI_API_KEY"] = "unused-test-key"
        with patch("streamlit.radio", return_value="OpenAI API"):
            app.checkbox[0].check().run()
        self.assertFalse(app.exception)
        self.assertIn("disabled", app.error[0].value)
        self.assertNotIn("analysis_result", app.session_state)
        client_class.assert_not_called()

    @patch("analysis.OpenAI")
    def test_enabled_api_works_and_disabling_clears_live_results(self, client_class):
        client = client_class.return_value.__enter__.return_value
        client.responses.parse.return_value = SimpleNamespace(
            status="completed",
            output_parsed=AnalysisResult.model_validate(
                json.loads((ROOT / "demo_results.json").read_text(encoding="utf-8"))
            ),
        )
        with patch.dict("os.environ", {"ENABLE_LIVE_API": "true"}):
            app = self.app()
            app.secrets["OPENAI_API_KEY"] = "unused-test-key"
            app.checkbox[0].check().run()
            self.assertIn("OpenAI API", app.radio[0].options)
            app.radio[0].set_value("OpenAI API").run()
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.metric), 4)
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.radio[0].options, ["Demo (no API)"])
        self.assertEqual(len(app.metric), 0)
        client.responses.parse.assert_called_once()

    def test_uploaded_values_preserve_id_and_na_text(self):
        source = BytesIO(b"response_id,answer\n001,N/A\n002,Useful\n")
        source.size = len(source.getvalue())
        with patch("streamlit.file_uploader", return_value=source):
            app = self.app()
        self.assertFalse(app.exception)
        self.assertEqual(app.dataframe[0].value.iloc[0].tolist(), ["001", "N/A"])
        self.assertTrue(app.button[0].disabled)


if __name__ == "__main__":
    unittest.main()
