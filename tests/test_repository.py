"""Check the public configuration example and Git's actual secret exclusions."""

import os
import shutil
import subprocess
import tempfile
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class RepositoryTests(unittest.TestCase):
    def test_configuration_example_has_no_key_and_disables_live_api(self):
        example = tomllib.loads(
            (ROOT / ".streamlit/secrets.toml.example").read_text(encoding="utf-8")
        )
        self.assertIs(example["ENABLE_LIVE_API"], False)
        self.assertEqual(example["OPENAI_API_KEY"], "")

    @unittest.skipUnless(shutil.which("git"), "Git is required to check ignore rules.")
    def test_git_excludes_local_secrets_but_allows_public_example(self):
        ignored = [
            ".streamlit/secrets.toml", "secrets.toml", "secrets.backup.toml",
            "nested/.streamlit/secrets.toml", ".env", ".env.local", ".env.production",
            "private.key", "private.pem", ".aws/credentials",
        ]
        allowed = [".streamlit/secrets.toml.example", ".env.example", "app.py", "settings.py"]
        with tempfile.TemporaryDirectory() as folder:
            subprocess.run(["git", "init", "--quiet", folder], check=True, capture_output=True)
            shutil.copyfile(ROOT / ".gitignore", Path(folder) / ".gitignore")
            result = subprocess.run(
                ["git", "-c", f"core.excludesFile={os.devnull}", "check-ignore", "--stdin", "-z"],
                input=("\0".join(ignored + allowed) + "\0").encode("utf-8"),
                cwd=folder, capture_output=True, check=True,
            )
        self.assertCountEqual(result.stdout.decode("utf-8").rstrip("\0").split("\0"), ignored)


if __name__ == "__main__":
    unittest.main()
