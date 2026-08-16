import hashlib
import subprocess
import unittest
from pathlib import Path
from unittest import mock

import freeze_adma2026_artifacts as freeze


class FreezeArtifactProvenanceTest(unittest.TestCase):
    def test_git_blob_artifact_records_historical_content(self):
        payload = b"data:\n  path: /historical/location\n"
        completed = subprocess.CompletedProcess(
            args=["git", "show"], returncode=0, stdout=payload, stderr=b""
        )
        with mock.patch.object(subprocess, "run", return_value=completed) as run:
            record = freeze.git_blob_artifact(
                "config.params", "source-commit", "params.yaml"
            )

        self.assertEqual(record["location"], "git:source-commit:params.yaml")
        self.assertEqual(record["sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(record["size_bytes"], len(payload))
        run.assert_called_once_with(
            ["git", "show", "source-commit:params.yaml"],
            cwd=freeze.REPO_ROOT,
            capture_output=True,
            check=True,
        )

    def test_git_blob_artifact_rejects_escaping_path(self):
        with self.assertRaises(freeze.AuditError):
            freeze.git_blob_artifact("config.params", "source-commit", "../params.yaml")

    def test_manifest_uses_source_commit_params_blob(self):
        historical = {
            "id": "config.params",
            "location": f"git:{freeze.SOURCE_COMMIT}:params.yaml",
            "sha256": freeze.PARAMS_SHA256,
            "size_bytes": 123,
        }
        with (
            mock.patch.object(freeze, "require_source_commit"),
            mock.patch.object(
                freeze, "git_blob_artifact", return_value=historical
            ) as git_blob,
            mock.patch.object(freeze, "validate_alignment", return_value=Path("data")),
            mock.patch.object(freeze, "validate_data"),
            mock.patch.object(freeze, "validate_multiseed", return_value={}),
            mock.patch.object(freeze, "validate_mces", return_value={}),
        ):
            manifest = freeze.build_manifest()

        git_blob.assert_called_once_with(
            "config.params", freeze.SOURCE_COMMIT, "params.yaml"
        )
        self.assertEqual(manifest["artifacts"], [historical])
        self.assertIn("source_commit", manifest["provenance_policy"])


if __name__ == "__main__":
    unittest.main()
