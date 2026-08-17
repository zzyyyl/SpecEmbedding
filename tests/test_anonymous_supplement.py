import tempfile
import unittest
from pathlib import Path

from reproducibility.build_anonymous_supplement import (
    AnonymousArchiveError,
    build_archive,
    load_allowlist,
    scan_entries,
    sha256_file,
    verify_archive,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = REPOSITORY_ROOT / "reproducibility" / "anonymous-allowlist.txt"


class AnonymousSupplementTest(unittest.TestCase):
    def test_repository_allowlist_builds_deterministically(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.zip"
            second = root / "second.zip"
            result = build_archive(
                repository_root=REPOSITORY_ROOT,
                allowlist_path=ALLOWLIST,
                output_path=first,
                deny_tokens=[],
            )
            build_archive(
                repository_root=REPOSITORY_ROOT,
                allowlist_path=ALLOWLIST,
                output_path=second,
                deny_tokens=[],
            )

            self.assertGreater(result["files"], 30)
            self.assertEqual(sha256_file(first), sha256_file(second))
            self.assertEqual(
                verify_archive(first, deny_tokens=[])["sha256"],
                sha256_file(first),
            )

    def test_repository_allowlist_passes_static_identity_scan(self):
        entries = load_allowlist(REPOSITORY_ROOT, ALLOWLIST)
        scan_entries(entries, deny_tokens=[])

    def test_private_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "unsafe.txt").write_text(
                "private=" + "/home/" + "example/project\n",
                encoding="utf-8",
            )
            allowlist = root / "allowlist.txt"
            allowlist.write_text("unsafe.txt\n", encoding="utf-8")

            with self.assertRaisesRegex(AnonymousArchiveError, "identity scan failed"):
                build_archive(
                    repository_root=root,
                    allowlist_path=allowlist,
                    output_path=root / "unsafe.zip",
                    deny_tokens=[],
                )

    def test_forbidden_artifact_type_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model.pth").write_bytes(b"weights")
            allowlist = root / "allowlist.txt"
            allowlist.write_text("model.pth\n", encoding="utf-8")

            with self.assertRaisesRegex(AnonymousArchiveError, "Forbidden archive file type"):
                load_allowlist(root, allowlist)

    def test_additional_identity_patterns_are_rejected(self):
        unsafe_values = {
            "Windows user path": "C:\\Users\\example\\project",
            "GPU UUID": "GPU-12345678-1234-1234-1234-123456789ABC",
            "ORCID": "0000-0002-1825-0097",
            "hosting link": "https://github.com/example/project",
        }
        for label, unsafe_value in unsafe_values.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "unsafe.txt").write_text(unsafe_value, encoding="utf-8")
                allowlist = root / "allowlist.txt"
                allowlist.write_text("unsafe.txt\n", encoding="utf-8")

                with self.assertRaisesRegex(AnonymousArchiveError, "identity scan failed"):
                    build_archive(
                        repository_root=root,
                        allowlist_path=allowlist,
                        output_path=root / "unsafe.zip",
                        deny_tokens=[],
                    )

    def test_internal_inventory_destination_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "inventory.yaml").write_text("files: []\n", encoding="utf-8")
            allowlist = root / "allowlist.txt"
            allowlist.write_text(
                "inventory.yaml -> runtime-snapshot.yaml\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(AnonymousArchiveError, "inventory file"):
                load_allowlist(root, allowlist)

    def test_windows_style_archive_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "safe.txt").write_text("safe\n", encoding="utf-8")
            allowlist = root / "allowlist.txt"
            allowlist.write_text(
                "safe.txt -> ..\\outside.txt\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(AnonymousArchiveError, "Unsafe archive destination"):
                load_allowlist(root, allowlist)


if __name__ == "__main__":
    unittest.main()
