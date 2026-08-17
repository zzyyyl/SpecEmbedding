import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from reproducibility.build_anonymous_supplement import (
    AnonymousArchiveError,
    build_archive,
    load_allowlist,
    scan_entries,
    sha256_bytes,
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

    def test_unknown_binary_type_is_rejected_instead_of_skipping_scan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "unsafe.dat").write_bytes(b"\xff/private/home/path")
            allowlist = root / "allowlist.txt"
            allowlist.write_text("unsafe.dat\n", encoding="utf-8")

            with self.assertRaisesRegex(AnonymousArchiveError, "Unsupported archive file type"):
                load_allowlist(root, allowlist)

    def test_figshare_doi_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "unsafe.txt").write_text(
                "https://doi.org/10.6084/m9.figshare.12345678",
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

    def test_verify_rejects_parent_traversal_member(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "unsafe.zip"
            payload = b"safe\n"
            relative = "../escape.txt"
            member = f"specembedding-supplement/{relative}"
            manifest_member = "specembedding-supplement/ANONYMOUS_MANIFEST.json"
            manifest = json.dumps(
                {
                    "schema_version": 1,
                    "files": [
                        {
                            "path": relative,
                            "bytes": len(payload),
                            "sha256": sha256_bytes(payload),
                            "mode": "0644",
                        }
                    ],
                },
                sort_keys=True,
            ).encode()
            with zipfile.ZipFile(archive_path, mode="w") as archive:
                for name, data in sorted(
                    ((member, payload), (manifest_member, manifest))
                ):
                    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                    info.create_system = 3
                    archive.writestr(info, data)

            with self.assertRaisesRegex(AnonymousArchiveError, "Unsafe archive member"):
                verify_archive(archive_path, deny_tokens=[])


if __name__ == "__main__":
    unittest.main()
