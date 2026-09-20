from __future__ import annotations

import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

import pyannote_model_manager as pmm


class _FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes, content_type: str = "application/zip"):
        super().__init__(payload)
        self.headers = {
            "Content-Type": content_type,
            "Content-Length": str(len(payload)),
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


class PyannoteMirrorTests(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("JERONIMO_PORTABLE_ROOT", None)
        os.environ.pop("JERONIMO_PYANNOTE_MODEL_DIR", None)
        os.environ.pop("JERONIMO_PYANNOTE_RELEASE_FILE", None)

    def _canonical_zip(self, revision: str) -> bytes:
        files = {
            "README.md": b"upstream readme",
            "config.yaml": b"pipeline:\n  name: test\n",
            "embedding/README.md": b"embedding notice",
            "embedding/pytorch_model.bin": b"embedding-model",
            "segmentation/pytorch_model.bin": b"segmentation-model",
            "plda/README.md": b"plda notice",
            "plda/plda.npz": b"plda-data",
            "plda/xvec_transform.npz": b"xvec-data",
        }
        manifest = {
            "schema": 1,
            "artifact_id": pmm.ARTIFACT_ID,
            "source": {
                "provider": "Hugging Face",
                "repo_id": pmm.OFFICIAL_REPO_ID,
                "revision": revision,
                "declared_license": pmm.EXPECTED_LICENSE,
            },
            "packaging": {"payload_dir": pmm.PAYLOAD_DIRNAME, "payload_modified": False},
            "files": [
                {
                    "path": name,
                    "size": len(data),
                    "sha256": __import__("hashlib").sha256(data).hexdigest(),
                }
                for name, data in sorted(files.items())
            ],
        }
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for name, data in files.items():
                zf.writestr(f"{pmm.PAYLOAD_DIRNAME}/{name}", data)
            zf.writestr("JERONIMO_MODEL_MANIFEST.json", json.dumps(manifest))
            zf.writestr("THIRD_PARTY_NOTICES.md", "CC BY 4.0")
        return out.getvalue()

    def _write_descriptor(self, root: Path, payload: bytes, revision: str) -> Path:
        import hashlib
        descriptor = {
            "schema": 1,
            "enabled": True,
            "artifact_id": pmm.ARTIFACT_ID,
            "source_repo_id": pmm.OFFICIAL_REPO_ID,
            "revision": revision,
            "archive_name": f"jeronimo-pyannote-community-1-{revision[:12]}.zip",
            "archive_sha256": hashlib.sha256(payload).hexdigest(),
            "archive_size_bytes": len(payload),
            "payload_dir": pmm.PAYLOAD_DIRNAME,
            "drive_file_id": "1AbCdEfGhIjKlMnOpQrStUvWxYz12345",
            "drive_share_url": "https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQrStUvWxYz12345/view?usp=sharing",
        }
        path = root / pmm.RELEASE_DESCRIPTOR
        path.write_text(json.dumps(descriptor), encoding="utf-8")
        return path

    def test_parse_google_drive_ids(self):
        file_id = "1AbCdEfGhIjKlMnOpQrStUvWxYz12345"
        self.assertEqual(pmm.parse_google_drive_file_id(file_id), file_id)
        self.assertEqual(
            pmm.parse_google_drive_file_id(f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"),
            file_id,
        )
        self.assertEqual(
            pmm.parse_google_drive_file_id(f"https://drive.google.com/open?id={file_id}"),
            file_id,
        )

    def test_drive_artifact_is_verified_and_installed_locally(self):
        revision = "a" * 40
        payload = self._canonical_zip(revision)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            os.environ["JERONIMO_PORTABLE_ROOT"] = str(root)
            self._write_descriptor(root, payload, revision)
            with mock.patch.object(pmm.urllib.request, "urlopen", return_value=_FakeResponse(payload)), \
                 mock.patch.object(pmm, "_token_candidates", return_value=[]):
                result = pmm.ensure_pyannote_model()
            self.assertTrue(result["ok"])
            self.assertEqual(result["source"], "jeronimo-drive")
            target = root / "runtime" / "models" / "pyannote" / "community-1"
            self.assertTrue((target / "config.yaml").is_file())
            self.assertTrue((target / "JERONIMO_MODEL_MANIFEST.json").is_file())
            self.assertTrue(os.path.samefile(os.environ.get(pmm.MODEL_ENV), target))
            status = pmm.local_model_status(verify_hashes=True)
            self.assertTrue(status["ready"], status)
            self.assertEqual(status["revision"], revision)

    def test_corrupt_drive_payload_is_rejected(self):
        revision = "b" * 40
        good = self._canonical_zip(revision)
        bad = good + b"corruption"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            os.environ["JERONIMO_PORTABLE_ROOT"] = str(root)
            self._write_descriptor(root, good, revision)
            with mock.patch.object(pmm.urllib.request, "urlopen", return_value=_FakeResponse(bad)), \
                 mock.patch.object(pmm, "_token_candidates", return_value=[]):
                with self.assertRaises(pmm.PyannoteModelError):
                    pmm.ensure_pyannote_model()
            self.assertFalse((root / "runtime" / "models" / "pyannote" / "community-1").exists())

    def test_builder_carries_mirror_manager_and_optional_descriptor(self):
        root = Path(__file__).resolve().parents[1]
        ps = (root / "build" / "build_portable.ps1").read_text(encoding="utf-8")
        spec = (root / "build" / "Jeronimo.spec").read_text(encoding="utf-8")
        self.assertIn('"pyannote_model_manager.py"', ps)
        self.assertIn('"pyannote_mirror_release.json"', ps)
        self.assertIn("'pyannote_model_manager'", spec)

    def test_core_can_use_local_community_model_without_hf_token(self):
        root = Path(__file__).resolve().parents[1]
        src = (root / "core_transcriber.py").read_text(encoding="utf-8")
        self.assertIn('JERONIMO_PYANNOTE_MODEL_DIR', src)
        self.assertIn('local_pyannote_ready', src)
        self.assertIn('When Community-1', src.replace('Cuando Community-1', 'When Community-1'))


if __name__ == "__main__":
    unittest.main()
