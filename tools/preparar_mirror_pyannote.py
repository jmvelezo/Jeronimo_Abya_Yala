#!/usr/bin/env python3
"""Prepara un artefacto canónico y verificable de pyannote Community-1.

Uso de mantenimiento (NO se ejecuta en PCs de usuarios finales):
    .venv\\Scripts\\python.exe tools\\preparar_mirror_pyannote.py

Objetivos:
- descargar EXCLUSIVAMENTE el repositorio oficial gated de pyannote;
- fijar la revisión exacta (commit SHA) antes de descargar;
- conservar README y avisos de procedencia;
- copiar el payload byte-a-byte a una carpeta limpia;
- calcular SHA-256 por archivo;
- producir un ZIP versionado + SHA-256 del ZIP para alojarlo en un mirror propio.

La credencial Hugging Face nunca se imprime ni se incluye en el artefacto.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from credential_store import resolve_secret  # noqa: E402
from team_credentials import get_embedded_team_hf_token  # noqa: E402

REPO_ID = "pyannote/speaker-diarization-community-1"
EXPECTED_LICENSE = "cc-by-4.0"
PAYLOAD_DIRNAME = "pyannote-speaker-diarization-community-1"

# Archivos indispensables según el layout oficial de Community-1.
REQUIRED_FILES = {
    "README.md",
    "config.yaml",
    "embedding/README.md",
    "embedding/pytorch_model.bin",
    "segmentation/pytorch_model.bin",
    "plda/README.md",
    "plda/plda.npz",
    "plda/xvec_transform.npz",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def get_token() -> tuple[str, str]:
    # 1) sesión/entorno/keyring; 2) credencial interna del equipo.
    token = str(resolve_secret("huggingface") or "").strip()
    if token:
        return token, "configured"
    token = str(get_embedded_team_hf_token() or "").strip()
    if token:
        return token, "team-embedded"
    return "", "missing"


def model_license(info: Any) -> str:
    card = getattr(info, "card_data", None)
    if card is None:
        return ""
    if isinstance(card, dict):
        return str(card.get("license") or "").strip().lower()
    value = getattr(card, "license", "")
    return str(value or "").strip().lower()


def copy_snapshot(snapshot: Path, payload: Path) -> None:
    payload.mkdir(parents=True, exist_ok=True)
    for src in sorted(snapshot.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(snapshot)
        # Metadatos internos del cache de HF no forman parte del modelo.
        if any(part in {".cache", ".git"} for part in rel.parts):
            continue
        dst = payload / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst, follow_symlinks=True)


def build_manifest(payload: Path, revision: str, license_id: str) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted(payload.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(payload).as_posix()
        files.append({
            "path": rel,
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return {
        "schema": 1,
        "artifact_id": "jeronimo-pyannote-community-1",
        "source": {
            "provider": "Hugging Face",
            "repo_id": REPO_ID,
            "revision": revision,
            "declared_license": license_id,
        },
        "packaging": {
            "payload_dir": PAYLOAD_DIRNAME,
            "payload_modified": False,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
        "files": files,
    }


def write_notices(path: Path, revision: str, license_id: str) -> None:
    text = (
        "# Third-party notices — pyannote Community-1\n\n"
        f"Source repository: `{REPO_ID}`\n\n"
        f"Frozen upstream revision: `{revision}`\n\n"
        f"Repository-declared license: `{license_id}`\n\n"
        "This Jerónimo mirror package does not modify the model payload. "
        "It only repackages the exact files downloaded from the pinned upstream revision "
        "for local/offline installation.\n\n"
        "The upstream `README.md` and component README files are preserved inside the payload "
        "and must remain distributed with it. In particular, the embedding README records "
        "WeSpeaker/VoxCeleb provenance and attribution information.\n\n"
        "License reference: Creative Commons Attribution 4.0 International (CC BY 4.0).\n"
        "Upstream model card and citations remain authoritative for acknowledgments and scientific references.\n"
    )
    path.write_text(text, encoding="utf-8")


def build_zip(staging: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(staging).as_posix())


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepara el mirror verificable de pyannote Community-1.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "tools" / "_mirror_output",
        help="Directorio donde se generarán ZIP, manifest y .sha256.",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    token, token_source = get_token()
    if not token:
        print("ERROR: no hay una credencial Hugging Face disponible para obtener el release oficial.", file=sys.stderr)
        return 2

    try:
        from huggingface_hub import HfApi, snapshot_download  # type: ignore
    except Exception as exc:
        print(f"ERROR: huggingface_hub no está disponible: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3

    print("Jerónimo Abya Yala — preparación de mirror Community-1")
    print(f"Repositorio oficial: {REPO_ID}")
    print(f"Credencial: disponible ({token_source}); no se mostrará.")
    print("Consultando revisión oficial...")

    api = HfApi()
    try:
        info = api.model_info(REPO_ID, token=token)
    except Exception as exc:
        print(f"ERROR: no se pudo consultar el repositorio oficial: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 4

    revision = str(getattr(info, "sha", "") or "").strip()
    if len(revision) < 20:
        print("ERROR: Hugging Face no devolvió una revisión inmutable válida.", file=sys.stderr)
        return 5

    license_id = model_license(info)
    if license_id != EXPECTED_LICENSE:
        print(
            f"ERROR: la licencia declarada cambió o no pudo verificarse. "
            f"Esperada={EXPECTED_LICENSE!r}, recibida={license_id!r}. Revisión manual obligatoria.",
            file=sys.stderr,
        )
        return 6

    print(f"Revisión congelada: {revision}")
    print(f"Licencia declarada: {license_id}")

    with tempfile.TemporaryDirectory(prefix="jeronimo_pyannote_mirror_") as td:
        tmp = Path(td)
        print("Descargando snapshot oficial fijado por SHA...")
        try:
            snapshot_path = Path(
                snapshot_download(
                    repo_id=REPO_ID,
                    revision=revision,
                    token=token,
                    cache_dir=tmp / "hf-cache",
                    local_files_only=False,
                )
            )
        except Exception as exc:
            print(f"ERROR: descarga oficial falló: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 7

        staging = tmp / "staging"
        payload = staging / PAYLOAD_DIRNAME
        copy_snapshot(snapshot_path, payload)

        present = {p.relative_to(payload).as_posix() for p in payload.rglob("*") if p.is_file()}
        missing = sorted(REQUIRED_FILES - present)
        if missing:
            print("ERROR: el layout upstream cambió; faltan archivos obligatorios:", file=sys.stderr)
            for item in missing:
                print(f"  - {item}", file=sys.stderr)
            print("No se genera el mirror hasta revisar el cambio.", file=sys.stderr)
            return 8

        manifest = build_manifest(payload, revision, license_id)
        manifest_path = staging / "JERONIMO_MODEL_MANIFEST.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_notices(staging / "THIRD_PARTY_NOTICES.md", revision, license_id)

        archive_name = f"jeronimo-pyannote-community-1-{revision[:12]}.zip"
        zip_path = output / archive_name
        build_zip(staging, zip_path)

    archive_hash = sha256_file(zip_path)
    sidecar = output / f"{zip_path.name}.sha256"
    sidecar.write_text(f"{archive_hash}  {zip_path.name}\n", encoding="ascii")

    # Copia externa del manifest para auditoría rápida sin abrir el ZIP.
    external_manifest = output / f"{zip_path.stem}.manifest.json"
    external_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    release_template = {
        "schema": 1,
        "enabled": False,
        "artifact_id": "jeronimo-pyannote-community-1",
        "source_repo_id": REPO_ID,
        "revision": revision,
        "archive_name": zip_path.name,
        "archive_sha256": archive_hash,
        "archive_size_bytes": zip_path.stat().st_size,
        "payload_dir": PAYLOAD_DIRNAME,
        "drive_file_id": "",
        "drive_share_url": "",
    }
    release_template_path = output / f"{zip_path.stem}.release-template.json"
    release_template_path.write_text(json.dumps(release_template, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\nOK — artefacto canónico generado")
    print(f"ZIP: {zip_path}")
    print(f"SHA-256: {archive_hash}")
    print(f"Manifest: {external_manifest}")
    print(f"SHA sidecar: {sidecar}")
    print(f"Release template: {release_template_path}")
    print("\nSiguiente paso: subir SOLO el ZIP a Google Drive y conservar manifest/.sha256 como registro de mantenimiento.")
    print("Después ejecutá tools/configurar_mirror_pyannote_drive.py con el enlace compartido de Drive.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
