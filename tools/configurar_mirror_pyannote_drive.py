#!/usr/bin/env python3
"""Configura el release congelado de Community-1 para Google Drive.

Se ejecuta después de `preparar_mirror_pyannote.py` y de subir SOLO el ZIP
canónico a Google Drive con permiso "cualquier persona con el enlace".

Ejemplo:
    .venv\\Scripts\\python.exe tools\\configurar_mirror_pyannote_drive.py \
        --drive-url "https://drive.google.com/file/d/FILE_ID/view?usp=sharing"

No sube archivos ni usa la API de Drive: sólo fija metadatos verificables para
que el instalador de Jerónimo descargue el ZIP público y compruebe su SHA-256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyannote_model_manager import (  # noqa: E402
    ARTIFACT_ID,
    OFFICIAL_REPO_ID,
    PAYLOAD_DIRNAME,
    RELEASE_DESCRIPTOR,
    parse_google_drive_file_id,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def choose_archive(output: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    candidates = sorted(output.glob("jeronimo-pyannote-community-1-*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("No se encontró el ZIP canónico en tools/_mirror_output. Ejecutá primero la Fase 1.")
    return candidates[0]


def read_manifest_from_zip(archive: Path) -> dict:
    with zipfile.ZipFile(archive, "r") as zf:
        try:
            raw = zf.read("JERONIMO_MODEL_MANIFEST.json")
        except KeyError as exc:
            raise RuntimeError("El ZIP no contiene JERONIMO_MODEL_MANIFEST.json.") from exc
    return json.loads(raw.decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Fija Google Drive como mirror primario de Community-1.")
    parser.add_argument("--drive-url", required=True, help="Enlace compartido de Google Drive o file ID.")
    parser.add_argument("--artifact", type=Path, default=None, help="ZIP canónico; por defecto usa el más reciente de tools/_mirror_output.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "tools" / "_mirror_output")
    args = parser.parse_args()

    file_id = parse_google_drive_file_id(args.drive_url)
    if not file_id:
        print("ERROR: no se pudo extraer un Google Drive file ID del enlace.", file=sys.stderr)
        return 2

    try:
        archive = choose_archive(args.output_dir.resolve(), args.artifact)
        manifest = read_manifest_from_zip(archive)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3

    if str(manifest.get("artifact_id") or "") != ARTIFACT_ID:
        print("ERROR: el artefacto no corresponde a Community-1 de Jerónimo.", file=sys.stderr)
        return 4
    source = manifest.get("source") or {}
    if str(source.get("repo_id") or "") != OFFICIAL_REPO_ID:
        print("ERROR: el manifiesto no declara el upstream oficial esperado.", file=sys.stderr)
        return 5
    revision = str(source.get("revision") or "").strip()
    if len(revision) < 20:
        print("ERROR: revisión congelada inválida.", file=sys.stderr)
        return 6

    archive_hash = sha256_file(archive)
    descriptor = {
        "schema": 1,
        "enabled": True,
        "artifact_id": ARTIFACT_ID,
        "source_repo_id": OFFICIAL_REPO_ID,
        "revision": revision,
        "archive_name": archive.name,
        "archive_sha256": archive_hash,
        "archive_size_bytes": archive.stat().st_size,
        "payload_dir": PAYLOAD_DIRNAME,
        "drive_file_id": file_id,
        "drive_share_url": str(args.drive_url).strip(),
    }
    target = ROOT / RELEASE_DESCRIPTOR
    target.write_text(json.dumps(descriptor, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("OK — mirror primario configurado")
    print(f"Descriptor: {target}")
    print(f"Artifact: {archive.name}")
    print(f"Revision: {revision}")
    print(f"SHA-256: {archive_hash}")
    print(f"Size: {archive.stat().st_size} bytes")
    print("Google Drive file ID: configurado (no es una credencial).")
    print("\nSiguiente paso: recompilar la BASE. El builder copiará este descriptor junto al EXE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
