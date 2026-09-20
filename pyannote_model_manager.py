from __future__ import annotations

"""Gestión verificable del modelo pyannote Community-1 de Jerónimo.

Orden de fuentes:
1) copia local validada dentro de ``runtime/models``;
2) mirror versionado de Jerónimo en Google Drive, validado por SHA-256;
3) repositorio oficial de Hugging Face fijado a la misma revisión, usando una
   credencial del equipo o una credencial configurada por el usuario.

Este módulo no procesa entrevistas ni contiene credenciales. Sólo prepara y
verifica el artefacto del modelo de diarización.
"""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from threading import Event
from typing import Any, Callable
import urllib.parse
import urllib.request
import zipfile

from credential_store import get_secure_secret
from team_credentials import get_embedded_team_hf_token

ProgressFn = Callable[[str, float | None], None]

OFFICIAL_REPO_ID = "pyannote/speaker-diarization-community-1"
EXPECTED_LICENSE = "cc-by-4.0"
ARTIFACT_ID = "jeronimo-pyannote-community-1"
PAYLOAD_DIRNAME = "pyannote-speaker-diarization-community-1"
RELEASE_DESCRIPTOR = "pyannote_mirror_release.json"
MODEL_ENV = "JERONIMO_PYANNOTE_MODEL_DIR"

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


class PyannoteModelError(RuntimeError):
    pass


@dataclass(frozen=True)
class MirrorRelease:
    enabled: bool
    revision: str
    archive_name: str
    archive_sha256: str
    archive_size_bytes: int
    drive_file_id: str
    drive_share_url: str
    payload_dir: str = PAYLOAD_DIRNAME

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "revision": self.revision,
            "archive_name": self.archive_name,
            "archive_sha256": self.archive_sha256,
            "archive_size_bytes": self.archive_size_bytes,
            "drive_file_id": self.drive_file_id,
            "drive_share_url": self.drive_share_url,
            "payload_dir": self.payload_dir,
        }


def _app_root() -> Path:
    override = str(os.environ.get("JERONIMO_PORTABLE_ROOT", "") or "").strip()
    if override:
        return Path(override).resolve()
    if bool(getattr(sys, "frozen", False)):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def release_descriptor_path() -> Path:
    override = str(os.environ.get("JERONIMO_PYANNOTE_RELEASE_FILE", "") or "").strip()
    if override:
        return Path(override).resolve()
    return _app_root() / RELEASE_DESCRIPTOR


def local_model_dir() -> Path:
    override = str(os.environ.get(MODEL_ENV, "") or "").strip()
    if override:
        return Path(override).resolve()
    return _app_root() / "runtime" / "models" / "pyannote" / "community-1"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_google_drive_file_id(value: str) -> str:
    """Extrae el file ID de enlaces compartidos habituales o acepta el ID solo."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    if re.fullmatch(r"[A-Za-z0-9_-]{10,}", raw):
        return raw
    try:
        parsed = urllib.parse.urlparse(raw)
        query = urllib.parse.parse_qs(parsed.query)
        for key in ("id", "file_id"):
            item = (query.get(key) or [""])[0]
            if re.fullmatch(r"[A-Za-z0-9_-]{10,}", item):
                return item
        match = re.search(r"/file/d/([A-Za-z0-9_-]{10,})", parsed.path)
        if not match:
            match = re.search(r"/d/([A-Za-z0-9_-]{10,})", parsed.path)
        if match:
            return match.group(1)
    except Exception:
        return ""
    return ""


def load_mirror_release(*, require_enabled: bool = False) -> MirrorRelease | None:
    path = release_descriptor_path()
    if not path.is_file():
        if require_enabled:
            raise PyannoteModelError(
                f"No existe {RELEASE_DESCRIPTOR}. El mirror de Jerónimo todavía no fue configurado."
            )
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PyannoteModelError(f"Descriptor de mirror inválido: {type(exc).__name__}: {exc}") from exc

    enabled = bool(data.get("enabled", False))
    if not enabled and not require_enabled:
        return None
    if not enabled:
        raise PyannoteModelError("El descriptor del mirror está deshabilitado.")

    if str(data.get("artifact_id") or "") != ARTIFACT_ID:
        raise PyannoteModelError("El descriptor no corresponde al artefacto Community-1 esperado.")
    if str(data.get("source_repo_id") or "") != OFFICIAL_REPO_ID:
        raise PyannoteModelError("El descriptor no apunta al repositorio upstream oficial esperado.")

    revision = str(data.get("revision") or "").strip()
    archive_name = str(data.get("archive_name") or "").strip()
    archive_sha256 = str(data.get("archive_sha256") or "").strip().lower()
    drive_file_id = parse_google_drive_file_id(str(data.get("drive_file_id") or data.get("drive_share_url") or ""))
    drive_share_url = str(data.get("drive_share_url") or "").strip()
    payload_dir = str(data.get("payload_dir") or PAYLOAD_DIRNAME).strip()
    try:
        archive_size_bytes = int(data.get("archive_size_bytes") or 0)
    except Exception:
        archive_size_bytes = 0

    if len(revision) < 20:
        raise PyannoteModelError("La revisión congelada del mirror no es válida.")
    if not archive_name.endswith(".zip"):
        raise PyannoteModelError("El nombre del artefacto del mirror no es un ZIP válido.")
    if not re.fullmatch(r"[0-9a-f]{64}", archive_sha256):
        raise PyannoteModelError("El SHA-256 fijado del mirror no es válido.")
    if not drive_file_id:
        raise PyannoteModelError("El descriptor no contiene un Google Drive file ID válido.")
    if payload_dir != PAYLOAD_DIRNAME:
        raise PyannoteModelError("El payload_dir del mirror no coincide con el layout congelado.")

    return MirrorRelease(
        enabled=True,
        revision=revision,
        archive_name=archive_name,
        archive_sha256=archive_sha256,
        archive_size_bytes=max(0, archive_size_bytes),
        drive_file_id=drive_file_id,
        drive_share_url=drive_share_url,
        payload_dir=payload_dir,
    )


def _read_installed_manifest(root: Path) -> dict[str, Any] | None:
    path = root / "JERONIMO_MODEL_MANIFEST.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _validate_payload(root: Path, *, expected_revision: str = "", verify_hashes: bool = True) -> tuple[bool, str]:
    if not root.is_dir():
        return False, "modelo local ausente"
    missing = [rel for rel in sorted(REQUIRED_FILES) if not (root / rel).is_file()]
    if missing:
        return False, "faltan archivos: " + ", ".join(missing[:3])

    manifest = _read_installed_manifest(root)
    if not manifest:
        return False, "falta JERONIMO_MODEL_MANIFEST.json"
    if str(manifest.get("artifact_id") or "") != ARTIFACT_ID:
        return False, "artifact_id inválido"
    source = manifest.get("source") or {}
    if str(source.get("repo_id") or "") != OFFICIAL_REPO_ID:
        return False, "repo upstream inesperado"
    if str(source.get("declared_license") or "").lower() != EXPECTED_LICENSE:
        return False, "licencia del manifiesto inesperada"
    revision = str(source.get("revision") or "").strip()
    if expected_revision and revision != expected_revision:
        return False, "revisión local distinta de la congelada"

    if verify_hashes:
        file_entries = manifest.get("files") or []
        by_path = {str(item.get("path") or ""): item for item in file_entries if isinstance(item, dict)}
        for rel in REQUIRED_FILES:
            item = by_path.get(rel)
            if not item:
                return False, f"manifiesto incompleto para {rel}"
            expected = str(item.get("sha256") or "").lower()
            if not re.fullmatch(r"[0-9a-f]{64}", expected):
                return False, f"hash inválido para {rel}"
            if sha256_file(root / rel) != expected:
                return False, f"hash local no coincide para {rel}"
    return True, "ok"


def local_model_status(*, verify_hashes: bool = False) -> dict[str, Any]:
    release = None
    try:
        release = load_mirror_release()
    except PyannoteModelError:
        release = None
    root = local_model_dir()
    expected_revision = release.revision if release else ""
    ok, detail = _validate_payload(root, expected_revision=expected_revision, verify_hashes=verify_hashes)
    manifest = _read_installed_manifest(root) if ok else None
    revision = str(((manifest or {}).get("source") or {}).get("revision") or "")
    return {
        "ready": ok,
        "path": str(root),
        "detail": detail,
        "revision": revision,
        "mirror_configured": bool(release),
    }


def activate_local_model() -> str:
    status = local_model_status(verify_hashes=False)
    if status["ready"]:
        os.environ[MODEL_ENV] = status["path"]
        return str(status["path"])
    existing = str(os.environ.get(MODEL_ENV, "") or "").strip()
    if existing and not Path(existing).joinpath("config.yaml").is_file():
        os.environ.pop(MODEL_ENV, None)
    return ""


def _drive_download_url(file_id: str) -> str:
    query = urllib.parse.urlencode({"id": file_id, "export": "download", "confirm": "t"})
    return "https://drive.usercontent.google.com/download?" + query


def _download_drive_archive(
    release: MirrorRelease,
    destination: Path,
    *,
    progress: ProgressFn | None = None,
    cancel_event: Event | None = None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(destination.suffix + ".part")
    part.unlink(missing_ok=True)
    request = urllib.request.Request(
        _drive_download_url(release.drive_file_id),
        headers={"User-Agent": "Jerónimo-Abya-Yala/1.0", "Accept": "application/octet-stream,*/*"},
    )
    expected = release.archive_size_bytes
    written = 0
    try:
        with urllib.request.urlopen(request, timeout=45) as response, part.open("wb") as handle:
            content_type = str(response.headers.get("Content-Type", "") or "").lower()
            if "text/html" in content_type:
                raise PyannoteModelError(
                    "Google Drive devolvió una página HTML en lugar del artefacto. Revisá que el archivo esté compartido como 'cualquier persona con el enlace'."
                )
            header_len = int(response.headers.get("Content-Length", "0") or 0)
            total = expected or header_len
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise InterruptedError("Descarga del modelo de hablantes cancelada.")
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                written += len(chunk)
                if progress:
                    frac = min(1.0, written / total) if total else None
                    progress("Descargando Community-1 desde el mirror del equipo…", frac)
    except Exception:
        part.unlink(missing_ok=True)
        raise

    if expected and written != expected:
        part.unlink(missing_ok=True)
        raise PyannoteModelError(
            f"El mirror devolvió {written} bytes y se esperaban {expected}. El archivo no se instalará."
        )
    actual_hash = sha256_file(part)
    if actual_hash != release.archive_sha256:
        part.unlink(missing_ok=True)
        raise PyannoteModelError("El SHA-256 del archivo descargado desde Drive no coincide con el release congelado.")
    os.replace(part, destination)


def _safe_extract(zip_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    base = destination.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            name = info.filename.replace("\\", "/")
            if name.startswith("/") or "../" in f"/{name}":
                raise PyannoteModelError("El ZIP del mirror contiene una ruta insegura.")
            target = (destination / name).resolve()
            if target != base and base not in target.parents:
                raise PyannoteModelError("El ZIP del mirror intenta escribir fuera del directorio temporal.")
        zf.extractall(destination)


def _install_extracted_staging(staging: Path, *, expected_revision: str = "") -> Path:
    payload = staging / PAYLOAD_DIRNAME
    manifest_src = staging / "JERONIMO_MODEL_MANIFEST.json"
    notices_src = staging / "THIRD_PARTY_NOTICES.md"
    if not manifest_src.is_file() or not notices_src.is_file():
        raise PyannoteModelError("El artefacto no contiene manifiesto/avisos de terceros obligatorios.")

    # Validamos el payload usando el manifiesto antes de tocar la instalación actual.
    temp_payload = staging / "_validated_payload"
    if temp_payload.exists():
        shutil.rmtree(temp_payload)
    shutil.copytree(payload, temp_payload)
    shutil.copy2(manifest_src, temp_payload / "JERONIMO_MODEL_MANIFEST.json")
    shutil.copy2(notices_src, temp_payload / "THIRD_PARTY_NOTICES.md")
    ok, detail = _validate_payload(temp_payload, expected_revision=expected_revision, verify_hashes=True)
    if not ok:
        raise PyannoteModelError(f"El artefacto Community-1 no pasó la verificación interna: {detail}")

    target = local_model_dir()
    target.parent.mkdir(parents=True, exist_ok=True)
    backup = target.with_name(target.name + ".old")
    shutil.rmtree(backup, ignore_errors=True)
    if target.exists():
        target.rename(backup)
    try:
        temp_payload.rename(target)
    except Exception:
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        if backup.exists():
            backup.rename(target)
        raise
    shutil.rmtree(backup, ignore_errors=True)
    os.environ[MODEL_ENV] = str(target)
    return target


def install_from_drive(
    release: MirrorRelease,
    *,
    progress: ProgressFn | None = None,
    cancel_event: Event | None = None,
) -> Path:
    downloads = _app_root() / "runtime" / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    archive = downloads / release.archive_name
    if archive.is_file() and sha256_file(archive) != release.archive_sha256:
        archive.unlink(missing_ok=True)
    if not archive.is_file():
        _download_drive_archive(release, archive, progress=progress, cancel_event=cancel_event)
    elif progress:
        progress("Artefacto Community-1 ya descargado; verificando…", 1.0)

    with tempfile.TemporaryDirectory(prefix="jeronimo_pyannote_install_") as td:
        staging = Path(td)
        _safe_extract(archive, staging)
        target = _install_extracted_staging(staging, expected_revision=release.revision)
    archive.unlink(missing_ok=True)
    return target


def _declared_license(info: Any) -> str:
    card = getattr(info, "card_data", None)
    if isinstance(card, dict):
        return str(card.get("license") or "").strip().lower()
    return str(getattr(card, "license", "") or "").strip().lower()


def _build_manifest_for_snapshot(payload: Path, revision: str) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted(payload.rglob("*")):
        if not path.is_file() or path.name in {"JERONIMO_MODEL_MANIFEST.json", "THIRD_PARTY_NOTICES.md"}:
            continue
        rel = path.relative_to(payload).as_posix()
        files.append({"path": rel, "size": path.stat().st_size, "sha256": sha256_file(path)})
    return {
        "schema": 1,
        "artifact_id": ARTIFACT_ID,
        "source": {
            "provider": "Hugging Face",
            "repo_id": OFFICIAL_REPO_ID,
            "revision": revision,
            "declared_license": EXPECTED_LICENSE,
        },
        "packaging": {"payload_dir": PAYLOAD_DIRNAME, "payload_modified": False},
        "files": files,
    }


def install_from_existing_hf_cache(*, revision: str = "") -> Path:
    """Migra una caché oficial ya presente sin usar red ni token."""
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as exc:
        raise PyannoteModelError("huggingface_hub no está disponible para revisar la caché histórica.") from exc
    try:
        snapshot = Path(snapshot_download(
            repo_id=OFFICIAL_REPO_ID,
            revision=revision or None,
            token=None,
            local_files_only=True,
        ))
    except Exception as exc:
        raise PyannoteModelError(f"No existe una caché oficial reutilizable: {type(exc).__name__}: {exc}") from exc

    resolved_revision = revision
    if not resolved_revision:
        # El nombre del snapshot de HF suele ser el commit SHA. Si no podemos
        # determinarlo con seguridad, no promovemos esa caché a instalación
        # canónica porque perderíamos reproducibilidad.
        candidate = snapshot.name.strip()
        if re.fullmatch(r"[0-9a-fA-F]{40,64}", candidate):
            resolved_revision = candidate
    if len(resolved_revision) < 20:
        raise PyannoteModelError("La caché existe pero no se pudo fijar su revisión inmutable.")

    with tempfile.TemporaryDirectory(prefix="jeronimo_pyannote_cache_") as td:
        staging = Path(td)
        payload = staging / PAYLOAD_DIRNAME
        payload.mkdir(parents=True, exist_ok=True)
        for src in sorted(snapshot.rglob("*")):
            if not src.is_file():
                continue
            rel = src.relative_to(snapshot)
            if any(part in {".cache", ".git"} for part in rel.parts):
                continue
            dst = payload / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst, follow_symlinks=True)
        manifest = _build_manifest_for_snapshot(payload, resolved_revision)
        (staging / "JERONIMO_MODEL_MANIFEST.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (staging / "THIRD_PARTY_NOTICES.md").write_text(
            "# Third-party notices — pyannote Community-1\n\n"
            f"Source repository: `{OFFICIAL_REPO_ID}`\n\n"
            f"Frozen upstream revision: `{resolved_revision}`\n\n"
            f"Repository-declared license: `{EXPECTED_LICENSE}`\n\n"
            "Migrated from an existing local Hugging Face cache. Upstream README files are preserved.\n",
            encoding="utf-8",
        )
        return _install_extracted_staging(staging, expected_revision=revision or resolved_revision)


def install_from_huggingface(
    token: str,
    *,
    revision: str = "",
    progress: ProgressFn | None = None,
) -> Path:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        from huggingface_hub import HfApi, snapshot_download  # type: ignore
        try:
            from huggingface_hub.utils import disable_progress_bars  # type: ignore
            disable_progress_bars()
        except Exception:
            pass
    except Exception as exc:
        raise PyannoteModelError("Falta huggingface_hub para usar el respaldo oficial de Community-1.") from exc

    if progress:
        progress("Mirror no disponible; comprobando respaldo oficial de Community-1…", None)
    try:
        api = HfApi()
        info = api.model_info(OFFICIAL_REPO_ID, revision=revision or None, token=token)
        resolved_revision = str(getattr(info, "sha", "") or revision).strip()
        if revision and resolved_revision != revision:
            raise PyannoteModelError("Hugging Face no resolvió la revisión congelada esperada.")
        license_id = _declared_license(info)
        if license_id != EXPECTED_LICENSE:
            raise PyannoteModelError(
                f"La licencia declarada del upstream cambió: esperada={EXPECTED_LICENSE}, recibida={license_id or 'no informada'}."
            )
        snapshot = Path(snapshot_download(
            repo_id=OFFICIAL_REPO_ID,
            revision=revision or resolved_revision or None,
            token=token,
        ))
    except Exception as exc:
        raise PyannoteModelError(f"Hugging Face oficial no pudo entregar Community-1: {type(exc).__name__}: {exc}") from exc

    if len(resolved_revision) < 20:
        raise PyannoteModelError("No se pudo determinar una revisión inmutable de Community-1.")

    with tempfile.TemporaryDirectory(prefix="jeronimo_pyannote_hf_") as td:
        staging = Path(td)
        payload = staging / PAYLOAD_DIRNAME
        payload.mkdir(parents=True, exist_ok=True)
        for src in sorted(snapshot.rglob("*")):
            if not src.is_file():
                continue
            rel = src.relative_to(snapshot)
            if any(part in {".cache", ".git"} for part in rel.parts):
                continue
            dst = payload / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst, follow_symlinks=True)
        manifest = _build_manifest_for_snapshot(payload, resolved_revision)
        (staging / "JERONIMO_MODEL_MANIFEST.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (staging / "THIRD_PARTY_NOTICES.md").write_text(
            "# Third-party notices — pyannote Community-1\n\n"
            f"Source repository: `{OFFICIAL_REPO_ID}`\n\n"
            f"Frozen upstream revision: `{resolved_revision}`\n\n"
            f"Repository-declared license: `{EXPECTED_LICENSE}`\n\n"
            "This local copy was downloaded from the official upstream repository. "
            "The upstream README/component README files are preserved with the model.\n",
            encoding="utf-8",
        )
        target = _install_extracted_staging(staging, expected_revision=revision or resolved_revision)
    return target


def _token_candidates() -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    team = str(get_embedded_team_hf_token() or "").strip()
    configured = str(get_secure_secret("huggingface") or "").strip()
    for source, token in (("team-embedded", team), ("configured", configured)):
        if token and all(existing != token for _, existing in candidates):
            candidates.append((source, token))
    return candidates


def ensure_pyannote_model(
    *,
    progress: ProgressFn | None = None,
    cancel_event: Event | None = None,
) -> dict[str, Any]:
    """Asegura Community-1 siguiendo la cadena local → Drive → HF oficial."""
    status = local_model_status(verify_hashes=False)
    if status["ready"]:
        os.environ[MODEL_ENV] = status["path"]
        return {"ok": True, "source": "local", **status}

    errors: list[str] = []
    release: MirrorRelease | None = None
    try:
        release = load_mirror_release()
    except Exception as exc:
        errors.append(f"descriptor: {type(exc).__name__}: {exc}")

    try:
        cached_target = install_from_existing_hf_cache(revision=release.revision if release else "")
        if progress:
            progress("Se reutilizó una copia oficial ya presente en la caché local.", 1.0)
        return {
            "ok": True,
            "source": "huggingface-cache-migrated",
            "path": str(cached_target),
            "revision": str((_read_installed_manifest(cached_target) or {}).get("source", {}).get("revision", "")),
            "mirror_configured": bool(release),
        }
    except Exception as exc:
        errors.append(f"hf-cache: {type(exc).__name__}: {exc}")

    if release is not None:
        try:
            if progress:
                progress("Preparando Community-1 desde el mirror verificado del equipo…", 0.0)
            target = install_from_drive(release, progress=progress, cancel_event=cancel_event)
            return {
                "ok": True,
                "source": "jeronimo-drive",
                "path": str(target),
                "revision": release.revision,
                "mirror_configured": True,
            }
        except InterruptedError:
            raise
        except Exception as exc:
            errors.append(f"mirror-drive: {type(exc).__name__}: {exc}")
            if progress:
                progress("El mirror de Drive no respondió o no pasó la verificación; probando el respaldo oficial…", None)
    else:
        errors.append("mirror-drive: no configurado")

    revision = release.revision if release else ""
    for source, token in _token_candidates():
        try:
            target = install_from_huggingface(token, revision=revision, progress=progress)
            return {
                "ok": True,
                "source": f"huggingface-{source}",
                "path": str(target),
                "revision": str((_read_installed_manifest(target) or {}).get("source", {}).get("revision", "")),
                "mirror_configured": bool(release),
            }
        except Exception as exc:
            errors.append(f"huggingface-{source}: {type(exc).__name__}: {exc}")

    summary = " | ".join(errors[-4:])
    raise PyannoteModelError(
        "No se pudo preparar Community-1 desde el mirror de Jerónimo ni desde el repositorio oficial. "
        "No se instaló ningún archivo no verificado. Detalle: " + summary
    )
