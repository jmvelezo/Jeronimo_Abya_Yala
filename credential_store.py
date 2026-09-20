"""Gestión mínima de credenciales de Jerónimo.

Prioridad de resolución:
1) secreto explícito de la sesión,
2) variable de entorno / .env ya cargado,
3) almacén seguro del sistema mediante keyring.

`keyring` es opcional en tiempo de importación para no romper instalaciones
existentes. Si no está disponible, Jerónimo Abya Yala sigue funcionando con claves sólo
de sesión o variables de entorno.
"""
from __future__ import annotations

import os

SERVICE_NAME = "Jeronimo Abya Yala Transcriptor"
ENV_NAMES = {
    "openai": ("OPENAI_API_KEY",),
    "huggingface": ("HUGGINGFACE_TOKEN", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"),
    "openai_compatible": ("JERONIMO_COMPATIBLE_API_KEY", "OPENAI_COMPATIBLE_API_KEY"),
}


def canonical_provider(provider: str) -> str:
    raw = str(provider or "").strip().lower()
    aliases = {
        "hf": "huggingface",
        "hugging_face": "huggingface",
        "pyannote": "huggingface",
        "compatible": "openai_compatible",
        "openai-compatible": "openai_compatible",
    }
    return aliases.get(raw, raw)


def _account(provider: str) -> str:
    return f"credential:{canonical_provider(provider)}"


def _load_keyring():
    try:
        import keyring  # type: ignore
        return keyring
    except Exception:
        return None


def environment_secret(provider: str) -> str:
    for name in ENV_NAMES.get(canonical_provider(provider), ()): 
        value = str(os.environ.get(name, "") or "").strip()
        if value:
            return value
    return ""


def get_secure_secret(provider: str) -> str:
    keyring = _load_keyring()
    if keyring is None:
        return ""
    try:
        return str(keyring.get_password(SERVICE_NAME, _account(provider)) or "").strip()
    except Exception:
        return ""


def set_secure_secret(provider: str, secret: str) -> None:
    value = str(secret or "").strip()
    if not value:
        raise ValueError("La credencial está vacía.")
    keyring = _load_keyring()
    if keyring is None:
        raise RuntimeError(
            "El paquete 'keyring' no está disponible. Ejecuta el instalador de Jerónimo Abya Yala "
            "o usa la credencial sólo durante esta sesión."
        )
    try:
        keyring.set_password(SERVICE_NAME, _account(provider), value)
    except Exception as exc:
        raise RuntimeError(f"No se pudo guardar la credencial en el almacén seguro: {exc}") from exc


def delete_secure_secret(provider: str) -> bool:
    keyring = _load_keyring()
    if keyring is None:
        return False
    try:
        account = _account(provider)
        current = keyring.get_password(SERVICE_NAME, account)
        if not current:
            return False
        keyring.delete_password(SERVICE_NAME, account)
        return True
    except Exception:
        return False


def resolve_secret(provider: str, explicit: str = "") -> str:
    explicit_value = str(explicit or "").strip()
    if explicit_value:
        return explicit_value
    env_value = environment_secret(provider)
    if env_value:
        return env_value
    return get_secure_secret(provider)


def credential_source(provider: str, explicit: str = "") -> str:
    if str(explicit or "").strip():
        return "session"
    if environment_secret(provider):
        return "environment"
    if get_secure_secret(provider):
        return "keyring"
    return "missing"


def credential_status_text(provider: str, explicit: str = "") -> str:
    source = credential_source(provider, explicit)
    if source == "session":
        return "credencial disponible sólo en esta sesión"
    if source == "environment":
        return "credencial detectada en variable de entorno/.env"
    if source == "keyring":
        return "credencial guardada en el almacén seguro del sistema"
    return "credencial no configurada"
