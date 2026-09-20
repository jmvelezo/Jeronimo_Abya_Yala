"""Credenciales internas del equipo para Jerónimo Abya Yala.

IMPORTANTE:
- Este módulo NO pretende ofrecer secreto criptográfico fuerte dentro de un cliente distribuido.
- La credencial de Hugging Face se almacena ofuscada para evitar texto plano casual.
- En el primer uso se copia al almacén seguro del sistema (keyring / Windows Credential Manager).
- Cualquier credencial embebida en una aplicación cliente puede recuperarse mediante ingeniería inversa.
  Por eso debe ser de alcance mínimo y el paquete debe mantenerse dentro del equipo autorizado.
"""
from __future__ import annotations

import base64
import hashlib

# Blob ofuscado. No contiene el token en texto plano.
_TEAM_HF_BLOB = "dWnCFBVj80AletBoRA/4SQRTWG/DedGrjszsUtckAkpHWNAoJg=="
_DERIVATION_CONTEXT = b"Jeronimo Abya Yala|IICE|team-hf|2026-09"


def get_embedded_team_hf_token() -> str:
    """Devuelve la credencial interna del equipo en memoria.

    La ofuscación evita texto plano accidental, pero no debe confundirse con un
    secreto irrecuperable: el cliente necesita poder reconstruirlo para usarlo.
    """
    try:
        raw = base64.b64decode(_TEAM_HF_BLOB.encode("ascii"), validate=True)
        key = hashlib.sha256(_DERIVATION_CONTEXT).digest()
        decoded = bytes(value ^ key[index % len(key)] for index, value in enumerate(raw))
        token = decoded.decode("utf-8").strip()
        if not token.startswith("hf_") or len(token) < 20:
            return ""
        return token
    except Exception:
        return ""
