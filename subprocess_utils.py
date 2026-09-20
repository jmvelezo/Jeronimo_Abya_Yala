from __future__ import annotations

"""Opciones comunes para ejecutar procesos auxiliares sin ventanas de consola.

En un ejecutable gráfico de Windows (PyInstaller ``console=False``), lanzar un
programa de consola como ffmpeg, ffprobe, PowerShell, Python u Ollama sin flags
especiales puede crear una ventana negra fugaz. Este helper centraliza la
configuración necesaria para ocultarla sin cambiar stdin/stdout, códigos de
salida ni la lógica del proceso.
"""

import os
import subprocess
from typing import Any


def hidden_process_kwargs() -> dict[str, Any]:
    """Devuelve kwargs seguros para ocultar procesos hijo en Windows.

    En otros sistemas no modifica el comportamiento. En Windows combina
    ``CREATE_NO_WINDOW`` con ``STARTUPINFO/SW_HIDE``. Se construye un
    ``STARTUPINFO`` nuevo en cada llamada para no compartir estado mutable.
    """
    if os.name != "nt":
        return {}

    kwargs: dict[str, Any] = {}
    create_no_window = int(getattr(subprocess, "CREATE_NO_WINDOW", 0) or 0)
    if create_no_window:
        kwargs["creationflags"] = create_no_window

    startupinfo_cls = getattr(subprocess, "STARTUPINFO", None)
    if startupinfo_cls is not None:
        startupinfo = startupinfo_cls()
        startupinfo.dwFlags |= int(getattr(subprocess, "STARTF_USESHOWWINDOW", 0) or 0)
        startupinfo.wShowWindow = int(getattr(subprocess, "SW_HIDE", 0) or 0)
        kwargs["startupinfo"] = startupinfo

    return kwargs
