from __future__ import annotations

import sys
from pathlib import Path

# Jerónimo Abya Yala — Territorio Vivo
# Capa visual únicamente. No contiene lógica de negocio ni modifica flujos.
COLORS = {
    "canvas": ("#F2EEE5", "#171D1B"),
    "sidebar": ("#173B32", "#10231F"),
    "card": ("#FAF8F2", "#202824"),
    "card_alt": ("#EEE8DB", "#26312B"),
    "text": ("#263029", "#F4F0E6"),
    "muted": ("#68736B", "#AAB7AF"),
    "sidebar_text": "#F5F0E6",
    "on_accent": "#FFFFFF",
    "sidebar_muted": "#C6D1C9",
    "accent": "#315C4A",
    "accent_hover": "#284D3E",
    "accent_soft": ("#DCE8E0", "#294339"),
    "terracotta": "#B85F40",
    "terracotta_hover": "#9E4F36",
    "external_soft": ("#F3E0D8", "#4B3028"),
    "maize": "#D4A84F",
    "water": "#477C82",
    "success": "#2F7554",
    "success_hover": "#26684A",
    "warning": "#B87A2D",
    "danger": "#A8493D",
    "danger_hover": "#873A33",
    "border": ("#D8D0C2", "#34433C"),
    "nav_hover": ("#245143", "#1B3B33"),
    "nav_active": ("#2B604D", "#244B3F"),
}

CARD_RADIUS = 16
CONTROL_RADIUS = 10
SIDEBAR_WIDTH = 236


def asset_path(*parts: str) -> Path:
    """Devuelve la ruta de un recurso visual en fuente o bundle PyInstaller."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base.joinpath("assets", *parts)
