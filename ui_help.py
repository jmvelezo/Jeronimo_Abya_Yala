"""Ayuda contextual reutilizable para Jerónimo Abya Yala.

Comportamiento:
- hover sobre ``?``: muestra el globo tras una pausa breve;
- el globo permanece visible mientras el puntero esté sobre ``?`` o sobre el propio globo;
- al salir de ambas zonas se cierra, con una tolerancia breve para mover el puntero entre ellas;
- clic sobre ``?``: abre/cierra el mismo globo, sin temporizadores de autocierre ni modo fijado permanente.
"""
from __future__ import annotations

import customtkinter as ctk

from visual_theme import COLORS


class HelpBubbleButton(ctk.CTkButton):
    """Botón ``?`` con ayuda contextual estable mientras se está leyendo.

    El hotfix anterior vigilaba la posición del puntero de forma periódica. En
    Windows/CustomTkinter esa comprobación podía producir falsos negativos y
    cerrar el globo aunque el usuario siguiera sobre el control. Esta versión
    no usa polling: sólo reacciona a entrada/salida real y, antes de cerrar,
    comprueba la geometría de las dos zonas válidas (botón y globo).
    """

    _SHOW_DELAY_MS = 180
    _HIDE_GRACE_MS = 220

    def __init__(self, master, *, title: str, help_text: str, **kwargs):
        self.help_title = str(title)
        self.help_text = str(help_text)
        self._bubble = None
        self._hover_after = None
        self._hide_after = None
        super().__init__(
            master,
            text="?",
            width=28,
            height=28,
            corner_radius=14,
            fg_color=COLORS["accent_soft"],
            hover_color=COLORS["border"],
            text_color=COLORS["accent"],
            border_width=1,
            border_color=COLORS["border"],
            command=self._toggle_bubble,
            **kwargs,
        )
        self.bind("<Enter>", self._on_enter, add="+")
        self.bind("<Leave>", self._on_leave, add="+")
        self.bind("<Destroy>", lambda _e: self._hide(), add="+")

    def _cancel_after(self, attr_name: str):
        after_id = getattr(self, attr_name, None)
        if after_id is None:
            return
        try:
            self.after_cancel(after_id)
        except Exception:
            pass
        setattr(self, attr_name, None)

    def _on_enter(self, _event=None):
        self._cancel_after("_hide_after")
        self._cancel_after("_hover_after")
        if self._bubble is not None:
            return
        try:
            self._hover_after = self.after(self._SHOW_DELAY_MS, self._show)
        except Exception:
            self._show()

    def _on_leave(self, _event=None):
        self._cancel_after("_hover_after")
        self._schedule_hide()

    def _toggle_bubble(self):
        self._cancel_after("_hover_after")
        self._cancel_after("_hide_after")
        if self._bubble is not None:
            self._hide()
        else:
            self._show()

    @staticmethod
    def _pointer_inside(widget) -> bool:
        if widget is None:
            return False
        try:
            if not bool(widget.winfo_exists()):
                return False
            px = widget.winfo_pointerx()
            py = widget.winfo_pointery()
            x0 = widget.winfo_rootx()
            y0 = widget.winfo_rooty()
            width = max(1, widget.winfo_width())
            height = max(1, widget.winfo_height())
            return x0 <= px < x0 + width and y0 <= py < y0 + height
        except Exception:
            return False

    def _pointer_in_help_region(self) -> bool:
        return self._pointer_inside(self) or self._pointer_inside(self._bubble)

    def _schedule_hide(self, _event=None):
        self._cancel_after("_hide_after")
        if self._bubble is None:
            return
        try:
            self._hide_after = self.after(self._HIDE_GRACE_MS, self._hide_if_outside)
        except Exception:
            self._hide_if_outside()

    def _hide_if_outside(self):
        self._hide_after = None
        if self._bubble is None:
            return
        if self._pointer_in_help_region():
            return
        self._hide()

    def _bubble_enter(self, _event=None):
        self._cancel_after("_hide_after")

    def _bubble_leave(self, _event=None):
        self._schedule_hide()

    def _show(self):
        self._hover_after = None
        self._cancel_after("_hide_after")
        if self._bubble is not None:
            return
        try:
            top = ctk.CTkToplevel(self)
            top.overrideredirect(True)
            try:
                top.attributes("-topmost", True)
            except Exception:
                pass
            top.configure(fg_color=COLORS["canvas"])
            frame = ctk.CTkFrame(
                top,
                corner_radius=12,
                border_width=1,
                fg_color=COLORS["card"],
                border_color=COLORS["border"],
            )
            frame.pack(fill="both", expand=True)
            ctk.CTkFrame(
                frame,
                height=3,
                corner_radius=2,
                fg_color=COLORS["terracotta"],
            ).pack(fill="x", padx=10, pady=(8, 2))
            ctk.CTkLabel(
                frame,
                text=self.help_title,
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color=COLORS["text"],
                anchor="w",
            ).pack(fill="x", padx=12, pady=(10, 3))
            ctk.CTkLabel(
                frame,
                text=self.help_text,
                justify="left",
                anchor="w",
                wraplength=360,
                text_color=COLORS["muted"],
            ).pack(fill="both", expand=True, padx=12, pady=(0, 10))

            top.update_idletasks()
            x = self.winfo_rootx() + self.winfo_width() + 8
            y = self.winfo_rooty() - 6
            sw = self.winfo_screenwidth()
            sh = self.winfo_screenheight()
            bw = top.winfo_reqwidth()
            bh = top.winfo_reqheight()
            if x + bw > sw - 12:
                x = max(12, self.winfo_rootx() - bw - 8)
            if y + bh > sh - 12:
                y = max(12, sh - bh - 12)
            top.geometry(f"+{int(x)}+{int(y)}")

            # El usuario puede mover el puntero desde el signo '?' hasta el
            # cartel para leerlo sin que desaparezca. Sólo se cierra al salir
            # de ambos o al volver a pulsar '?'.
            top.bind("<Enter>", self._bubble_enter, add="+")
            top.bind("<Leave>", self._bubble_leave, add="+")
            top.bind("<Button-1>", lambda _e: self._hide(), add="+")
            self._bubble = top
        except Exception:
            self._bubble = None

    def _hide(self):
        self._cancel_after("_hover_after")
        self._cancel_after("_hide_after")
        bubble, self._bubble = self._bubble, None
        if bubble is not None:
            try:
                bubble.destroy()
            except Exception:
                pass


def add_help(master, *, title: str, help_text: str, **kwargs) -> HelpBubbleButton:
    return HelpBubbleButton(master, title=title, help_text=help_text, **kwargs)
