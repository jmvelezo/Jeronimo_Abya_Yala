from __future__ import annotations

import threading
from threading import Event
from typing import Any

import customtkinter as ctk
import tkinter.messagebox as mb

from model_manager import (
    ASR_MODELS,
    TEXT_MODELS,
    ModelManager,
    benchmark_ollama_model,
    compatibility,
    delete_faster_whisper_model,
    delete_ollama_model,
    download_faster_whisper_model,
    pull_ollama_model,
)
from system_diagnostics import run_system_diagnostics
from ui_help import HelpBubbleButton
from visual_theme import CARD_RADIUS, COLORS, asset_path


def _fmt_bytes(value: int) -> str:
    n = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


class RecommendedModelChoiceDialog(ctk.CTkToplevel):
    """Elección explícita cuando el modelo configurado ya no está disponible."""

    def __init__(self, parent, current_model: str, recommended_name: str, recommended_id: str, recommended_installed: bool):
        super().__init__(parent)
        self.result = "cancel"
        self.title("Modelo local no disponible")
        self.geometry("620x300")
        self.resizable(False, False)
        self.configure(fg_color=COLORS["canvas"])
        try:
            self.iconbitmap(str(asset_path("brand", "app.ico")))
        except Exception:
            pass
        self.transient(parent)
        self.grab_set()
        self.grid_columnconfigure(0, weight=1)

        card = ctk.CTkFrame(self, corner_radius=CARD_RADIUS, fg_color=COLORS["card"], border_width=1, border_color=COLORS["border"])
        card.grid(row=0, column=0, padx=18, pady=18, sticky="nsew")
        card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(card, text="Modelo local no disponible", font=ctk.CTkFont(size=20, weight="bold"), anchor="w").grid(row=0, column=0, padx=18, pady=(16, 6), sticky="ew")
        state = "ya está instalado" if recommended_installed else "puede descargarse automáticamente"
        ctk.CTkLabel(
            card,
            text=(
                f"No se encontró el modelo configurado '{current_model}'.\n\n"
                f"Recomendado para este equipo: {recommended_name} ({recommended_id}). El modelo recomendado {state}. "
                "Jerónimo puede preparar e iniciar su motor privado automáticamente; no hace falta instalar Ollama por separado."
            ),
            justify="left", anchor="w", wraplength=550, text_color=COLORS["muted"],
        ).grid(row=1, column=0, padx=18, pady=(0, 16), sticky="ew")
        actions = ctk.CTkFrame(card, fg_color="transparent")
        actions.grid(row=2, column=0, padx=18, pady=(0, 16), sticky="ew")
        ctk.CTkButton(
            actions, text="Usar recomendado" if recommended_installed else "Descargar y usar recomendado",
            width=210, command=lambda: self._finish("recommended"),
        ).pack(side="left", padx=(0, 8))
        ctk.CTkButton(actions, text="Elegir otro modelo", width=155, fg_color=COLORS["card_alt"], text_color=COLORS["text"], command=lambda: self._finish("other")).pack(side="left", padx=8)
        ctk.CTkButton(actions, text="Cancelar", width=100, fg_color=COLORS["card_alt"], text_color=COLORS["text"], command=lambda: self._finish("cancel")).pack(side="right")
        self.protocol("WM_DELETE_WINDOW", lambda: self._finish("cancel"))
        self.after(50, self.lift)

    def _finish(self, result: str):
        self.result = result
        try:
            self.grab_release()
        except Exception:
            pass
        self.destroy()


def ask_recommended_model_action(parent, current_model: str, recommended_spec, *, recommended_installed: bool) -> str:
    dialog = RecommendedModelChoiceDialog(
        parent, current_model=current_model, recommended_name=recommended_spec.display_name,
        recommended_id=recommended_spec.model_id, recommended_installed=recommended_installed,
    )
    parent.wait_window(dialog)
    return dialog.result


class ModelManagerWindow(ctk.CTkToplevel):
    def __init__(self, parent, ollama_url: str, work_dir: str = "", *, auto_recommended: bool = False):
        super().__init__(parent)
        self.title("Jerónimo Abya Yala — Modelos locales")
        self.geometry("980x690")
        self.configure(fg_color=COLORS["canvas"])
        try:
            self.iconbitmap(str(asset_path("brand", "app.ico")))
        except Exception:
            pass
        self.minsize(850, 560)
        self.ollama_url = ollama_url
        self.work_dir = work_dir or "."
        self.manager = ModelManager(ollama_url)
        self.cancel_event = Event()
        self._busy = False
        self._last_report = None
        self._last_inventory = {}
        self._recommended_spec = None
        self._auto_recommended = bool(auto_recommended)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(
            self, corner_radius=CARD_RADIUS, fg_color=COLORS["card"],
            border_width=1, border_color=COLORS["border"],
        )
        header.grid(row=0, column=0, padx=16, pady=(16, 10), sticky="ew")
        header.grid_columnconfigure(0, weight=1)
        head_text = ctk.CTkFrame(header, fg_color="transparent")
        head_text.grid(row=0, column=0, padx=14, pady=(11, 3), sticky="ew")
        ctk.CTkLabel(
            head_text, text="MODELOS EN ESTE EQUIPO", font=ctk.CTkFont(size=10, weight="bold"),
            text_color=COLORS["terracotta"], anchor="w",
        ).pack(anchor="w")
        ctk.CTkLabel(
            head_text, text="Recursos locales", font=ctk.CTkFont(size=22, weight="bold"),
            text_color=COLORS["text"], anchor="w",
        ).pack(anchor="w", pady=(1, 2))
        self.lbl_summary = ctk.CTkLabel(head_text, text="Analizando equipo y modelos…", anchor="w", text_color=COLORS["muted"])
        self.lbl_summary.pack(fill="x")
        HelpBubbleButton(header, title="Gestor de modelos", help_text="Muestra modelos de resumen y transcripción disponibles, tamaño aproximado, compatibilidad con este equipo y estado de instalación. Descargar un modelo no cambia automáticamente el pipeline validado salvo que una configuración lo seleccione explícitamente.").grid(row=0, column=1, rowspan=2, padx=5, pady=8)
        self.btn_refresh = ctk.CTkButton(
            header, text="Actualizar", width=105, command=self.refresh,
            fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"], corner_radius=10,
        )
        self.btn_refresh.grid(row=0, column=2, rowspan=2, padx=(5, 12), pady=8)

        self.body = ctk.CTkScrollableFrame(self, fg_color=COLORS["canvas"])
        self.body.grid(row=1, column=0, padx=8, pady=(0, 10), sticky="nsew")
        self.body.grid_columnconfigure(0, weight=1)

        footer = ctk.CTkFrame(
            self, corner_radius=CARD_RADIUS, fg_color=COLORS["card"],
            border_width=1, border_color=COLORS["border"],
        )
        footer.grid(row=2, column=0, padx=16, pady=(0, 16), sticky="ew")
        footer.grid_columnconfigure(0, weight=1)
        self.progress = ctk.CTkProgressBar(footer, progress_color=COLORS["terracotta"], fg_color=COLORS["card_alt"])
        self.progress.grid(row=0, column=0, padx=10, pady=8, sticky="ew")
        self.progress.set(0)
        self.lbl_progress = ctk.CTkLabel(footer, text="Listo", anchor="w", text_color=COLORS["muted"])
        self.lbl_progress.grid(row=1, column=0, padx=10, pady=(0, 8), sticky="ew")
        self.btn_cancel = ctk.CTkButton(
            footer, text="Cancelar descarga", state="disabled", width=140, command=self._cancel,
            fg_color=COLORS["card_alt"], hover_color=COLORS["border"], text_color=COLORS["text"],
            border_width=1, border_color=COLORS["border"], corner_radius=10,
        )
        self.btn_cancel.grid(row=0, column=1, rowspan=2, padx=10, pady=8)
        self.after(150, self.refresh)

    def _cancel(self):
        self.cancel_event.set()
        self.lbl_progress.configure(text="Cancelación solicitada…")

    def _set_busy(self, value: bool, cancellable: bool = True):
        self._busy = value
        self.btn_refresh.configure(state="disabled" if value else "normal")
        self.btn_cancel.configure(state="normal" if (value and cancellable) else "disabled")

    def refresh(self):
        if self._busy:
            return
        self.lbl_summary.configure(text="Analizando equipo y modelos…")
        threading.Thread(target=self._refresh_worker, daemon=True).start()

    def _refresh_worker(self):
        try:
            report = run_system_diagnostics(self.work_dir, self.ollama_url)
            inventory = self.manager.inventory()
            self.after(0, lambda: self._render(report, inventory))
        except Exception as exc:
            self.after(0, lambda: mb.showerror("Modelos", f"No se pudo actualizar el gestor: {exc}", parent=self))

    def _clear_body(self):
        for widget in self.body.winfo_children():
            widget.destroy()

    def _render(self, report, inventory):
        self._clear_body()
        self._last_report = report
        self._last_inventory = inventory
        rec = report.recommendations.get("text_primary", "qwen3:4b")
        self._recommended_spec = next((spec for spec in TEXT_MODELS if spec.model_id == rec), None)
        ollama_info = report.ollama or {}
        if ollama_info.get("available"):
            ollama_state = "disponible"
        elif ollama_info.get("runtime_installed"):
            ollama_state = "instalado; Jerónimo lo iniciará automáticamente"
        else:
            ollama_state = "aún no instalado; Jerónimo lo preparará automáticamente"
        self.lbl_summary.configure(text=f"Resumen local recomendado para este equipo: {rec} · Motor local: {ollama_state}")

        row = 0
        if self._recommended_spec is not None:
            spec = self._recommended_spec
            info = inventory.get(spec.model_id, {})
            installed = bool(info.get("installed"))
            reason = str((report.recommendations or {}).get("text_note") or "seleccionado según los recursos detectados")
            basis = (report.recommendations or {}).get("basis", {}) or {}
            ram = float(basis.get("ram_gb", 0) or 0)
            vram = float(basis.get("vram_gb", 0) or 0)
            rec_card = ctk.CTkFrame(
                self.body, corner_radius=14, fg_color=COLORS["card"], border_width=2, border_color=COLORS["maize"],
            )
            rec_card.grid(row=row, column=0, padx=8, pady=(8, 12), sticky="ew")
            rec_card.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(
                rec_card, text="MODELO RECOMENDADO PARA ESTE EQUIPO",
                font=ctk.CTkFont(size=11, weight="bold"), text_color=COLORS["terracotta"], anchor="w",
            ).grid(row=0, column=0, padx=14, pady=(12, 2), sticky="ew")
            ctk.CTkLabel(
                rec_card, text=f"{spec.display_name} · ~{spec.approx_size_gb:.1f} GB",
                font=ctk.CTkFont(size=18, weight="bold"), text_color=COLORS["text"], anchor="w",
            ).grid(row=1, column=0, padx=14, pady=(0, 2), sticky="ew")
            hw = f"{ram:.0f} GB RAM" + (f" · {vram:.1f} GB VRAM" if vram else " · sin VRAM dedicada detectada")
            ctk.CTkLabel(
                rec_card,
                text=f"{reason}. {hw}. Estado: {'instalado' if installed else 'no instalado'}.\nJerónimo puede preparar/iniciar su motor privado automáticamente; no hace falta instalar Ollama por separado.",
                justify="left", anchor="w", text_color=COLORS["muted"], wraplength=700,
            ).grid(row=2, column=0, padx=14, pady=(0, 12), sticky="ew")
            button_text = "Usar recomendado" if installed else "Descargar y usar recomendado"
            ctk.CTkButton(
                rec_card, text=button_text, width=220, fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
                command=lambda s=spec, ok=installed: self._use_recommended(s, ok),
            ).grid(row=0, column=1, rowspan=3, padx=14, pady=14)
            row += 1
            if self._auto_recommended:
                self._auto_recommended = False
                self.after(120, self._use_recommended, spec, installed)

        title = ctk.CTkLabel(self.body, text="Resumen local", font=ctk.CTkFont(size=19, weight="bold"), anchor="w", text_color=COLORS["text"])
        title.grid(row=row, column=0, padx=8, pady=(8, 4), sticky="ew")
        HelpBubbleButton(self.body, title="Modelos de resumen local", help_text="Se ejecutan mediante Ollama. Jerónimo recomienda un tamaño según RAM/VRAM, pero el rendimiento real depende del equipo y del contexto. Puedes descargar, probar y borrar modelos sin usar terminal.").grid(row=row, column=0, padx=8, pady=(8, 4), sticky="e")
        row += 1
        desc = ctk.CTkLabel(self.body, text="Modelos de texto bajo control local. Jerónimo recomienda según RAM/VRAM; confirma el rendimiento con una prueba real tras descargar.", anchor="w", text_color=COLORS["muted"])
        desc.grid(row=row, column=0, padx=8, pady=(0, 8), sticky="ew"); row += 1
        for spec in TEXT_MODELS:
            info = inventory.get(spec.model_id, {})
            frame = self._model_row(spec, info, report, recommended=(spec.model_id == rec))
            frame.grid(row=row, column=0, padx=8, pady=4, sticky="ew"); row += 1

        title2 = ctk.CTkLabel(self.body, text="Transcripción local", font=ctk.CTkFont(size=19, weight="bold"), anchor="w", text_color=COLORS["text"])
        title2.grid(row=row, column=0, padx=8, pady=(18, 4), sticky="ew")
        HelpBubbleButton(self.body, title="Modelos de transcripción", help_text="El baseline con hablantes sigue usando WhisperX large-v2. Otros modelos pueden descargarse para pruebas o para el perfil local sin hablantes; instalar un modelo no lo convierte automáticamente en predeterminado.").grid(row=row, column=0, padx=8, pady=(18, 4), sticky="e")
        row += 1
        note = ctk.CTkLabel(
            self.body,
            text="El baseline real de diarización actual es large-v2. Descargar otro modelo NO cambia automáticamente el pipeline.",
            anchor="w", text_color=COLORS["muted"],
        )
        note.grid(row=row, column=0, padx=8, pady=(0, 8), sticky="ew"); row += 1
        for spec in ASR_MODELS:
            info = inventory.get(spec.model_id, {})
            frame = self._model_row(spec, info, report, recommended=(spec.model_id == "large-v2"))
            frame.grid(row=row, column=0, padx=8, pady=4, sticky="ew"); row += 1

    def _model_row(self, spec, info, report, recommended=False):
        frame = ctk.CTkFrame(
            self.body, corner_radius=12, fg_color=COLORS["card"], border_width=1,
            border_color=COLORS["maize"] if recommended else COLORS["border"],
        )
        frame.grid_columnconfigure(0, weight=1)
        state, reason = compatibility(spec, report)
        installed = bool(info.get("installed"))
        suffix = " · RECOMENDADO" if recommended else ""
        title = ctk.CTkLabel(
            frame, text=f"{spec.display_name}{suffix}", font=ctk.CTkFont(weight="bold"), anchor="w",
            text_color=COLORS["terracotta"] if recommended else COLORS["text"],
        )
        title.grid(row=0, column=0, padx=10, pady=(8, 2), sticky="ew")
        status = "instalado" if installed else "no instalado"
        details = ctk.CTkLabel(
            frame, text=f"{status} · ~{spec.approx_size_gb:.1f} GB · {state}: {reason}\n{spec.notes}",
            anchor="w", justify="left", text_color=COLORS["muted"],
        )
        details.grid(row=1, column=0, padx=10, pady=(0, 8), sticky="ew")
        HelpBubbleButton(frame, title=spec.display_name, help_text=f"{spec.notes} Tamaño aproximado: {spec.approx_size_gb:.1f} GB. Compatibilidad detectada: {state} — {reason}. Estado actual: {status}.").grid(row=0, column=3, rowspan=2, padx=(4, 10), pady=8)
        if installed:
            if spec.backend == "ollama":
                ctk.CTkButton(frame, text="Probar", width=85, fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"], command=lambda s=spec: self._benchmark(s)).grid(row=0, column=1, rowspan=2, padx=4, pady=8)
            ctk.CTkButton(frame, text="Borrar", width=85, fg_color=COLORS["danger"], hover_color=COLORS["danger_hover"], command=lambda s=spec: self._delete(s)).grid(row=0, column=2, rowspan=2, padx=(4, 10), pady=8)
        else:
            ctk.CTkButton(frame, text="Descargar", width=100, fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"], command=lambda s=spec: self._download(s)).grid(row=0, column=2, rowspan=2, padx=(4, 10), pady=8)
        return frame

    def _select_text_model(self, spec) -> None:
        parent = self.master
        try:
            parent.ollama_model = spec.model_id
            models = list(getattr(parent, "ollama_models", []) or [])
            if spec.model_id not in models:
                models.append(spec.model_id)
            parent.ollama_models = sorted(set(models))
            refresh = getattr(parent, "_refresh_text_model_label", None)
            if callable(refresh):
                refresh()
        except Exception:
            pass
        self.lbl_progress.configure(text=f"{spec.display_name}: seleccionado como modelo local de análisis y resumen.")

    def _use_recommended(self, spec, installed: bool):
        if self._busy:
            return
        if installed:
            self._select_text_model(spec)
            return
        self.cancel_event.clear()
        self._set_busy(True, cancellable=True)
        self.progress.set(0)
        self.lbl_progress.configure(
            text=f"Preparando motor privado y {spec.display_name}… Jerónimo instalará/iniciará Ollama automáticamente si hace falta."
        )
        threading.Thread(target=self._download_worker, args=(spec, True), daemon=True).start()

    def _download(self, spec):
        if self._busy:
            return
        # No bloquear porque el servidor privado esté detenido o aún no exista.
        # pull_ollama_model() prepara, inicia y, si es necesario, repara el runtime
        # privado de Jerónimo antes de solicitar el modelo.
        self.cancel_event.clear()
        self._set_busy(True, cancellable=(spec.backend == "ollama"))
        self.progress.set(0)
        if spec.backend == "ollama":
            self.lbl_progress.configure(text=f"Preparando motor privado y {spec.display_name}…")
        else:
            self.lbl_progress.configure(text=f"Preparando {spec.display_name}… Hugging Face puede reanudar la caché si se interrumpe.")
        threading.Thread(target=self._download_worker, args=(spec, False), daemon=True).start()

    def _download_worker(self, spec, select_after: bool = False):
        try:
            if spec.backend == "ollama":
                pull_ollama_model(spec.model_id, self.ollama_url, progress=self._progress_event, cancel_event=self.cancel_event)
            else:
                download_faster_whisper_model(spec, progress=self._progress_event)
            if select_after and spec.backend == "ollama":
                self.after(0, self._select_text_model, spec)
            else:
                self.after(0, lambda: self.lbl_progress.configure(text=f"{spec.display_name}: descarga completada."))
        except Exception as exc:
            message = str(exc)
            self.after(0, lambda m=message: mb.showerror("Descarga", m, parent=self))
            self.after(0, lambda: self.lbl_progress.configure(text=f"{spec.display_name}: descarga interrumpida."))
        finally:
            self.after(0, lambda: self._set_busy(False))
            self.after(250, self.refresh)

    def _progress_event(self, event: dict[str, Any]):
        fraction = event.get("fraction")
        status = str(event.get("status", "descargando"))
        total = int(event.get("total", 0) or 0)
        completed = int(event.get("completed", 0) or 0)
        def apply():
            if fraction is None:
                self.progress.configure(mode="indeterminate")
                self.progress.start()
                self.lbl_progress.configure(text=status)
            else:
                self.progress.stop()
                self.progress.configure(mode="determinate")
                self.progress.set(max(0.0, min(1.0, float(fraction))))
                text = status
                if total:
                    text += f" · {_fmt_bytes(completed)} / {_fmt_bytes(total)}"
                self.lbl_progress.configure(text=text)
        self.after(0, apply)

    def _delete(self, spec):
        if self._busy:
            return
        warning = "Esto borrará el modelo local. Puede ser utilizado también por otra aplicación que comparta la misma caché.\n\n¿Continuar?"
        if not mb.askyesno("Borrar modelo", warning, parent=self):
            return
        try:
            if spec.backend == "ollama":
                delete_ollama_model(spec.model_id, self.ollama_url)
            else:
                delete_faster_whisper_model(spec)
            self.lbl_progress.configure(text=f"{spec.display_name}: eliminado.")
            self.refresh()
        except Exception as exc:
            mb.showerror("Borrar modelo", str(exc), parent=self)

    def _benchmark(self, spec):
        if self._busy:
            return
        self._set_busy(True)
        self.lbl_progress.configure(text=f"Probando {spec.display_name}…")
        threading.Thread(target=self._benchmark_worker, args=(spec,), daemon=True).start()

    def _benchmark_worker(self, spec):
        try:
            result = benchmark_ollama_model(spec.model_id, self.ollama_url)
            tps = result.get("tokens_per_second", 0.0)
            elapsed = result.get("elapsed_seconds", 0.0)
            text = f"{spec.display_name}: {tps:.1f} tokens/s · prueba completa en {elapsed:.1f} s"
            self.after(0, lambda: self.lbl_progress.configure(text=text))
        except Exception as exc:
            self.after(0, lambda: mb.showerror("Prueba del modelo", str(exc), parent=self))
        finally:
            self.after(0, lambda: self._set_busy(False))


def open_model_manager(parent, ollama_url: str, work_dir: str = "", *, auto_recommended: bool = False):
    win = ModelManagerWindow(parent, ollama_url=ollama_url, work_dir=work_dir, auto_recommended=auto_recommended)
    win.transient(parent)
    return win
