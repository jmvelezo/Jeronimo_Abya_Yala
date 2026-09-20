"""Asistente de primer inicio de Jerónimo (FASE 7)."""
from __future__ import annotations

import os
import threading
import webbrowser
from pathlib import Path

import customtkinter as ctk
import tkinter.messagebox as mb
from PIL import Image

from credential_store import resolve_secret, set_secure_secret, credential_status_text
from automatic_setup import build_automatic_plan, run_automatic_setup
from model_manager import TEXT_MODELS, ollama_inventory, ollama_model_present, pull_ollama_model
from model_manager_ui import open_model_manager
from onboarding import (
    HF_PYANNOTE_URL,
    HF_SIGNUP_URL,
    HF_TOKEN_URL,
    OPENAI_API_KEYS_URL,
    build_readiness,
    complete_onboarding,
    mark_step,
    validate_huggingface,
)
from system_diagnostics import run_system_diagnostics
from diarization_runtime_manager import runtime_status as diarization_runtime_status
from ui_help import HelpBubbleButton
from visual_theme import CARD_RADIUS, COLORS, asset_path
from text_providers import (
    DEFAULT_OPENAI_COMPATIBLE_URL,
    DEFAULT_OLLAMA_URL,
    TextProviderSettings,
    test_provider,
)


TEAM_INSTAGRAM_URL = "https://www.instagram.com/jovenesyescuela/"


def _apply_territorio_controls(root) -> None:
    """Aplica sólo presentación Territorio Vivo a controles CTk ya creados."""
    try:
        default_button = ctk.ThemeManager.theme["CTkButton"]["fg_color"]
    except Exception:
        default_button = None
    stack = [root]
    while stack:
        widget = stack.pop()
        try:
            stack.extend(widget.winfo_children())
        except Exception:
            continue
        if isinstance(widget, HelpBubbleButton):
            continue
        try:
            if isinstance(widget, ctk.CTkButton):
                current = widget.cget("fg_color")
                if default_button is None or current == default_button:
                    widget.configure(
                        fg_color=COLORS["accent"],
                        hover_color=COLORS["accent_hover"],
                        text_color=COLORS["sidebar_text"],
                        corner_radius=10,
                    )
            elif isinstance(widget, ctk.CTkEntry):
                widget.configure(border_color=COLORS["border"], fg_color=COLORS["card_alt"])
            elif isinstance(widget, ctk.CTkProgressBar):
                widget.configure(progress_color=COLORS["terracotta"], fg_color=COLORS["card_alt"])
            elif isinstance(widget, ctk.CTkCheckBox):
                widget.configure(fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"], border_color=COLORS["border"])
        except Exception:
            pass

STEP_KEYS = ("intro", "automatic", "privacy", "system", "transcription", "local", "huggingface", "apis", "done")
STEP_TITLES = {
    "intro": "Bienvenida",
    "automatic": "Automática",
    "privacy": "Privacidad",
    "system": "Este equipo",
    "transcription": "Transcripción",
    "local": "Resumen local",
    "huggingface": "Hablantes",
    "apis": "APIs opcionales",
    "done": "Listo",
}

PAGE_HELP_TEXT = {
    "Bienvenido/a a Jerónimo Abya Yala": "Esta pantalla presenta el proyecto y te permite elegir configuración Automática o Avanzada. Automática diagnostica el equipo, prepara la transcripción local con hablantes y el resumen local recomendado. Avanzada expone cada decisión por separado. Ninguna entrevista se sube a un servicio externo por elegir el modo Automático.",
    "Configuración automática": "Analiza CPU, RAM, GPU/VRAM, CUDA, disco, el motor privado de texto y los modelos existentes. Descarga sólo lo necesario y deja configurado el camino local recomendado sin instalar Ollama como programa global. Puede descargar varios gigas de información y exigir bastante al equipo durante la instalación y el procesamiento.",
    "Privacidad y ética de los datos": "Explica qué datos permanecen en la computadora y cuándo una API externa podría recibir audio o texto. El flujo local es el predeterminado. En Automática, Community-1 se obtiene primero desde el mirror verificado del equipo; Hugging Face queda como contingencia de descarga. El audio de la entrevista no se envía a esos servicios durante la diarización local.",
    "Comprobación del equipo": "Revisa recursos y componentes sin subir información del equipo. CUDA indica si PyTorch puede usar una GPU NVIDIA compatible. FFmpeg prepara audio; WhisperX realiza transcripción/alineación y pyannote separa hablantes; el runtime privado de Ollama administrado por Jerónimo ejecuta el análisis de texto local.",
    "Transcripción local": "El baseline de calidad con hablantes sigue siendo WhisperX large-v2 porque es el pipeline validado. Los modelos alternativos pueden instalarse, pero no sustituyen silenciosamente el baseline. El gestor permite comprobar qué modelos están disponibles.",
    "Resumen local": "Jerónimo administra su propio runtime standalone de Ollama y sus modelos, sin depender de una instalación global. El recomendador considera RAM/VRAM y espacio; descargar el runtime y un modelo puede requerir varios GB. El resumen es un derivado y nunca reemplaza la transcripción original.",
    "Separación de hablantes": "La diarización separa voces como Hablante 1, Hablante 2, etc.; no identifica personas por su identidad. Community-1 se instala como copia local verificada y luego funciona sin token ni conexión. Automática prioriza el mirror versionado del equipo y usa Hugging Face sólo como respaldo.",
    "APIs externas — opcional": "Las APIs permiten transcribir o procesar texto fuera del equipo. Son opcionales. Las credenciales se guardan mediante el almacén seguro del sistema cuando eliges guardarlas. Jerónimo avisa antes de enviar audio o texto a un endpoint externo.",
    "Jerónimo Abya Yala está preparado": "Resume el estado final de la configuración. Antes del primer corpus completo conviene probar una entrevista corta y contrastar transcripción, hablantes y timestamps con el audio original.",
}



def _open_url(url: str) -> None:
    try:
        webbrowser.open(url, new=2)
    except Exception as exc:
        mb.showerror("Abrir enlace", f"No se pudo abrir el navegador:\n{url}\n\n{exc}")


class OnboardingWizard(ctk.CTkToplevel):
    def __init__(
        self, master, *, on_complete=None, required: bool = True,
        startup_mode: str = "full", repair_reasons: tuple[str, ...] = (),
    ):
        super().__init__(master)
        self.app = master
        self.on_complete = on_complete
        self.required = bool(required)
        self.startup_mode = str(startup_mode or "full")
        self.repair_mode = self.required and self.startup_mode == "repair"
        self.repair_reasons = tuple(str(x) for x in (repair_reasons or ()) if str(x).strip())
        self.report = getattr(master, "_diag_report", None)
        self._step_index = 0
        self._busy = False
        self._hf_test_ok = False
        self._setup_mode = ""
        self._automatic_complete = False
        self._automatic_result = {}
        self._automatic_cancel = threading.Event()
        self._privacy_var = ctk.BooleanVar(value=False)
        self._model_download_cancel = threading.Event()
        self._closing = False
        self._force_exit_timer = None

        # Ventana modal sin barra nativa. La configuración sigue siendo obligatoria
        # para entrar al programa, pero la persona siempre puede cerrar Jerónimo.
        # La X propia intenta un cierre normal y fuerza la salida tras 5 s si algún
        # instalador/descarga de terceros mantiene vivo el proceso.
        self.overrideredirect(True)
        # Si la raíz está retirada durante el primer inicio, no convertir el
        # onboarding en transient de una ventana invisible: en Windows puede
        # alterar su visibilidad/z-order. En aperturas posteriores sí conserva
        # la relación transient con la aplicación principal.
        try:
            if str(master.state()) != "withdrawn":
                self.transient(master)
        except Exception:
            pass
        self.grab_set()
        self.bind("<Escape>", lambda _e: "break")
        self.bind("<Alt-F4>", self._close_event, add="+")
        self.protocol("WM_DELETE_WINDOW", self._request_application_exit)

        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        # Nunca crear una ventana mayor que la pantalla disponible. En equipos
        # con menor resolución el contenido central pasa a desplazarse, mientras
        # cabecera, navegación y cierre permanecen siempre accesibles.
        available_width = max(1, sw - 20)
        available_height = max(1, sh - 35)
        width = min(1220, available_width)
        height = min(820, available_height)
        if available_width >= 760:
            width = max(760, width)
        if available_height >= 520:
            height = max(520, height)
        x = max(0, (sw - width) // 2)
        y = max(0, (sh - height) // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")
        self._compact = height < 720 or width < 1000
        self._wraplength = max(300, min(780, width - (260 if self._compact else 360)))
        # Mantener el asistente por delante de la aplicación principal durante
        # toda la configuración.  En Windows también deshabilitamos la ventana
        # principal para que no pueda tapar ni recibir interacción mientras el
        # proceso inicial obligatorio está activo.
        self._parent_disabled = False
        self._front_guard_after = None
        self._lock_parent_window()
        try:
            self.attributes("-topmost", True)
        except Exception:
            pass
        # Elevar una sola vez. Mantener ``-topmost`` y bloquear la ventana
        # principal es suficiente para que el onboarding quede delante.
        #
        # NO ejecutar un ``lift()`` periódico aquí: ese guard de z-order
        # volvía a elevar el onboarding cada 750 ms y tapaba los Toplevel
        # de ayuda contextual. Por eso el globo parecía desaparecer al cabo
        # de aproximadamente un segundo aunque seguía existiendo.
        self.after_idle(self._bring_to_front)

        self.configure(fg_color=COLORS["canvas"])
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Cabecera propia: misma identidad Territorio Vivo de la aplicación principal.
        header = ctk.CTkFrame(self, height=44, corner_radius=0, fg_color=COLORS["sidebar"])
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        header.grid_propagate(False)
        title = ctk.CTkLabel(
            header, text=("Jerónimo Abya Yala — Reparar configuración" if self.repair_mode else "Jerónimo Abya Yala — Configuración inicial"),
            font=ctk.CTkFont(size=14, weight="bold"), text_color=COLORS["sidebar_text"],
        )
        title.pack(side="left", padx=16, pady=10)
        self.btn_window_close = ctk.CTkButton(
            header, text="X", width=36, height=30, corner_radius=7,
            fg_color="transparent", hover_color=COLORS["terracotta"],
            text_color=COLORS["sidebar_text"], font=ctk.CTkFont(size=13, weight="bold"),
            command=self._request_application_exit,
        )
        self.btn_window_close.pack(side="right", padx=(0, 8), pady=7)
        ctk.CTkLabel(
            header, text=("ACTUALIZACIÓN TÉCNICA · Reparación obligatoria" if self.repair_mode else "CONFIGURACIÓN LOCAL · Proceso inicial obligatorio"), font=ctk.CTkFont(size=10, weight="bold"),
            text_color=COLORS["maize"],
        ).pack(side="right", padx=(16, 10))
        for widget in (header, title):
            widget.bind("<ButtonPress-1>", self._drag_start, add="+")
            widget.bind("<B1-Motion>", self._drag_move, add="+")

        sidebar_width = 190 if self._compact else 236
        sidebar = ctk.CTkFrame(self, width=sidebar_width, corner_radius=0, fg_color=COLORS["sidebar"])
        sidebar.grid(row=1, column=0, sticky="nsew")
        sidebar.grid_propagate(False)

        brand = ctk.CTkFrame(sidebar, fg_color="transparent")
        brand.pack(fill="x", padx=18, pady=(20, 18))
        self._visual_images = {}
        try:
            mark = Image.open(asset_path("brand", "jeronimo_mark_light.png"))
            self._visual_images["brand"] = ctk.CTkImage(light_image=mark, dark_image=mark, size=(42, 42))
            ctk.CTkLabel(brand, text="", image=self._visual_images["brand"], width=44).pack(side="left", padx=(0, 10))
        except Exception:
            pass
        brand_text = ctk.CTkFrame(brand, fg_color="transparent")
        brand_text.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(brand_text, text="JERÓNIMO", font=ctk.CTkFont(size=15, weight="bold"), text_color=COLORS["sidebar_text"], anchor="w").pack(fill="x")
        ctk.CTkLabel(brand_text, text="ABYA YALA", font=ctk.CTkFont(size=10, weight="bold"), text_color=COLORS["sidebar_muted"], anchor="w").pack(fill="x")
        ctk.CTkLabel(sidebar, text="CONFIGURACIÓN INICIAL", font=ctk.CTkFont(size=9, weight="bold"), text_color=COLORS["maize"], anchor="w").pack(fill="x", padx=20, pady=(0, 8))

        self.step_labels = {}
        for index, key in enumerate(STEP_KEYS, start=1):
            label = ctk.CTkLabel(
                sidebar, text=f"{index:02d}   {STEP_TITLES[key]}", anchor="w",
                height=29 if self._compact else 34, corner_radius=9,
                fg_color="transparent", text_color=COLORS["sidebar_muted"],
                font=ctk.CTkFont(size=11 if self._compact else 12),
            )
            label.pack(fill="x", padx=14, pady=1)
            self.step_labels[key] = label

        ctk.CTkLabel(
            sidebar, text="DATOS BAJO TU CONTROL", font=ctk.CTkFont(size=9, weight="bold"),
            text_color=COLORS["sidebar_muted"], anchor="w",
        ).pack(side="bottom", fill="x", padx=20, pady=(0, 18))

        main = ctk.CTkFrame(self, fg_color=COLORS["canvas"], corner_radius=0)
        main.grid(row=1, column=1, sticky="nsew", padx=18 if self._compact else 28, pady=14 if self._compact else 22)
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(0, weight=1)

        self.host = ctk.CTkFrame(main, fg_color="transparent")
        self.host.grid(row=0, column=0, sticky="nsew")
        self.host.grid_columnconfigure(0, weight=1)
        self.host.grid_rowconfigure(0, weight=1)
        self.pages = {}
        self._build_pages()

        footer = ctk.CTkFrame(main, fg_color="transparent")
        footer.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        footer.grid_columnconfigure(1, weight=1)
        self.btn_back = ctk.CTkButton(
            footer, text="Atrás", width=100, command=self._back,
            fg_color=COLORS["card_alt"], hover_color=COLORS["border"], text_color=COLORS["text"],
            border_width=1, border_color=COLORS["border"], corner_radius=10,
        )
        self.btn_back.grid(row=0, column=0, sticky="w")
        self.lbl_footer = ctk.CTkLabel(footer, text="", text_color=COLORS["muted"])
        self.lbl_footer.grid(row=0, column=1)
        self.btn_next = ctk.CTkButton(
            footer, text="Continuar", width=140, command=self._next,
            fg_color=COLORS["terracotta"], hover_color=COLORS["terracotta_hover"], text_color=COLORS["on_accent"], corner_radius=10,
        )
        self.btn_next.grid(row=0, column=2, sticky="e")

        _apply_territorio_controls(self)
        # Los botones de navegación principal conservan jerarquía propia tras el estilo general.
        self.btn_back.configure(fg_color=COLORS["card_alt"], hover_color=COLORS["border"], text_color=COLORS["text"], border_width=1, border_color=COLORS["border"])
        self.btn_next.configure(fg_color=COLORS["terracotta"], hover_color=COLORS["terracotta_hover"], text_color=COLORS["on_accent"])
        if self.repair_mode:
            self._setup_mode = "automatic"
            self._privacy_var.set(True)
            self._configure_sidebar_for_mode("repair")
            self._show_step(STEP_KEYS.index("automatic"))
            reasons = "\n".join(f"• {item}" for item in self.repair_reasons)
            self.lbl_auto_status.configure(
                text=(
                    "Esta PC ya completó una configuración anterior. Jerónimo debe comprobar y reparar "
                    "los componentes locales que exige esta versión antes de continuar."
                    + (f"\n\nMotivos detectados:\n{reasons}" if reasons else "")
                    + "\n\nNo se repetirán la bienvenida ni la aceptación de privacidad. "
                      "Pulsa ‘Reparar configuración’ para reutilizar lo válido y descargar sólo lo que falte."
                )
            )
        else:
            self._configure_sidebar_for_mode("")
            self._show_step(0)
        self.after(120, self._refresh_system_async)

    def _lock_parent_window(self):
        """Impide que la ventana principal quede por encima del onboarding."""
        try:
            if getattr(self, "app", None) is not None and self.app.winfo_exists():
                # -disabled es una extensión de Tk disponible en Windows.
                self.app.attributes("-disabled", True)
                self._parent_disabled = True
        except Exception:
            self._parent_disabled = False

    def _unlock_parent_window(self):
        if not getattr(self, "_parent_disabled", False):
            return
        try:
            if getattr(self, "app", None) is not None and self.app.winfo_exists():
                self.app.attributes("-disabled", False)
        except Exception:
            pass
        self._parent_disabled = False

    def _bring_to_front(self):
        try:
            if self.winfo_exists():
                self.lift()
                self.attributes("-topmost", True)
        except Exception:
            pass

    def _schedule_front_guard(self):
        """Compatibilidad: ya no hace polling/lift periódico.

        El onboarding permanece delante mediante ``-topmost`` y el bloqueo de
        la ventana principal. Un ``lift`` recurrente compite con los globos
        de ayuda (que también son Toplevel) y puede ocultarlos visualmente.
        """
        self._front_guard_after = None

    def destroy(self):
        # Restaurar siempre la ventana principal, incluso si una excepción cierra
        # el asistente durante desarrollo.
        try:
            if self._front_guard_after is not None:
                self.after_cancel(self._front_guard_after)
        except Exception:
            pass
        self._front_guard_after = None
        try:
            self.attributes("-topmost", False)
        except Exception:
            pass
        self._unlock_parent_window()
        return super().destroy()

    def _drag_start(self, event):
        self._drag_x = event.x_root - self.winfo_x()
        self._drag_y = event.y_root - self.winfo_y()

    def _drag_move(self, event):
        try:
            self.geometry(f"+{event.x_root - self._drag_x}+{event.y_root - self._drag_y}")
        except Exception:
            pass

    def _page(self, key: str):
        # Todos los pasos viven en un área desplazable. Esto evita que una
        # resolución baja o un texto más largo oculte opciones esenciales.
        # El pie Atrás/Continuar queda fuera de este frame y permanece fijo.
        frame = ctk.CTkScrollableFrame(
            self.host, fg_color="transparent", corner_radius=0,
        )
        frame.grid(row=0, column=0, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)
        self.pages[key] = frame
        return frame

    def _title(self, page, title: str, subtitle: str, help_text: str = ""):
        row = ctk.CTkFrame(page, fg_color="transparent")
        row.grid(row=0, column=0, sticky="ew", pady=(2, 3))
        row.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            row, text=title, font=ctk.CTkFont(size=25 if self._compact else 29, weight="bold"),
            anchor="w", text_color=COLORS["text"],
        ).grid(row=0, column=0, sticky="w")
        HelpBubbleButton(
            row,
            title=title,
            help_text=help_text or PAGE_HELP_TEXT.get(title, subtitle),
        ).grid(row=0, column=1, sticky="e", padx=(10, 0))
        ctk.CTkLabel(page, text=subtitle, justify="left", wraplength=self._wraplength, anchor="w", text_color=COLORS["muted"]).grid(row=1, column=0, sticky="ew", pady=(0, 12 if self._compact else 18))

    def _card(self, page, row: int):
        card = ctk.CTkFrame(
            page, corner_radius=CARD_RADIUS, fg_color=COLORS["card"],
            border_width=1, border_color=COLORS["border"],
        )
        card.grid(row=row, column=0, sticky="ew", pady=7)
        card.grid_columnconfigure(0, weight=1)
        return card

    def _build_pages(self):
        # Bienvenida
        page = self._page("intro")
        self._title(
            page,
            "Bienvenido/a a Jerónimo Abya Yala",
            "Desgrabación de entrevistas con procesamiento local por defecto, trazabilidad y control explícito sobre los datos.",
        )
        project = self._card(page, 2)
        project_title = ctk.CTkFrame(project, fg_color="transparent")
        project_title.pack(fill="x", padx=20, pady=(14, 4))
        ctk.CTkLabel(project_title, text="Proyecto y propósito", font=ctk.CTkFont(size=18, weight="bold"), anchor="w").pack(side="left")
        HelpBubbleButton(project_title, title="Proyecto y propósito", help_text="Jerónimo Abya Yala está orientado al trabajo con entrevistas del equipo Jóvenes y Escuela Secundaria, IICE, Facultad de Filosofía y Letras de la UBA. Su diseño prioriza procesamiento local, control de datos y trazabilidad de cada transformación.").pack(side="left", padx=8)
        ctk.CTkLabel(
            project,
            text=(
                "Jerónimo Abya Yala fue creado para apoyar el trabajo con entrevistas del equipo Jóvenes y Escuela Secundaria, "
                "vinculado al Instituto de Investigaciones en Ciencias de la Educación (IICE) de la Facultad de Filosofía y Letras de la Universidad de Buenos Aires. "
                "Su diseño prioriza procesamiento local, control de los datos y trazabilidad."
            ),
            justify="left", wraplength=710, anchor="w"
        ).pack(fill="x", padx=20, pady=(0, 8))

        developer_note = ctk.CTkFrame(project, corner_radius=12, fg_color=COLORS["card_alt"])
        developer_note.pack(fill="x", padx=20, pady=(2, 10))
        ctk.CTkLabel(
            developer_note, text="Nota del desarrollador",
            font=ctk.CTkFont(size=14, weight="bold"), anchor="w", text_color=COLORS["terracotta"],
        ).pack(fill="x", padx=14, pady=(10, 3))
        ctk.CTkLabel(
            developer_note,
            text=(
                "“La inteligencia artificial requiere un poder de cómputo importante. En aras de procurar un sistema más ético "
                "para los procesos operativos de desgrabación de entrevistas, priorizando la seguridad y la ética de los datos, "
                "Jerónimo Abya Yala opta por descentralizar esta tecnología: descarga modelos de código abierto y los ejecuta de forma aislada dentro de tu computadora.\n\n"
                "El costo de esta decisión es importante tenerlo en cuenta: trasladamos el trabajo de la inteligencia artificial a tu propio equipo. "
                "Cuanto más potente sea la computadora, más rápido podrá trabajar y también podrá utilizar modelos más exigentes, que pueden ofrecer mejores resultados. "
                "Las computadoras con tarjetas gráficas NVIDIA, frecuentes en equipos usados para videojuegos, pueden aprovechar especialmente este procesamiento cuando son compatibles.\n\n"
                "También es importante prever espacio en el disco. Como referencia sencilla, la configuración automática puede necesitar aproximadamente entre 10 y 30 gigas libres, "
                "y puede ocupar más si se descargan varios modelos. Jerónimo intenta elegir una combinación razonable para cada equipo. "
                "Durante el trabajo puede aumentar el uso de memoria, energía y temperatura; por eso se recomienda utilizar computadoras relativamente recientes y, "
                "en notebooks o portátiles, mantenerlas en un lugar ventilado que favorezca la disipación del calor.\n\n"
                "Si encontrás algún inconveniente, sentite libre de comentarlo. Cada experiencia ayuda a mejorar el proceso.\n"
                "Con cariño, José Manuel.”"
            ),
            justify="left", wraplength=690, anchor="w", text_color=COLORS["text"],
            font=ctk.CTkFont(size=12 if self._compact else 13),
        ).pack(fill="x", padx=14, pady=(0, 10))
        ctk.CTkButton(project, text="Instagram · Jóvenes y Escuela Secundaria", width=265, command=lambda: _open_url(TEAM_INSTAGRAM_URL)).pack(anchor="w", padx=20, pady=(0, 14))

        modes = self._card(page, 3)
        mode_title = ctk.CTkFrame(modes, fg_color="transparent")
        mode_title.pack(fill="x", padx=20, pady=(14, 6))
        ctk.CTkLabel(mode_title, text="¿Cómo querés configurar este equipo?", font=ctk.CTkFont(size=19, weight="bold"), anchor="w").pack(side="left")
        HelpBubbleButton(mode_title, title="Automática o Avanzada", help_text="Automática es la opción recomendada para el equipo: detecta recursos y prepara componentes/modelos locales adecuados. Avanzada conserva el control manual de privacidad, modelos, Hugging Face y APIs. Ambas llegan al mismo programa; cambia cuánto debes decidir durante el primer inicio.").pack(side="left", padx=8)
        auto_box = ctk.CTkFrame(modes, corner_radius=12)
        auto_box.pack(fill="x", padx=20, pady=(4, 8))
        auto_title = ctk.CTkFrame(auto_box, fg_color="transparent")
        auto_title.pack(fill="x", padx=16, pady=(10, 2))
        ctk.CTkLabel(auto_title, text="Automática · recomendada", font=ctk.CTkFont(size=17, weight="bold"), anchor="w").pack(side="left")
        HelpBubbleButton(auto_title, title="Configuración automática", help_text="Detecta CPU, RAM, GPU/VRAM, CUDA, espacio, Ollama y modelos disponibles. Reutiliza lo instalado y prepara WhisperX/large-v2 + diarización local y el modelo de resumen recomendado. No configura APIs externas.").pack(side="left", padx=8)
        ctk.CTkLabel(
            auto_box,
            text=(
                "Analiza el equipo y prepara automáticamente la transcripción local con hablantes y el modelo de resumen recomendado. "
                "Puede descargar varios gigas y exigir bastante a la computadora; no configura servicios externos ni envía entrevistas durante esas descargas."
            ),
            justify="left", wraplength=670, anchor="w", text_color=COLORS["muted"]
        ).pack(fill="x", padx=16, pady=(0, 10))
        ctk.CTkButton(auto_box, text="Usar configuración automática", height=38, command=self._choose_automatic).pack(anchor="w", padx=16, pady=(0, 14))

        advanced_box = ctk.CTkFrame(modes, corner_radius=12)
        advanced_box.pack(fill="x", padx=20, pady=(0, 18))
        advanced_title = ctk.CTkFrame(advanced_box, fg_color="transparent")
        advanced_title.pack(fill="x", padx=16, pady=(10, 2))
        ctk.CTkLabel(advanced_title, text="Avanzada", font=ctk.CTkFont(size=17, weight="bold"), anchor="w").pack(side="left")
        HelpBubbleButton(advanced_title, title="Configuración avanzada", help_text="Recorre cada decisión manualmente: privacidad, diagnóstico, transcripción, resumen local, hablantes y APIs. Úsala si necesitas cambiar el baseline, revisar credenciales o usar servicios externos.").pack(side="left", padx=8)
        ctk.CTkLabel(
            advanced_box,
            text="Permite revisar manualmente privacidad, diagnóstico, modelos, hablantes y APIs opcionales.",
            justify="left", wraplength=670, anchor="w", text_color=COLORS["muted"]
        ).pack(fill="x", padx=16, pady=(0, 10))
        ctk.CTkButton(advanced_box, text="Configurar manualmente", height=38, fg_color=COLORS["card_alt"], text_color=COLORS["text"], command=self._choose_advanced).pack(anchor="w", padx=16, pady=(0, 14))

        # Configuración automática
        page = self._page("automatic")
        self._title(
            page,
            "Configuración automática",
            "Jerónimo Abya Yala prepara una configuración local adecuada para este equipo. Antes de descargar, muestra qué eligió y qué recursos utilizará.",
        )
        auto_card = self._card(page, 2)
        self.lbl_auto_plan = ctk.CTkLabel(auto_card, text="Comprobando este equipo…", justify="left", anchor="w", wraplength=700)
        self.lbl_auto_plan.pack(fill="x", padx=20, pady=(18, 10))
        self.auto_progress = ctk.CTkProgressBar(auto_card)
        self.auto_progress.pack(fill="x", padx=20, pady=(4, 8))
        self.auto_progress.set(0)
        self.lbl_auto_status = ctk.CTkLabel(
            auto_card,
            text=(
                "El modo automático no configura OpenAI ni otras APIs externas. Community-1 se intenta primero desde el mirror verificado del equipo; sólo si ese mirror falla se prueba el repositorio oficial de Hugging Face como contingencia."
            ),
            justify="left", anchor="w", wraplength=700, text_color=COLORS["muted"]
        )
        self.lbl_auto_status.pack(fill="x", padx=20, pady=(0, 18))

        # Privacidad
        page = self._page("privacy")
        self._title(page, "Privacidad y ética de los datos", "El procesamiento local es el camino predeterminado. Las APIs externas son opcionales y deben elegirse de forma explícita.")
        card = self._card(page, 2)
        privacy_title = ctk.CTkFrame(card, fg_color="transparent")
        privacy_title.pack(fill="x", padx=20, pady=(14, 4))
        ctk.CTkLabel(privacy_title, text="Qué permanece local por defecto", font=ctk.CTkFont(size=18, weight="bold"), anchor="w").pack(side="left")
        HelpBubbleButton(privacy_title, title="Datos locales", help_text="Audio, transcripción local, diarización y resumen con Ollama pueden permanecer dentro de la computadora. Sólo una elección explícita de API o un endpoint remoto cambia ese alcance.").pack(side="left", padx=8)
        ctk.CTkLabel(card, text="• Audio de la entrevista\n• Transcripción local\n• Diarización una vez descargados los modelos\n• Resúmenes con un modelo local", justify="left", anchor="w").pack(fill="x", padx=20, pady=(0, 14))
        ctk.CTkLabel(card, text="En Automática, Community-1 se descarga primero desde el mirror verificado del equipo. Hugging Face queda como respaldo de instalación si ese mirror no está disponible. Ninguna de esas fuentes recibe el audio de la entrevista durante la diarización local.", justify="left", wraplength=690, anchor="w", text_color=COLORS["muted"]).pack(fill="x", padx=20, pady=(0, 18))
        self.chk_privacy = ctk.CTkCheckBox(page, text="Entiendo que las APIs externas son opcionales y que el programa me avisará antes de enviar audio o texto fuera del equipo.", variable=self._privacy_var)
        self.chk_privacy.grid(row=3, column=0, sticky="w", pady=15)

        # Sistema
        page = self._page("system")
        self._title(page, "Comprobación del equipo", "Jerónimo Abya Yala revisa hardware, motores, espacio y herramientas. No hace benchmark de tus entrevistas ni sube información del equipo.")
        self.system_card = self._card(page, 2)
        system_head = ctk.CTkFrame(self.system_card, fg_color="transparent")
        system_head.pack(fill="x", padx=20, pady=(12, 0))
        ctk.CTkLabel(system_head, text="Estado detectado", font=ctk.CTkFont(size=16, weight="bold")).pack(side="left")
        HelpBubbleButton(system_head, title="Estado detectado", help_text="Cada línea indica si una pieza necesaria está disponible. FFmpeg prepara audio; WhisperX transcribe/alinea; CUDA permite usar GPU; el motor privado de texto ejecuta el análisis local; la caché de Hugging Face indica si los modelos gated ya están descargados.").pack(side="left", padx=8)
        self.lbl_system = ctk.CTkLabel(self.system_card, text="Comprobando…", justify="left", anchor="w", wraplength=self._wraplength)
        self.lbl_system.pack(fill="x", padx=20, pady=(6, 14))
        ctk.CTkButton(page, text="Volver a comprobar", width=150, command=self._refresh_system_async).grid(row=3, column=0, sticky="w", pady=10)

        # Transcripción
        page = self._page("transcription")
        self._title(page, "Transcripción local", "El diagnóstico recomienda una configuración, pero no cambia silenciosamente el motor validado. El baseline de calidad con hablantes sigue usando WhisperX + large-v2.")
        self.transcription_card = self._card(page, 2)
        trans_head = ctk.CTkFrame(self.transcription_card, fg_color="transparent")
        trans_head.pack(fill="x", padx=20, pady=(12, 0))
        ctk.CTkLabel(trans_head, text="Configuración de transcripción", font=ctk.CTkFont(size=16, weight="bold")).pack(side="left")
        HelpBubbleButton(trans_head, title="Configuración de transcripción", help_text="WhisperX large-v2 permanece como baseline con hablantes. El diagnóstico puede recomendar aceleración y confirmar si el modelo está en caché, pero no sustituye silenciosamente el motor validado.").pack(side="left", padx=8)
        self.lbl_transcription = ctk.CTkLabel(self.transcription_card, text="Esperando diagnóstico…", justify="left", anchor="w", wraplength=self._wraplength)
        self.lbl_transcription.pack(fill="x", padx=20, pady=(6, 8))
        ctk.CTkButton(self.transcription_card, text="Abrir gestor de modelos", width=175, command=self._open_models).pack(anchor="w", padx=20, pady=(4, 18))

        # Resumen local
        page = self._page("local")
        self._title(page, "Resumen local", "Para analizar y resumir entrevistas sin enviarlas a una API, Jerónimo Abya Yala usa un runtime privado de Ollama. El modelo se ajusta a RAM/VRAM y el resumen se genera con análisis jerárquico y verificación final.")
        self.local_card = self._card(page, 2)
        local_head = ctk.CTkFrame(self.local_card, fg_color="transparent")
        local_head.pack(fill="x", padx=20, pady=(12, 0))
        ctk.CTkLabel(local_head, text="Modelo de resumen", font=ctk.CTkFont(size=16, weight="bold")).pack(side="left")
        HelpBubbleButton(local_head, title="Modelo de resumen local", help_text="El modelo se ejecuta con el runtime privado administrado por Jerónimo. No hace falta instalar Ollama aparte. Jerónimo recomienda un tamaño según RAM/VRAM y prioriza calidad; un modelo mayor puede tardar más. La descarga sólo ocurre cuando la aceptas o cuando usas el modo Automático.").pack(side="left", padx=8)
        self.lbl_local = ctk.CTkLabel(self.local_card, text="Esperando diagnóstico…", justify="left", anchor="w", wraplength=self._wraplength)
        self.lbl_local.pack(fill="x", padx=20, pady=(6, 8))
        local_actions = ctk.CTkFrame(self.local_card, fg_color="transparent")
        local_actions.pack(fill="x", padx=20, pady=(5, 8))
        self.btn_download_recommended = ctk.CTkButton(local_actions, text="Descargar modelo recomendado", width=205, command=self._download_recommended_text)
        self.btn_download_recommended.pack(side="left", padx=(0, 8))
        ctk.CTkButton(local_actions, text="Abrir gestor de modelos", command=self._open_models).pack(side="left", padx=(0, 8))
        self.local_progress = ctk.CTkProgressBar(self.local_card)
        self.local_progress.pack(fill="x", padx=20, pady=(2, 6))
        self.local_progress.set(0)
        self.btn_cancel_model = ctk.CTkButton(self.local_card, text="Cancelar descarga", width=140, state="disabled", command=self._cancel_recommended_text)
        self.btn_cancel_model.pack(anchor="e", padx=20, pady=(0, 18))

        # HF
        page = self._page("huggingface")
        self._title(page, "Separación de hablantes", "WhisperX necesita acceso a modelos de pyannote. En versiones actuales, Community-1 es un modelo gated: debes aceptar sus condiciones y usar un token de Hugging Face para descargarlo.")
        steps = self._card(page, 2)
        hf_head = ctk.CTkFrame(steps, fg_color="transparent")
        hf_head.pack(fill="x", padx=20, pady=(10, 2))
        ctk.CTkLabel(hf_head, text="Acceso a Community-1", font=ctk.CTkFont(size=16, weight="bold")).pack(side="left")
        HelpBubbleButton(hf_head, title="Hugging Face y Community-1", help_text="Hugging Face sólo autentica la descarga inicial del modelo gated. La diarización posterior se ejecuta localmente. El token se guarda en el almacén seguro del sistema si eliges guardarlo y no se escribe en los archivos del proyecto.").pack(side="left", padx=8)
        ctk.CTkLabel(steps, text="1. Crea/inicia sesión y acepta Community-1.\n2. Crea un token de lectura o fine-grained.\n3. Guárdalo y prueba el acceso.\n4. Una vez cacheado el modelo, la diarización puede funcionar localmente.", justify="left", anchor="w", wraplength=self._wraplength).pack(fill="x", padx=20, pady=(2, 12))
        links = ctk.CTkFrame(page, fg_color="transparent")
        links.grid(row=3, column=0, sticky="w", pady=(0, 10))
        ctk.CTkButton(links, text="Crear cuenta", width=105, command=lambda: _open_url(HF_SIGNUP_URL)).pack(side="left", padx=(0, 6))
        ctk.CTkButton(links, text="Aceptar Community-1", width=155, command=lambda: _open_url(HF_PYANNOTE_URL)).pack(side="left", padx=6)
        ctk.CTkButton(links, text="Crear token", width=105, command=lambda: _open_url(HF_TOKEN_URL)).pack(side="left", padx=6)
        hf_card = self._card(page, 4)
        self.hf_entry = ctk.CTkEntry(hf_card, show="•", placeholder_text="hf_…")
        self.hf_entry.pack(fill="x", padx=20, pady=(18, 8))
        hf_actions = ctk.CTkFrame(hf_card, fg_color="transparent")
        hf_actions.pack(fill="x", padx=20)
        ctk.CTkButton(hf_actions, text="Guardar seguro", width=120, command=self._save_hf).pack(side="left", padx=(0, 8))
        ctk.CTkButton(hf_actions, text="Probar acceso", width=120, command=self._test_hf_async).pack(side="left")
        self.lbl_hf = ctk.CTkLabel(hf_card, text=credential_status_text("huggingface"), justify="left", anchor="w", wraplength=690, text_color=COLORS["muted"])
        self.lbl_hf.pack(fill="x", padx=20, pady=(8, 18))

        # APIs
        page = self._page("apis")
        self._title(page, "APIs externas — opcional", "No necesitas una API para usar Jerónimo Abya Yala localmente. Configúrala sólo si quieres transcripción o procesamiento de texto remoto. Las pruebas usan una frase fija y no contenido de entrevistas; según el proveedor pueden generar una llamada facturable mínima.")
        oa = self._card(page, 2)
        oa_head = ctk.CTkFrame(oa, fg_color="transparent")
        oa_head.pack(fill="x", padx=20, pady=(12, 4))
        ctk.CTkLabel(oa_head, text="OpenAI", font=ctk.CTkFont(size=18, weight="bold"), anchor="w").pack(side="left")
        HelpBubbleButton(oa_head, title="OpenAI", help_text="Opcional. Puede usarse para transcripción o procesamiento de texto remoto. Si lo eliges, el audio o texto correspondiente sale de esta computadora. La key puede guardarse en Windows Credential Manager mediante keyring.").pack(side="left", padx=8)
        ctk.CTkButton(oa, text="Abrir página de API keys", width=165, command=lambda: _open_url(OPENAI_API_KEYS_URL)).pack(anchor="w", padx=20, pady=(0, 8))
        self.openai_key = ctk.CTkEntry(oa, show="•", placeholder_text="API key (opcional)")
        self.openai_key.pack(fill="x", padx=20, pady=5)
        self.openai_model = ctk.CTkEntry(oa, placeholder_text="Modelo de texto")
        self.openai_model.insert(0, getattr(self.app, "openai_chat_model", "gpt-4.1-mini"))
        self.openai_model.pack(fill="x", padx=20, pady=5)
        oa_actions = ctk.CTkFrame(oa, fg_color="transparent")
        oa_actions.pack(fill="x", padx=20, pady=(5, 8))
        ctk.CTkButton(oa_actions, text="Guardar seguro", width=120, command=self._save_openai).pack(side="left", padx=(0, 8))
        ctk.CTkButton(oa_actions, text="Probar", width=90, command=self._test_openai_async).pack(side="left")
        self.lbl_openai = ctk.CTkLabel(oa, text=credential_status_text("openai"), anchor="w", text_color=COLORS["muted"])
        self.lbl_openai.pack(fill="x", padx=20, pady=(0, 18))

        compat = self._card(page, 3)
        compat_head = ctk.CTkFrame(compat, fg_color="transparent")
        compat_head.pack(fill="x", padx=20, pady=(12, 4))
        ctk.CTkLabel(compat_head, text="API compatible con OpenAI", font=ctk.CTkFont(size=18, weight="bold"), anchor="w").pack(side="left")
        HelpBubbleButton(compat_head, title="API compatible", help_text="Permite usar un servidor que implemente el endpoint chat/completions compatible con OpenAI. Una URL localhost puede ser local; una IP LAN o dominio remoto implica que el texto se envía a otro equipo/servicio. Debes indicar el nombre exacto del modelo.").pack(side="left", padx=8)
        self.compat_url = ctk.CTkEntry(compat, placeholder_text="URL base")
        self.compat_url.insert(0, getattr(self.app, "compatible_api_url", DEFAULT_OPENAI_COMPATIBLE_URL))
        self.compat_url.pack(fill="x", padx=20, pady=5)
        self.compat_model = ctk.CTkEntry(compat, placeholder_text="Nombre exacto del modelo")
        self.compat_model.insert(0, getattr(self.app, "compatible_model", ""))
        self.compat_model.pack(fill="x", padx=20, pady=5)
        self.compat_key = ctk.CTkEntry(compat, show="•", placeholder_text="API key (opcional en servidores locales)")
        self.compat_key.pack(fill="x", padx=20, pady=5)
        compat_actions = ctk.CTkFrame(compat, fg_color="transparent")
        compat_actions.pack(fill="x", padx=20, pady=(5, 18))
        ctk.CTkButton(compat_actions, text="Guardar key", width=110, command=self._save_compatible).pack(side="left", padx=(0, 8))
        ctk.CTkButton(compat_actions, text="Probar", width=90, command=self._test_compatible_async).pack(side="left")

        # Fin
        page = self._page("done")
        self._title(page, "Jerónimo Abya Yala está preparado", "Puedes volver a este asistente desde la barra lateral. Nada de lo que configures aquí impide cambiar después entre procesamiento local y APIs.")
        done = self._card(page, 2)
        self.lbl_done = ctk.CTkLabel(done, text="", justify="left", anchor="w", wraplength=700)
        self.lbl_done.pack(fill="x", padx=20, pady=18)
        ctk.CTkLabel(page, text="Recomendación para el primer uso: prueba una entrevista corta antes de procesar un corpus completo y revisa hablantes/timestamps contra el audio.", justify="left", wraplength=700, anchor="w", text_color=COLORS["muted"]).grid(row=3, column=0, sticky="ew", pady=12)

    def _configure_sidebar_for_mode(self, mode: str):
        for label in self.step_labels.values():
            try:
                label.pack_forget()
            except Exception:
                pass
        if mode == "repair":
            keys = ("automatic",)
        elif mode == "automatic":
            keys = ("intro", "automatic")
        elif mode == "advanced":
            keys = ("intro", "privacy", "system", "transcription", "local", "huggingface", "apis", "done")
        else:
            keys = ("intro",)
        for key in keys:
            self.step_labels[key].pack(fill="x", padx=22, pady=2)

    def _choose_automatic(self):
        self._setup_mode = "automatic"
        self._configure_sidebar_for_mode("automatic")
        self._privacy_var.set(True)
        self._automatic_complete = False
        self._automatic_result = {}
        self._show_step(STEP_KEYS.index("automatic"))
        if self.report is None:
            self._refresh_system_async()
        else:
            self._render_automatic_plan()

    def _choose_advanced(self):
        self._setup_mode = "advanced"
        self._configure_sidebar_for_mode("advanced")
        self._show_step(STEP_KEYS.index("privacy"))

    def _render_automatic_plan(self):
        if self.report is None:
            self.lbl_auto_plan.configure(text="Comprobando este equipo…")
            return
        models = list(getattr(self.app, "ollama_models", []) or [])
        plan = build_automatic_plan(self.report, models)
        items = [
            "Configuración elegida por Jerónimo Abya Yala:",
            f"• Transcripción + hablantes: WhisperX / Whisper {plan.asr_model} (baseline validado)",
            f"• Aceleración: {'CUDA/GPU' if plan.cuda else 'CPU / sin CUDA confirmada'}",
            f"• Análisis y resumen local: {plan.text_model}",
            "• APIs externas: no se configuran",
            "• Modelo de hablantes: mirror verificado de Jerónimo (Google Drive) → Hugging Face oficial si hace falta",
            "",
            "Descargas que faltan:",
            f"• Runtime de transcripción con hablantes: {'sí' if plan.needs_diarization_runtime else 'no'} · aprox. {plan.diarization_runtime_size_gb:.1f} GB si falta",
            f"• Whisper {plan.asr_model}: {'sí' if plan.needs_asr_model else 'no'} · aprox. {plan.asr_model_size_gb:.1f} GB si falta",
            f"• Community-1 local verificado: {'descargar/preparar' if plan.needs_pyannote_model else 'ya disponible'}",
            f"• Motor privado de texto: {'descargar/preparar' if plan.needs_ollama else 'ya disponible'}",
            f"• {plan.text_model}: {'sí' if plan.needs_text_model else 'no'} · aprox. {plan.text_model_size_gb:.1f} GB si falta",
        ]
        if plan.free_disk_gb:
            items.append(f"• Espacio libre actual: {plan.free_disk_gb:.1f} GB")
        if plan.warnings:
            items.extend(["", "Advertencias:"] + [f"! {item}" for item in plan.warnings])
        items.extend([
            "",
            "El programa base no incluye PyTorch/WhisperX ni modelos pesados. Se descargan aquí sólo si esta configuración los necesita.",
            "Durante la instalación puede haber uso intenso de red, disco, CPU/GPU y memoria. Las entrevistas no se suben durante esta preparación.",
        ])
        self.lbl_auto_plan.configure(text="\n".join(items))
        if not self._automatic_complete:
            if self.repair_mode:
                reasons = "\n".join(f"• {item}" for item in self.repair_reasons)
                self.lbl_auto_status.configure(
                    text=(
                        "Actualización técnica requerida. Se reutilizará todo lo que siga siendo válido y sólo se reparará lo faltante."
                        + (f"\n\nMotivos detectados:\n{reasons}" if reasons else "")
                        + "\n\nPulsa ‘Reparar configuración’ para continuar."
                    )
                )
            else:
                self.lbl_auto_status.configure(text="Pulsa “Instalar configuración recomendada” para continuar. Se omiten automáticamente las descargas que ya estén presentes.")

    def _start_automatic_setup(self):
        if self._busy:
            return
        if self.report is None:
            self.lbl_auto_status.configure(text="Todavía se está comprobando el equipo. Esperá unos segundos y volvé a intentar.")
            self._refresh_system_async()
            return
        self._automatic_cancel.clear()
        self._set_busy(True, "Preparando configuración automática…")
        self.auto_progress.configure(mode="determinate")
        self.auto_progress.set(0)
        self.lbl_auto_status.configure(text="Iniciando configuración local…")
        threading.Thread(target=self._automatic_setup_worker, daemon=True).start()

    def _automatic_setup_worker(self):
        def progress(message: str, fraction: float | None):
            if self._closing:
                return
            def apply():
                if self._closing:
                    return
                if fraction is None:
                    self.auto_progress.configure(mode="indeterminate")
                    self.auto_progress.start()
                else:
                    self.auto_progress.stop()
                    self.auto_progress.configure(mode="determinate")
                    self.auto_progress.set(max(0.0, min(1.0, float(fraction))))
                self.lbl_auto_status.configure(text=message)
                self.lbl_footer.configure(text=message)
            try:
                self.after(0, apply)
            except Exception:
                pass
        try:
            result = run_automatic_setup(
                self.report,
                ollama_url=getattr(self.app, "ollama_url", DEFAULT_OLLAMA_URL),
                progress=progress,
                cancel_event=self._automatic_cancel,
            )
            self._automatic_result = result
            self._automatic_complete = True
            try:
                self.app.ollama_model = str(result.get("text_model", "") or getattr(self.app, "ollama_model", ""))
                self.app.ollama_url = str(result.get("ollama_url", "") or DEFAULT_OLLAMA_URL)
                self.app.text_engine = "ollama"
            except Exception:
                pass
            if not self._closing:
                self.after(0, self._automatic_success)
        except Exception as exc:
            message = str(exc)
            if not self._closing:
                try:
                    self.after(0, lambda m=message: self._automatic_failure(m))
                except Exception:
                    pass
        finally:
            if not self._closing:
                try:
                    self.after(0, self._set_busy, False, "")
                except Exception:
                    pass

    def _automatic_success(self):
        self.auto_progress.stop()
        self.auto_progress.configure(mode="determinate")
        self.auto_progress.set(1)
        text_model = str(self._automatic_result.get("text_model", "modelo recomendado"))
        self.lbl_auto_status.configure(
            text=(
                "Configuración automática completada.\n"
                "• Transcripción y separación de hablantes: local\n"
                f"• Análisis y resumen: {text_model} local\n"
                "• APIs externas: no configuradas\n\n"
                "Pulsa Finalizar para empezar a trabajar."
            )
        )
        self.btn_next.configure(text="Finalizar", state="normal")
        self.after(500, self._refresh_system_async)

    def _automatic_failure(self, message: str):
        self.auto_progress.stop()
        self.auto_progress.configure(mode="determinate")
        suffix = (
            "\n\nPodés reintentar. Jerónimo no permitirá entrar hasta que la reparación técnica termine."
            if self.repair_mode else
            "\n\nPodés reintentar o volver a Bienvenida y usar la configuración avanzada."
        )
        self.lbl_auto_status.configure(
            text=f"No se pudo completar la configuración automática:\n{message}{suffix}"
        )
        self.btn_next.configure(text="Reintentar", state="normal")

    def _show_step(self, index: int):
        self._step_index = max(0, min(index, len(STEP_KEYS) - 1))
        key = STEP_KEYS[self._step_index]
        self.pages[key].lift()
        for k, label in self.step_labels.items():
            active = k == key
            label.configure(
                font=ctk.CTkFont(size=11 if self._compact else 12, weight="bold" if active else "normal"),
                text_color=COLORS["sidebar_text"] if active else COLORS["sidebar_muted"],
                fg_color=COLORS["nav_active"] if active else "transparent",
            )
        self.btn_back.configure(state="disabled" if (self._step_index == 0 or self.repair_mode) else "normal")
        if key == "intro":
            self.btn_next.configure(text="Elegí un modo", state="disabled")
        elif key == "automatic":
            if self._automatic_complete:
                action_text = "Finalizar"
            else:
                action_text = "Reparar configuración" if self.repair_mode else "Instalar configuración recomendada"
            self.btn_next.configure(text=action_text, state="normal")
            self._render_automatic_plan()
        elif key == "done":
            self.btn_next.configure(text="Terminar", state="normal")
        else:
            self.btn_next.configure(text="Continuar", state="normal")
        # En reparación no tocar el estado persistido hasta terminar con éxito.
        # Así, si el usuario cierra o una descarga falla, el próximo inicio vuelve
        # a ofrecer reparación y no degrada un estado legacy a "primer inicio".
        if not self.repair_mode:
            mark_step(key)
        if key == "done":
            self._render_done()

    def _next(self):
        if self._busy:
            return
        key = STEP_KEYS[self._step_index]
        if key == "intro":
            return
        if key == "automatic":
            if not self._automatic_complete:
                self._start_automatic_setup()
                return
            complete_onboarding(privacy_acknowledged=True, preferred_text_mode="local", setup_mode="automatic")
            callback = self.on_complete
            app = self.app
            self.destroy()
            # La ventana principal sólo puede mostrarse DESPUÉS de destruir el
            # onboarding. Esto evita cualquier solapamiento visual al finalizar.
            if callable(callback):
                try:
                    app.after_idle(callback)
                except Exception:
                    callback()
            return
        if key == "privacy" and not self._privacy_var.get():
            self.lbl_footer.configure(text="Marca la casilla de privacidad para continuar.")
            try:
                self.chk_privacy.focus_set()
            except Exception:
                pass
            return
        if key == "done":
            complete_onboarding(privacy_acknowledged=True, preferred_text_mode="local", setup_mode="advanced")
            callback = self.on_complete
            app = self.app
            self.destroy()
            if callable(callback):
                try:
                    app.after_idle(callback)
                except Exception:
                    callback()
            return
        self._show_step(self._step_index + 1)

    def _back(self):
        if self._busy or self.repair_mode:
            return
        key = STEP_KEYS[self._step_index]
        if key in {"automatic", "privacy"}:
            self._show_step(STEP_KEYS.index("intro"))
            return
        self._show_step(self._step_index - 1)

    def _close_event(self, _event=None):
        return self._request_application_exit()

    def _force_application_exit(self):
        # Último recurso solicitado para instalaciones/descargas que no liberan
        # el proceso. os._exit evita quedar atrapado en finalizadores de terceros.
        os._exit(0)

    def _graceful_application_exit(self):
        app = getattr(self, "app", None)
        try:
            self.auto_progress.stop()
        except Exception:
            pass
        try:
            self.grab_release()
        except Exception:
            pass
        try:
            self._unlock_parent_window()
        except Exception:
            pass
        try:
            if self.winfo_exists():
                self.destroy()
        except Exception:
            pass
        if app is not None:
            try:
                app.quit()
            except Exception:
                pass
            try:
                if app.winfo_exists():
                    app.destroy()
            except Exception:
                pass

    def _request_application_exit(self):
        """Cierra Jerónimo desde el onboarding sin permitir saltarlo.

        Primero solicita cancelación de operaciones, libera el modal y destruye la
        raíz normalmente. Un watchdog daemon fuerza la salida sólo si, cinco
        segundos después, algún componente externo todavía mantiene vivo Python.
        """
        if getattr(self, "_closing", False):
            return "break"
        self._closing = True
        self._automatic_cancel.set()
        self._model_download_cancel.set()
        try:
            self.lbl_footer.configure(text="Cerrando Jerónimo…")
            self.btn_next.configure(state="disabled")
            self.btn_back.configure(state="disabled")
            self.btn_window_close.configure(state="disabled")
        except Exception:
            pass

        # El Timer es daemon: no retrasa un cierre normal. Sólo actúa si otra
        # librería/hilo no permite que el proceso termine por sí mismo.
        try:
            timer = threading.Timer(5.0, self._force_application_exit)
            timer.daemon = True
            timer.start()
            self._force_exit_timer = timer
        except Exception:
            self._force_exit_timer = None

        try:
            self.after_idle(self._graceful_application_exit)
        except Exception:
            self._graceful_application_exit()
        return "break"

    def _close_without_complete(self):
        # Alias conservado para compatibilidad con versiones/pruebas anteriores.
        return self._request_application_exit()

    def _set_busy(self, value: bool, text: str = ""):
        self._busy = bool(value)
        key = STEP_KEYS[self._step_index]
        if value:
            self.btn_next.configure(state="disabled")
            self.btn_back.configure(state="disabled")
        else:
            self.btn_back.configure(state="disabled" if (self._step_index == 0 or self.repair_mode) else "normal")
            if key == "intro":
                self.btn_next.configure(state="disabled", text="Elegí un modo")
            elif key == "automatic":
                action_text = "Finalizar" if self._automatic_complete else ("Reparar configuración" if self.repair_mode else "Instalar configuración recomendada")
                self.btn_next.configure(state="normal", text=action_text)
            else:
                self.btn_next.configure(state="normal")
        self.lbl_footer.configure(text=text)

    def _refresh_system_async(self):
        if self._busy:
            return
        self._set_busy(True, "Comprobando equipo…")
        self.lbl_system.configure(text="Comprobando hardware, espacio y motores…")
        threading.Thread(target=self._refresh_system_worker, daemon=True).start()

    def _refresh_system_worker(self):
        try:
            report = run_system_diagnostics(getattr(self.app, "output_dir", Path.cwd()), getattr(self.app, "ollama_url", DEFAULT_OLLAMA_URL))
            inv = ollama_inventory(getattr(self.app, "ollama_url", DEFAULT_OLLAMA_URL))
            self.report = report
            self.after(0, self._apply_system, report, list(inv.keys()))
        except Exception as exc:
            message = f"No se pudo completar el diagnóstico: {exc}"
            self.after(0, lambda m=message: self.lbl_system.configure(text=m))
        finally:
            self.after(0, self._set_busy, False, "")

    def _apply_system(self, report, models: list[str]):
        rec = str((report.recommendations or {}).get("text_primary") or "qwen3:4b")
        ready = build_readiness(report, ollama_models=models, recommended_model=rec)
        symbols = {"ok": "✓", "warning": "!", "missing": "×", "optional": "–"}
        lines = [f"{symbols.get(i.status, '•')} {i.label}: {i.detail}" for i in ready.items]
        if ready.blocking:
            lines.append("\nHay componentes necesarios para algunos perfiles que todavía faltan. Jerónimo Abya Yala no los instalará silenciosamente.")
        self.lbl_system.configure(text="\n".join(lines))
        cached = set(getattr(report, "hf_cached_models", []) or [])
        baseline_cached = any("faster-whisper-large-v2" in item.lower() for item in cached)
        worker = getattr(report, "diarization_worker", {}) or {}
        wx = getattr(report, "whisperx_version", "") or ""
        managed = diarization_runtime_status()
        if worker.get("available") or managed.ready:
            runtime_text = "runtime local de hablantes disponible"
        elif managed.required:
            runtime_text = "runtime pesado aún no descargado; Automática lo prepara bajo demanda"
        else:
            runtime_text = f"WhisperX {wx}" if wx else "WhisperX no detectado en este proceso"
        self.lbl_transcription.configure(text=(
            f"Recomendación de calidad: WhisperX + large-v2\n"
            f"Runtime: {runtime_text}\n"
            f"Modelo large-v2: {'detectado en caché' if baseline_cached else 'todavía no detectado en caché'}\n\n"
            f"{(report.recommendations or {}).get('asr', '')}"
        ))
        ollama = report.ollama or {}
        if ollama.get("available"):
            has = ollama_model_present(models, rec)
            self.lbl_local.configure(text=f"Modelo recomendado: {rec}\nMotor privado de texto: disponible\nModelo recomendado: {'instalado' if has else 'todavía no instalado'}\n\nJerónimo usa contexto ampliado y análisis jerárquico. El gestor permite probar rendimiento sin usar entrevistas.")
        else:
            self.lbl_local.configure(text=f"Modelo recomendado según este equipo: {rec}\nEl motor privado de texto todavía no está preparado. Jerónimo lo descargará automáticamente al instalar el modelo o al usar el modo Automático.\n\nNo hace falta instalar Ollama por separado.")
        try:
            self.app._diag_report = report
            self.app.ollama_models = models
            if not getattr(self.app, "ollama_model", ""):
                self.app.ollama_model = rec
        except Exception:
            pass
        if self._setup_mode == "automatic" or STEP_KEYS[self._step_index] == "automatic":
            try:
                self._render_automatic_plan()
            except Exception:
                pass

    def _open_models(self):
        try:
            self.grab_release()
            self._unlock_parent_window()
            try:
                self.attributes("-topmost", False)
            except Exception:
                pass
            win = open_model_manager(self.app, getattr(self.app, "ollama_url", DEFAULT_OLLAMA_URL), str(getattr(self.app, "output_dir", Path.cwd())))
            self.wait_window(win)
        except Exception as exc:
            mb.showerror("Modelos locales", str(exc), parent=self)
        finally:
            try:
                if self.winfo_exists():
                    self._lock_parent_window()
                    self.attributes("-topmost", True)
                    self.lift()
                    self.grab_set()
                    self._refresh_system_async()
            except Exception:
                pass

    def _recommended_text_spec(self):
        if self.report is None:
            return None
        model_id = str((self.report.recommendations or {}).get("text_primary") or "qwen3:4b")
        return next((spec for spec in TEXT_MODELS if spec.model_id == model_id), None)

    def _select_recommended_text_model(self, spec) -> None:
        self.app.ollama_model = spec.model_id
        models = list(getattr(self.app, "ollama_models", []) or [])
        if spec.model_id not in models:
            models.append(spec.model_id)
        self.app.ollama_models = sorted(set(models))
        refresh = getattr(self.app, "_refresh_text_model_label", None)
        if callable(refresh):
            refresh()

    def _download_recommended_text(self):
        if self._busy:
            return
        spec = self._recommended_text_spec()
        if spec is None:
            mb.showwarning("Modelo local", "Primero completa el diagnóstico del equipo.", parent=self)
            return
        inv = ollama_inventory(getattr(self.app, "ollama_url", DEFAULT_OLLAMA_URL))
        if ollama_model_present(inv, spec.model_id):
            self._select_recommended_text_model(spec)
            self.lbl_local.configure(text=f"{spec.display_name}: ya estaba instalado y quedó seleccionado como modelo recomendado.")
            mb.showinfo("Modelo local", f"{spec.display_name} ya está instalado y quedó seleccionado.", parent=self)
            return
        if not mb.askyesno(
            "Descargar modelo local",
            f"Jerónimo Abya Yala recomienda {spec.display_name} para este equipo.\n\n"
            f"Descarga aproximada del modelo: {spec.approx_size_gb:.1f} GB.\n"
            "Si el motor privado todavía no existe, Jerónimo descargará también su runtime standalone.\n"
            "Durante el uso puede consumir CPU/GPU y memoria de forma intensiva.\n\n¿Descargar y preparar ahora?",
            parent=self,
        ):
            return
        self._model_download_cancel.clear()
        self._set_busy(True, f"Descargando {spec.display_name}…")
        self.btn_cancel_model.configure(state="normal")
        self.local_progress.stop(); self.local_progress.configure(mode="determinate"); self.local_progress.set(0)
        threading.Thread(target=self._download_recommended_text_worker, args=(spec,), daemon=True).start()

    def _download_recommended_text_worker(self, spec):
        def progress(event):
            fraction = event.get("fraction")
            status = str(event.get("status", "descargando"))
            completed = int(event.get("completed", 0) or 0)
            total = int(event.get("total", 0) or 0)
            def apply():
                if fraction is None:
                    self.local_progress.configure(mode="indeterminate"); self.local_progress.start()
                else:
                    self.local_progress.stop(); self.local_progress.configure(mode="determinate"); self.local_progress.set(max(0.0, min(1.0, float(fraction))))
                detail = status
                if total:
                    detail += f" · {completed / (1024**3):.1f} / {total / (1024**3):.1f} GB"
                self.lbl_local.configure(text=f"{spec.display_name}: {detail}")
            self.after(0, apply)
        try:
            pull_ollama_model(spec.model_id, getattr(self.app, "ollama_url", DEFAULT_OLLAMA_URL), progress=progress, cancel_event=self._model_download_cancel)
            self.after(0, self._select_recommended_text_model, spec)
            self.after(0, lambda: self.lbl_local.configure(text=f"{spec.display_name}: instalado y seleccionado como modelo recomendado."))
        except Exception as exc:
            msg = str(exc)
            self.after(0, lambda m=msg: mb.showerror("Descarga del modelo", m, parent=self))
        finally:
            self.after(0, lambda: self.btn_cancel_model.configure(state="disabled"))
            self.after(0, self._set_busy, False, "")
            self.after(350, self._refresh_system_async)

    def _cancel_recommended_text(self):
        self._model_download_cancel.set()
        self.lbl_footer.configure(text="Cancelación solicitada…")

    def _save_hf(self):
        value = self.hf_entry.get().strip()
        if not value:
            mb.showwarning("Hugging Face", "Pega un token antes de guardarlo.", parent=self)
            return
        try:
            set_secure_secret("huggingface", value)
            self.hf_entry.delete(0, "end")
            self.lbl_hf.configure(text="Token guardado en el almacén seguro del sistema. Pulsa Probar acceso.")
        except Exception as exc:
            mb.showerror("Hugging Face", str(exc), parent=self)

    def _test_hf_async(self):
        if self._busy:
            return
        explicit = self.hf_entry.get().strip()
        self._set_busy(True, "Comprobando Hugging Face…")
        self.lbl_hf.configure(text="Validando token y acceso al modelo…")
        threading.Thread(target=self._test_hf_worker, args=(explicit,), daemon=True).start()

    def _test_hf_worker(self, explicit: str):
        result = validate_huggingface(explicit_token=explicit, test_model_access=True)
        self._hf_test_ok = bool(result.get("ok"))
        self.after(0, lambda: self.lbl_hf.configure(text=str(result.get("message", ""))))
        self.after(0, self._set_busy, False, "")

    def _save_openai(self):
        key = self.openai_key.get().strip()
        if not key:
            mb.showwarning("OpenAI", "Escribe una API key antes de guardarla.", parent=self)
            return
        try:
            set_secure_secret("openai", key)
            self.openai_key.delete(0, "end")
            self.lbl_openai.configure(text="Credencial guardada en el almacén seguro del sistema.")
        except Exception as exc:
            mb.showerror("OpenAI", str(exc), parent=self)

    def _test_openai_async(self):
        key = self.openai_key.get().strip() or resolve_secret("openai")
        model = self.openai_model.get().strip()
        settings = TextProviderSettings(engine="openai", api_key=key, model=model)
        self._test_provider_async(settings, self.lbl_openai)

    def _save_compatible(self):
        key = self.compat_key.get().strip()
        if not key:
            mb.showwarning("API compatible", "La key está vacía. Si tu servidor local no usa clave, no es necesario guardarla.", parent=self)
            return
        try:
            set_secure_secret("openai_compatible", key)
            self.compat_key.delete(0, "end")
            mb.showinfo("API compatible", "Credencial guardada de forma segura.", parent=self)
        except Exception as exc:
            mb.showerror("API compatible", str(exc), parent=self)

    def _test_compatible_async(self):
        settings = TextProviderSettings(
            engine="openai_compatible",
            base_url=self.compat_url.get().strip(),
            api_key=self.compat_key.get().strip() or resolve_secret("openai_compatible"),
            model=self.compat_model.get().strip(),
        )
        self._test_provider_async(settings, None)

    def _test_provider_async(self, settings, label):
        if self._busy:
            return
        self._set_busy(True, "Probando conexión con texto fijo…")
        if label is not None:
            label.configure(text="Probando conexión con una frase fija; no se envía ninguna entrevista…")

        def worker():
            try:
                result = test_provider(settings)
                message = str(result.get("message", "Conexión correcta."))
                self.after(0, lambda: label.configure(text=message) if label is not None else mb.showinfo("API compatible", message, parent=self))
                if settings.engine == "openai":
                    self.app.openai_chat_model = settings.model
                else:
                    self.app.compatible_api_url = settings.base_url
                    self.app.compatible_model = settings.model
            except Exception as exc:
                message = str(exc)
                self.after(0, lambda m=message: label.configure(text=f"Error: {m}") if label is not None else mb.showerror("API compatible", m, parent=self))
            finally:
                self.after(0, self._set_busy, False, "")
        threading.Thread(target=worker, daemon=True).start()

    def _render_done(self):
        if self.report is None:
            self.lbl_done.configure(text="Configuración básica completa. El diagnóstico puede volver a ejecutarse desde Jerónimo Abya Yala.")
            return
        rec = str((self.report.recommendations or {}).get("text_primary") or "qwen3:4b")
        hf = credential_status_text("huggingface")
        oa = credential_status_text("openai")
        ollama = "disponible" if (self.report.ollama or {}).get("available") else "no disponible todavía"
        self.lbl_done.configure(text=(
            f"Transcripción: local-first\n"
            f"Resumen local recomendado: {rec} · Ollama {ollama}\n"
            f"Hugging Face: {hf}\n"
            f"OpenAI: {oa}\n\n"
            "Las APIs externas siguen siendo opcionales. Antes de cada trabajo Jerónimo Abya Yala muestra si audio o texto saldrán de este equipo."
        ))
