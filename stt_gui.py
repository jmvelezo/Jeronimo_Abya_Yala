from __future__ import annotations
import threading
from pathlib import Path
import tkinter.filedialog as fd
import tkinter.messagebox as mb

import customtkinter as ctk

from core_transcriber import (
    TranscriberConfig,
    run_batch,
    run_analysis_only,
    load_api_key_from_env,
)

ctk.set_appearance_mode("system")
ctk.set_default_color_theme("blue")

LANG_OPTIONS = {
    "Español (Argentina)": "es",
    "Español (México)": "es",
    "Español (España)": "es",
    "English": "en",
    "Português (Brasil)": "pt",
    "Français": "fr",
}

STT_MODEL_OPTIONS = [
    "gpt-4o-mini-transcribe",
    "gpt-4o-transcribe",
    "whisperx-large-v3-turbo-local",  # STT local pesado (GPU)
    "whisperx-small-int8-local",      # STT local liviano (CPU / GPU modesta)
]

CHAT_MODEL_OPTIONS = [
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-4o-mini",
    "gpt-4o",
]

DEFAULT_WHISPER_PROMPT = (
    "Transcripción de clases y entrevistas de investigación social y educativa. "
    "El contexto incluye pedagogías críticas, educación secundaria en el Área "
    "Metropolitana de Buenos Aires, inteligencia artificial en educación, "
    "colonialidad, y nombres propios como José Manuel, UBA, Freire, Quijano, "
    "Castells, hooks, entre otros. Reconoce estos términos y escríbelos bien."
)

DEFAULT_CLEAN_PROMPT = (
    "Eres un asistente que mejora transcripciones de audio de proyectos de "
    "investigación social.\n\n"
    "Tareas:\n"
    "1. Corrige errores de reconocimiento (especialmente nombres propios y "
    "conceptos teóricos).\n"
    "2. Agrega puntuación y párrafos para que el texto sea legible.\n"
    "3. Mantén el contenido lo más fiel posible al audio original, sin resumir.\n"
    "4. Elimina muletillas obvias ('eh', 'este', 'tipo', repeticiones muy claras), "
    "pero conserva las ideas.\n"
    "5. Mantén el texto en español rioplatense formal-coloquial.\n"
    "6. Si el texto contiene líneas que empiezan con etiquetas de hablante "
    "como 'ENTREVISTADO/A (...):' o 'ENTREVISTADOR/A (...):', respétalas y "
    "no las borres ni las cambies.\n\n"
    "Devuelve SOLO el texto corregido, sin comentarios adicionales."
)


DEFAULT_SUMMARY_PROMPT = (
    "Eres un asistente que ayuda a sistematizar entrevistas y clases en proyectos "
    "de investigación social.\n\n"
    "A partir de la transcripción limpia que recibes, realiza:\n"
    "1. Un resumen denso de 10-15 líneas con las ideas principales.\n"
    "2. Una lista de viñetas con:\n"
    "   - Temas centrales tratados.\n"
    "   - Conceptos teóricos mencionados.\n"
    "   - Posibles citas textuales relevantes (entre comillas), útiles para un "
    "posterior análisis cualitativo.\n\n"
    "No inventes contenido que no esté en la transcripción."
)


class STTApp(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("Jeronimo Abya Yala - Transcriptor y Analizador de Voz a Texto")
        self.geometry("1100x720")

        self._is_running = False

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self._create_top_frame()
        self._create_config_frame()
        self._create_log_frame()

    def _create_top_frame(self):
        frame = ctk.CTkFrame(self)
        frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        frame.grid_columnconfigure(3, weight=1)

        lbl_key = ctk.CTkLabel(frame, text="OpenAI API Key:")
        lbl_key.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.entry_api_key = ctk.CTkEntry(frame, placeholder_text="sk-...", width=320, show="*")
        self.entry_api_key.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        btn_load_env = ctk.CTkButton(
            frame, text="Cargar de .env",
            command=self._load_key_from_env,
            width=120
        )
        btn_load_env.grid(row=0, column=2, padx=5, pady=5, sticky="w")

        self.btn_start = ctk.CTkButton(
            frame, text="Iniciar transcripción / análisis",
            command=self._on_start_clicked,
            fg_color="#1b6f3d", hover_color="#14502d"
        )
        self.btn_start.grid(row=0, column=3, padx=5, pady=5, sticky="e")

    def _create_config_frame(self):
        frame = ctk.CTkFrame(self)
        frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_columnconfigure(3, weight=1)
        frame.grid_rowconfigure(6, weight=1)

        lbl_in = ctk.CTkLabel(frame, text="Carpeta de entrada (audios o textos):")
        lbl_in.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.entry_input = ctk.CTkEntry(frame, placeholder_text="Ej.: ./audios")
        self.entry_input.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        btn_in = ctk.CTkButton(frame, text="Seleccionar", width=100,
                               command=self._select_input_dir)
        btn_in.grid(row=0, column=2, padx=5, pady=5)

        lbl_out = ctk.CTkLabel(frame, text="Carpeta de salida (textos):")
        lbl_out.grid(row=1, column=0, padx=5, pady=5, sticky="w")

        self.entry_output = ctk.CTkEntry(frame, placeholder_text="Ej.: ./transcripciones")
        self.entry_output.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        btn_out = ctk.CTkButton(frame, text="Seleccionar", width=100,
                                command=self._select_output_dir)
        btn_out.grid(row=1, column=2, padx=5, pady=5)

        lbl_tmp = ctk.CTkLabel(frame, text="Carpeta temporal (audio procesado):")
        lbl_tmp.grid(row=2, column=0, padx=5, pady=5, sticky="w")

        self.entry_work = ctk.CTkEntry(frame, placeholder_text="Ej.: ./tmp_procesado")
        self.entry_work.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        lbl_stt = ctk.CTkLabel(frame, text="Modelo STT (audio → texto):")
        lbl_stt.grid(row=3, column=0, padx=5, pady=5, sticky="w")

        self.combo_stt = ctk.CTkComboBox(
            frame,
            values=STT_MODEL_OPTIONS,
            state="normal"
        )
        self.combo_stt.set("gpt-4o-mini-transcribe")
        self.combo_stt.grid(row=3, column=1, padx=5, pady=5, sticky="ew")

        btn_stt_help = ctk.CTkButton(
            frame,
            text="?",
            width=30,
            command=self._show_stt_help,
            fg_color="#444444",
            hover_color="#333333"
        )
        btn_stt_help.grid(row=3, column=2, padx=5, pady=5, sticky="w")

        lbl_chat = ctk.CTkLabel(frame, text="Modelo Chat (limpieza / resumen):")
        lbl_chat.grid(row=4, column=0, padx=5, pady=5, sticky="w")

        self.combo_chat = ctk.CTkComboBox(
            frame,
            values=CHAT_MODEL_OPTIONS,
            state="normal"
        )
        self.combo_chat.set("gpt-4.1-mini")
        self.combo_chat.grid(row=4, column=1, padx=5, pady=5, sticky="ew")

        btn_chat_help = ctk.CTkButton(
            frame,
            text="?",
            width=30,
            command=self._show_chat_help,
            fg_color="#444444",
            hover_color="#333333"
        )
        btn_chat_help.grid(row=4, column=2, padx=5, pady=5, sticky="w")

        lbl_lang = ctk.CTkLabel(frame, text="Idioma principal del audio:")
        lbl_lang.grid(row=3, column=3, padx=5, pady=5, sticky="w")

        self.combo_lang = ctk.CTkComboBox(
            frame,
            values=list(LANG_OPTIONS.keys()),
            state="readonly"
        )
        self.combo_lang.set("Español (Argentina)")
        self.combo_lang.grid(row=4, column=3, padx=5, pady=5, sticky="w")

        self.chk_ffmpeg = ctk.CTkCheckBox(frame, text="Usar reducción de ruido (ffmpeg)",
                                          checkbox_width=18, checkbox_height=18)
        self.chk_ffmpeg.select()
        self.chk_ffmpeg.grid(row=2, column=3, padx=5, pady=5, sticky="w")

 # 👉 NUEVO: opción de diarización local
        self.chk_diar = ctk.CTkCheckBox(
            frame,
            text="Diarización local (WhisperX, GPU)",
            checkbox_width=18,
            checkbox_height=18
        )
        self.chk_diar.deselect()
        self.chk_diar.grid(row=2, column=2, padx=5, pady=5, sticky="w")

        self.chk_clean = ctk.CTkCheckBox(frame, text="Generar versión 'limpia'",
                                         checkbox_width=18, checkbox_height=18)
        self.chk_clean.select()
        self.chk_clean.grid(row=0, column=3, padx=5, pady=5, sticky="w")

        self.chk_summary = ctk.CTkCheckBox(frame, text="Generar resumen analítico",
                                           checkbox_width=18, checkbox_height=18)
        self.chk_summary.select()
        self.chk_summary.grid(row=1, column=3, padx=5, pady=5, sticky="w")

        # Modo de trabajo
        self.mode_var = ctk.IntVar(value=2)  # 1=Solo STT, 2=STT+analisis, 3=Solo análisis
        lbl_mode = ctk.CTkLabel(frame, text="Modo de trabajo:")
        lbl_mode.grid(row=5, column=0, padx=5, pady=5, sticky="w")

        rb1 = ctk.CTkRadioButton(frame, text="Solo transcribir audio (STT)",
                                 variable=self.mode_var, value=1)
        rb1.grid(row=5, column=1, padx=5, pady=5, sticky="w")

        rb2 = ctk.CTkRadioButton(frame, text="Transcribir + analizar (completo)",
                                 variable=self.mode_var, value=2)
        rb2.grid(row=5, column=2, padx=5, pady=5, sticky="w")

        rb3 = ctk.CTkRadioButton(frame, text="Solo analizar textos existentes (.txt)",
                                 variable=self.mode_var, value=3)
        rb3.grid(row=5, column=3, padx=5, pady=5, sticky="w")

        # Tabs para contexto y prompts
        self.tabview = ctk.CTkTabview(frame)
        self.tabview.grid(row=6, column=0, columnspan=4, padx=5, pady=(10, 5), sticky="nsew")

        
        # CREACIÓN DE TABS (una sola vez)
        tab_context = self.tabview.add("Contexto")
        tab_prompts = self.tabview.add("Prompts análisis")


    
        tab_context.grid_columnconfigure(1, weight=1)
        tab_context.grid_rowconfigure(6, weight=1)

        lbl_proj = ctk.CTkLabel(tab_context, text="Proyecto / investigación:")
        lbl_proj.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.entry_project = ctk.CTkEntry(tab_context, placeholder_text="Nombre del proyecto o cátedra")
        self.entry_project.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        lbl_interviewer = ctk.CTkLabel(tab_context, text="Entrevistador/a / Docente:")
        lbl_interviewer.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.entry_interviewer = ctk.CTkEntry(
            tab_context,
            placeholder_text="Ej.: José Manuel, equipo de investigación, etc."
        )
        self.entry_interviewer.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        # 👇 NUEVO: nombre del entrevistado/a
        lbl_interviewee = ctk.CTkLabel(tab_context, text="Entrevistado/a (nombre o código):")
        lbl_interviewee.grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.entry_interviewee = ctk.CTkEntry(
            tab_context,
            placeholder_text="Ej.: Estudiante 3ºA, Directivo, Docente TIC, etc."
        )
        self.entry_interviewee.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        lbl_place = ctk.CTkLabel(tab_context, text="Lugar / institución:")
        lbl_place.grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.entry_place = ctk.CTkEntry(
            tab_context,
            placeholder_text="Ej.: Escuela secundaria privada en CABA, UBA, etc."
        )
        self.entry_place.grid(row=3, column=1, padx=5, pady=5, sticky="ew")

        lbl_date = ctk.CTkLabel(tab_context, text="Fecha / cohorte (opcional):")
        lbl_date.grid(row=4, column=0, padx=5, pady=5, sticky="w")
        self.entry_date = ctk.CTkEntry(tab_context, placeholder_text="Ej.: 2025, cohorte 2024-2025, etc.")
        self.entry_date.grid(row=4, column=1, padx=5, pady=5, sticky="ew")

        lbl_extra = ctk.CTkLabel(tab_context, text="Notas contextuales adicionales:")
        lbl_extra.grid(row=5, column=0, padx=5, pady=5, sticky="nw")
        self.txt_extra = ctk.CTkTextbox(tab_context, height=60)
        self.txt_extra.grid(row=5, column=1, padx=5, pady=5, sticky="nsew")

        lbl_whisper = ctk.CTkLabel(
            tab_context,
            text="Prompt base para la transcripción (se mezclará con el contexto de arriba):"
        )
        lbl_whisper.grid(row=6, column=0, padx=5, pady=(10, 0), sticky="w")

        self.txt_whisper = ctk.CTkTextbox(tab_context, height=80)
        self.txt_whisper.grid(row=6, column=1, padx=5, pady=(10, 5), sticky="nsew")
        self.txt_whisper.insert("1.0", DEFAULT_WHISPER_PROMPT)

        # ----- Tab PROMPTS -----
        tab_prompts.grid_columnconfigure(0, weight=1)
        tab_prompts.grid_rowconfigure(3, weight=1)

        lbl_clean = ctk.CTkLabel(
            tab_prompts,
            text="Prompt para limpiar la transcripción (para análisis cualitativo):"
        )
        lbl_clean.grid(row=0, column=0, padx=5, pady=(10, 0), sticky="w")

        self.txt_clean = ctk.CTkTextbox(tab_prompts, height=120)
        self.txt_clean.grid(row=1, column=0, padx=5, pady=(5, 5), sticky="nsew")
        self.txt_clean.insert("1.0", DEFAULT_CLEAN_PROMPT)

        lbl_summary = ctk.CTkLabel(
            tab_prompts,
            text="Prompt para el resumen (fichas de análisis, temas, citas):"
        )
        lbl_summary.grid(row=2, column=0, padx=5, pady=(10, 0), sticky="w")

        self.txt_summary = ctk.CTkTextbox(tab_prompts, height=120)
        self.txt_summary.grid(row=3, column=0, padx=5, pady=(5, 5), sticky="nsew")
        self.txt_summary.insert("1.0", DEFAULT_SUMMARY_PROMPT)

    def _create_log_frame(self):
        frame = ctk.CTkFrame(self)
        frame.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="nsew")

        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)

        lbl = ctk.CTkLabel(frame, text="Log de proceso (también se guarda en transcribir_stt.log):")
        lbl.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.txt_log = ctk.CTkTextbox(frame, height=140)
        self.txt_log.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.txt_log.configure(state="disabled")

    def _create_log_frame(self):
        frame = ctk.CTkFrame(self)
        frame.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="nsew")

        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(2, weight=1)

        lbl = ctk.CTkLabel(frame, text="Log de proceso (también se guarda en transcribir_stt.log):")
        lbl.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        # Barra de progreso aproximada
        self.progress_bar = ctk.CTkProgressBar(frame)
        self.progress_bar.grid(row=1, column=0, padx=5, pady=(0, 5), sticky="ew")
        self.progress_bar.set(0.0)

        self.txt_log = ctk.CTkTextbox(frame, height=140)
        self.txt_log.grid(row=2, column=0, padx=5, pady=5, sticky="nsew")
        self.txt_log.configure(state="disabled")

    def set_progress_bar(self, value: float):
        """value entre 0.0 y 1.0"""
        try:
            v = max(0.0, min(1.0, float(value)))
        except Exception:
            v = 0.0
        self.progress_bar.set(v)
        self.update_idletasks()

    def append_log(self, msg: str):
        self.txt_log.configure(state="normal")
        self.txt_log.insert("end", msg + "\n")
        self.txt_log.see("end")
        self.txt_log.configure(state="disabled")
        self.update_idletasks()

    def _select_input_dir(self):
            path = fd.askopenfilename(
                title="Seleccionar archivo de audio / video / texto",
                filetypes=[
                    ("Audio / video / texto", "*.wav *.mp3 *.m4a *.mp4 *.mov *.mkv *.txt"),
                    ("Todos los archivos", "*.*"),
                ],
            )
            if path:
                self.entry_input.delete(0, "end")
                self.entry_input.insert(0, path)


    def _select_output_dir(self):
        path = fd.askdirectory(title="Seleccionar carpeta de salida")
        if path:
            self.entry_output.delete(0, "end")
            self.entry_output.insert(0, path)

    def _load_key_from_env(self):
        key = load_api_key_from_env()
        if key:
            self.entry_api_key.delete(0, "end")
            self.entry_api_key.insert(0, key)
            mb.showinfo("API Key", "API key cargada desde .env / entorno.")
        else:
            mb.showwarning("API Key", "No se encontró OPENAI_API_KEY en .env ni en el entorno.")

    def _show_stt_help(self):
        win = ctk.CTkToplevel(self)
        win.title("Ayuda – Modelos STT (audio → texto)")
        win.geometry("620x420")
        win.grab_set()

        text = (
                "Modelos STT (speech-to-text):\n\n"
                "- gpt-4o-mini-transcribe (recomendado online):\n"
                "  • Más barato y rápido usando la API de OpenAI.\n"
                "  • Ideal para la mayoría de clases y entrevistas.\n\n"
                "- gpt-4o-transcribe:\n"
                "  • Más preciso con audio ruidoso o acentos complicados.\n"
                "  • Un poco más caro.\n\n"
                "- whisperx-large-v3-turbo-local:\n"
                "  • Transcripción local con WhisperX + faster-whisper (modelo large-v3-turbo).\n"
                "  • Recomendado si tenés una GPU potente (12–16 GB de VRAM o más).\n\n"
                "- whisperx-small-int8-local:\n"
                "  • Transcripción local liviana (modelo pequeño cuantizado a int8).\n"
                "  • Pensado para PCs sin GPU dedicada o con GPU modesta.\n\n"
                "Para usar los modelos locales necesitás tener instalado 'whisperx' y sus\n"
                "dependencias, y si activás diarización, un HUGGINGFACE_TOKEN configurado en .env.\n"
                "También podés escribir a mano otro nombre de modelo de transcripción disponible "
                "en tu cuenta de OpenAI."
        )

        txt = ctk.CTkTextbox(win, wrap="word")
        txt.insert("1.0", text)
        txt.configure(state="disabled")  # solo lectura
        txt.pack(padx=15, pady=15, fill="both", expand=True)

        btn_close = ctk.CTkButton(win, text="Cerrar", command=win.destroy)
        btn_close.pack(padx=10, pady=(0, 10))




    def _show_chat_help(self):
        win = ctk.CTkToplevel(self)
        win.title("Ayuda – Modelos Chat (limpieza / resumen)")
        win.geometry("620x420")
        win.grab_set()

        text = (
            "Modelos de chat (para limpiar texto y hacer resúmenes):\n\n"
            "- gpt-4.1-mini (recomendado):\n"
            "  • Rápido y barato.\n"
            "  • Ideal para limpiar transcripciones y hacer resúmenes básicos.\n\n"
            "- gpt-4.1 / gpt-4o:\n"
            "  • Más potentes y caros.\n"
            "  • Útiles para análisis más fino o textos largos/complicados.\n\n"
            "- gpt-4o-mini:\n"
            "  • Buen equilibrio costo/calidad para tareas ligeras.\n\n"
            "También podés escribir a mano cualquier otro modelo de chat "
            "disponible en tu cuenta."
        )

        txt = ctk.CTkTextbox(win, wrap="word")
        txt.insert("1.0", text)
        txt.configure(state="disabled")  # solo lectura
        txt.pack(padx=15, pady=15, fill="both", expand=True)

        btn_close = ctk.CTkButton(win, text="Cerrar", command=win.destroy)
        btn_close.pack(padx=10, pady=(0, 10))

    def _on_start_clicked(self):
        # Evitar doble ejecución
        if self._is_running:
            mb.showwarning("En ejecución", "Ya hay un proceso en marcha.")
            return

        # --- API KEY ---
        api_key = self.entry_api_key.get().strip()
        if not api_key:
            mb.showerror("Falta API Key", "Por favor ingresa tu OpenAI API Key.")
            return

        # --- RUTAS ---
        input_path_str = self.entry_input.get().strip()
        output_dir_str = self.entry_output.get().strip() or "./transcripciones"
        work_dir_str = self.entry_work.get().strip() or "./tmp_procesado"

        if not input_path_str:
            mb.showerror("Error", "Debes seleccionar un archivo o carpeta de entrada.")
            return

        input_path = Path(input_path_str)

        if input_path.is_file():
            # Modo archivo único
            input_dir = input_path.parent
            selected_files: list[Path] | None = [input_path]
        elif input_path.is_dir():
            # Modo carpeta (lote)
            input_dir = input_path
            selected_files = None
        else:
            mb.showerror("Error", "La ruta de entrada no existe o no es válida.")
            return

        # --- MODELOS E IDIOMA ---
        stt_model = self.combo_stt.get().strip() or "gpt-4o-mini-transcribe"
        chat_model = self.combo_chat.get().strip() or "gpt-4.1-mini"

        # Mapear ID interno de modelo STT → backend real de WhisperX (para modelos locales)
        whisperx_model = "large-v2"  # default que ya usabas para diarización

        if stt_model == "whisperx-large-v3-turbo-local":
            # Usa el modelo turbo grande convertido a CTranslate2.
            # Cambiá esta string por el repo concreto que estés usando,
            # por ejemplo:
            #   "openai/whisper-large-v3-turbo"
            #   "faster-whisper-large-v3-turbo"
            #   "openwhisper-turbo-large-v3-ct2"
            whisperx_model = "openai/whisper-large-v3-turbo"

        elif stt_model == "whisperx-small-int8-local":
            # Modelo pequeño; WhisperX sabe resolver "small" con faster-whisper.
            whisperx_model = "small"


        lang_label = self.combo_lang.get()
        language = LANG_OPTIONS.get(lang_label, "es")

        use_ffmpeg = bool(self.chk_ffmpeg.get())
        do_clean = bool(self.chk_clean.get())
        do_summary = bool(self.chk_summary.get())
        use_diarization = bool(self.chk_diar.get())

        # --- PROMPT WHISPER + CONTEXTO ---
        base_prompt = self.txt_whisper.get("1.0", "end").strip()
        meta_parts: list[str] = []

        project = self.entry_project.get().strip()
        interviewer = self.entry_interviewer.get().strip()
        interviewee = self.entry_interviewee.get().strip()
        place = self.entry_place.get().strip()
        date = self.entry_date.get().strip()
        extra = self.txt_extra.get("1.0", "end").strip()

        if project:
            meta_parts.append(f"Proyecto de investigación: {project}.")
        if interviewer:
            meta_parts.append(f"Entrevistador/a o docente: {interviewer}.")
        if interviewee:
            meta_parts.append(f"Persona entrevistada (nombre o código): {interviewee}.")
        if place:
            meta_parts.append(f"Lugar / institución: {place}.")
        if date:
            meta_parts.append(f"Fecha o cohorte: {date}.")
        if extra:
            meta_parts.append(f"Notas contextuales: {extra}")

        meta_context = " ".join(meta_parts)

        if base_prompt and meta_context:
            whisper_prompt = (
                base_prompt
                + "\n\nContexto específico de esta grabación:\n"
                + meta_context
            )
        elif base_prompt:
            whisper_prompt = base_prompt
        else:
            whisper_prompt = meta_context

        # --- PROMPTS DE LIMPIEZA Y RESUMEN ---
        base_clean_prompt = self.txt_clean.get("1.0", "end").strip()
        base_summary_prompt = self.txt_summary.get("1.0", "end").strip()

        # Versión simple: no intentamos etiquetar quién habla,
        # solo usamos los prompts tal como están escritos en la GUI.
        clean_prompt = base_clean_prompt
        summary_prompt = base_summary_prompt

        # --- CONFIG GLOBAL ---
        cfg = TranscriberConfig(
            api_key=api_key,
            input_dir=input_dir,
            output_dir=Path(output_dir_str),
            work_dir=Path(work_dir_str),
            selected_files=selected_files,
            stt_model=stt_model,
            chat_model=chat_model,
            language=language,
            use_ffmpeg=use_ffmpeg,
            whisper_prompt=whisper_prompt,
            clean_prompt=clean_prompt,
            summary_prompt=summary_prompt,
            do_clean=do_clean,
            do_summary=do_summary,
            use_diarization=use_diarization,
            whisperx_model=whisperx_model,
            interviewer_name=interviewer,
            interviewee_name=interviewee,
        )


        # --- LIMPIAR LOG Y PREPARAR UI ---
        self.txt_log.configure(state="normal")
        self.txt_log.delete("1.0", "end")
        self.txt_log.configure(state="disabled")
        self.progress_bar.set(0.0)

        self._is_running = True
        self.btn_start.configure(state="disabled", text="Procesando...")

        # --- SELECCIÓN DE MODO ---
        mode = self.mode_var.get()

        if mode == 1:
            # Solo STT: no limpiamos ni resumimos
            cfg.do_clean = False
            cfg.do_summary = False
            target = lambda: self._run_thread_wrapper(run_batch, cfg)
        elif mode == 2:
            # STT + análisis (según checkboxes)
            target = lambda: self._run_thread_wrapper(run_batch, cfg)
        elif mode == 3:
            # Solo análisis de textos existentes
            if not cfg.do_clean and not cfg.do_summary:
                mb.showwarning(
                    "Configuración",
                    "En modo 'Solo analizar textos' debes activar al menos 'Generar versión limpia' o 'Generar resumen analítico'.",
                )
                self._reset_ui()
                return
            target = lambda: self._run_thread_wrapper(run_analysis_only, cfg)
        else:
            mb.showerror("Modo desconocido", f"Modo de trabajo inválido: {mode}")
            self._reset_ui()
            return

        # --- LANZAR HILO ---
        thread = threading.Thread(target=target, daemon=True)
        thread.start()

    def _run_thread_wrapper(self, func, cfg: TranscriberConfig):
        def progress(msg: str):
            self.after(0, self.append_log, msg)

        def set_progress(p: float):
            self.after(0, self.set_progress_bar, p)

        try:
            func(cfg, progress, set_progress)
        except Exception as e:
            self.after(0, lambda: mb.showerror("Error", str(e)))
            self.after(0, self.append_log, f"ERROR general: {e}")
        finally:
            self.after(0, self._reset_ui)

    def _reset_ui(self):
        self._is_running = False
        self.btn_start.configure(state="normal", text="Iniciar transcripción / análisis")


if __name__ == "__main__":
    app = STTApp()
    app.mainloop()
