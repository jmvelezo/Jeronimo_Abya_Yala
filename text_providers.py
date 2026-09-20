"""Proveedores de texto para Jerónimo Abya Yala.

FASE 4: desacopla limpieza/resumen del motor ASR.

Objetivos:
- conservar el comportamiento de Ollama ya validado (fragmentación + consolidación de resúmenes),
- conservar el comportamiento directo de OpenAI existente,
- añadir APIs compatibles con OpenAI sin acoplar Jerónimo a un proveedor concreto,
- describir correctamente si un endpoint es local o remoto,
- permitir una prueba de conexión que no envía contenido de entrevistas.
"""
from __future__ import annotations

from dataclasses import dataclass
from ipaddress import ip_address
import json
import re
from typing import Callable, Optional, Protocol
from urllib.parse import urlparse
import urllib.error
import urllib.request

from local_text_runtime import PRIVATE_OLLAMA_URL, is_private_ollama_url, start_private_ollama_if_installed

try:
    from openai import OpenAI  # type: ignore
    HAS_OPENAI = True
except Exception:
    OpenAI = None  # type: ignore
    HAS_OPENAI = False

TEXT_ENGINE_NONE = "none"
TEXT_ENGINE_OPENAI = "openai"
TEXT_ENGINE_OLLAMA = "ollama"
TEXT_ENGINE_OPENAI_COMPATIBLE = "openai_compatible"
DEFAULT_OLLAMA_URL = PRIVATE_OLLAMA_URL
DEFAULT_OPENAI_COMPATIBLE_URL = "http://localhost:1234/v1"
DEFAULT_CHUNK_CHARS = 12000

ProgressFn = Callable[[str], None]


@dataclass(frozen=True)
class TextProviderSettings:
    engine: str
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    ollama_url: str = DEFAULT_OLLAMA_URL
    ollama_model: str = ""
    chunk_chars: int = DEFAULT_CHUNK_CHARS


@dataclass(frozen=True)
class TextEndpointInfo:
    engine: str
    label: str
    scope: str  # local | remote | disabled
    endpoint: str
    model: str
    sends_text_off_device: bool


class TextProvider(Protocol):
    engine: str
    model: str

    def generate(self, messages: list[dict[str, str]], temperature: float = 0.0, timeout: int = 180) -> str:
        ...


class OllamaProvider:
    engine = TEXT_ENGINE_OLLAMA

    def __init__(self, base_url: str, model: str):
        self.base_url = (base_url or DEFAULT_OLLAMA_URL).rstrip("/")
        self.model = str(model or "").strip()
        if not self.model:
            raise RuntimeError("Ollama está seleccionado, pero no hay modelo elegido.")
        if is_private_ollama_url(self.base_url):
            try:
                start_private_ollama_if_installed(wait=True)
            except Exception:
                # Si todavía no está instalado, la validación superior explicará
                # que debe ejecutarse la configuración automática/gestor.
                pass

    @property
    def num_ctx(self) -> int:
        """Contexto conservador para sostener calidad sin agotar VRAM/RAM."""
        model = self.model.lower()
        if "30b" in model:
            return 32768
        if "14b" in model or "12b" in model:
            return 24576
        if "8b" in model:
            return 16384
        if "4b" in model:
            return 12288
        return 8192

    @property
    def supports_reasoning(self) -> bool:
        return self.model.lower().startswith("qwen3")

    def _request(self, payload: dict, timeout: int) -> dict:
        url = f"{self.base_url}/api/chat"
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=raw, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:600]
            except Exception:
                pass
            raise RuntimeError(f"Ollama devolvió HTTP {exc.code}. {detail}".strip()) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"El motor local de texto no responde en {self.base_url}. "
                "Jerónimo intentará iniciarlo automáticamente si es su runtime privado."
            ) from exc
        except TimeoutError as exc:
            raise RuntimeError(f"Ollama excedió el tiempo de espera en {self.base_url}.") from exc

    def generate_advanced(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        timeout: int = 600,
        num_ctx: int | None = None,
        num_predict: int | None = None,
        think: bool | str | None = None,
        response_format: dict | str | None = None,
        keep_alive: str | int = "15m",
    ) -> str:
        options: dict[str, object] = {
            "temperature": temperature,
            "num_ctx": int(num_ctx or self.num_ctx),
        }
        if num_predict is not None:
            options["num_predict"] = int(num_predict)
        payload: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": keep_alive,
            "options": options,
        }
        if think is not None:
            payload["think"] = think
        if response_format is not None:
            payload["format"] = response_format

        try:
            data = self._request(payload, timeout)
        except RuntimeError:
            # Compatibilidad con runtimes/modelos que todavía no acepten `think`
            # o JSON schema: reintenta sin esas extensiones, manteniendo num_ctx.
            if "think" not in payload and "format" not in payload:
                raise
            payload.pop("think", None)
            payload.pop("format", None)
            data = self._request(payload, timeout)

        if data.get("error"):
            raise RuntimeError(f"Ollama devolvió un error: {data['error']}")
        text = str(data.get("message", {}).get("content", "") or "").strip()
        if not text:
            raise RuntimeError("Ollama respondió sin contenido de texto.")
        return text

    def generate(self, messages: list[dict[str, str]], temperature: float = 0.0, timeout: int = 180) -> str:
        return self.generate_advanced(
            messages,
            temperature=temperature,
            timeout=max(timeout, 300),
            num_ctx=self.num_ctx,
            keep_alive="10m",
        )


class OpenAIProvider:
    engine = TEXT_ENGINE_OPENAI

    def __init__(self, api_key: str, model: str):
        if not HAS_OPENAI or OpenAI is None:
            raise RuntimeError("Falta el paquete 'openai'. Ejecutá el instalador de Jerónimo Abya Yala.")
        self.api_key = str(api_key or "").strip()
        self.model = str(model or "").strip()
        if not self.api_key:
            raise RuntimeError("OpenAI está seleccionado, pero falta la API key.")
        if not self.model:
            raise RuntimeError("OpenAI está seleccionado, pero falta el modelo de texto.")
        self.client = OpenAI(api_key=self.api_key)

    def generate(self, messages: list[dict[str, str]], temperature: float = 0.0, timeout: int = 180) -> str:
        del timeout  # El SDK maneja sus propios timeouts/reintentos.
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        content = response.choices[0].message.content
        text = str(content or "").strip()
        if not text:
            raise RuntimeError("OpenAI respondió sin contenido de texto.")
        return text


class OpenAICompatibleProvider:
    engine = TEXT_ENGINE_OPENAI_COMPATIBLE

    def __init__(self, base_url: str, api_key: str, model: str):
        if not HAS_OPENAI or OpenAI is None:
            raise RuntimeError("Falta el paquete 'openai'. Ejecutá el instalador de Jerónimo Abya Yala.")
        self.base_url = str(base_url or "").strip().rstrip("/")
        self.api_key = str(api_key or "").strip()
        self.model = str(model or "").strip()
        if not self.base_url:
            raise RuntimeError("La API compatible requiere una URL base, por ejemplo http://localhost:1234/v1.")
        if not self.model:
            raise RuntimeError("La API compatible requiere indicar el nombre del modelo.")
        # Muchos servidores locales compatibles no verifican la clave pero el SDK exige una.
        effective_key = self.api_key or "jeronimo-local-no-key"
        self.client = OpenAI(api_key=effective_key, base_url=self.base_url)

    def generate(self, messages: list[dict[str, str]], temperature: float = 0.0, timeout: int = 180) -> str:
        del timeout
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        content = response.choices[0].message.content
        text = str(content or "").strip()
        if not text:
            raise RuntimeError("La API compatible respondió sin contenido de texto.")
        return text


def normalize_engine(value: str | None) -> str:
    engine = str(value or TEXT_ENGINE_OPENAI).strip().lower()
    allowed = {
        TEXT_ENGINE_NONE,
        TEXT_ENGINE_OPENAI,
        TEXT_ENGINE_OLLAMA,
        TEXT_ENGINE_OPENAI_COMPATIBLE,
    }
    return engine if engine in allowed else TEXT_ENGINE_OPENAI


def _host_is_local(host: str) -> bool:
    host = (host or "").strip().lower().strip("[]")
    if not host:
        return False
    if host in {"localhost", "localhost.localdomain"}:
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        pass
    # Criterio conservador de privacidad: nombres de host distintos de localhost
    # se consideran remotos aunque puedan resolver dentro de la LAN.
    return False


def endpoint_scope(url: str) -> str:
    try:
        parsed = urlparse(str(url or "").strip())
        return "local" if _host_is_local(parsed.hostname or "") else "remote"
    except Exception:
        return "remote"


def describe_endpoint(settings: TextProviderSettings) -> TextEndpointInfo:
    engine = normalize_engine(settings.engine)
    if engine == TEXT_ENGINE_NONE:
        return TextEndpointInfo(engine, "Desactivado", "disabled", "", "", False)
    if engine == TEXT_ENGINE_OLLAMA:
        endpoint = settings.ollama_url or DEFAULT_OLLAMA_URL
        scope = endpoint_scope(endpoint)
        return TextEndpointInfo(
            engine,
            "Ollama local" if scope == "local" else "Ollama en otro equipo/servidor",
            scope,
            endpoint,
            settings.ollama_model,
            scope != "local",
        )
    if engine == TEXT_ENGINE_OPENAI_COMPATIBLE:
        endpoint = settings.base_url
        scope = endpoint_scope(endpoint)
        return TextEndpointInfo(
            engine,
            "API compatible local" if scope == "local" else "API compatible externa",
            scope,
            endpoint,
            settings.model,
            scope != "local",
        )
    return TextEndpointInfo(
        engine,
        "OpenAI API",
        "remote",
        "https://api.openai.com/v1",
        settings.model,
        True,
    )


def create_provider(settings: TextProviderSettings) -> Optional[TextProvider]:
    engine = normalize_engine(settings.engine)
    if engine == TEXT_ENGINE_NONE:
        return None
    if engine == TEXT_ENGINE_OLLAMA:
        return OllamaProvider(settings.ollama_url, settings.ollama_model)
    if engine == TEXT_ENGINE_OPENAI_COMPATIBLE:
        return OpenAICompatibleProvider(settings.base_url, settings.api_key, settings.model)
    return OpenAIProvider(settings.api_key, settings.model)


def get_ollama_models(base_url: str = DEFAULT_OLLAMA_URL) -> list[str]:
    try:
        url = f"{(base_url or DEFAULT_OLLAMA_URL).rstrip('/')}/api/tags"
        req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        return [
            str(m.get("name", "") or m.get("model", "")).strip()
            for m in data.get("models", [])
            if str(m.get("name", "") or m.get("model", "")).strip()
        ]
    except Exception:
        return []


def split_text_for_llm(text: str, max_chars: int = DEFAULT_CHUNK_CHARS) -> list[str]:
    """Divide por párrafos para evitar cortar turnos cuando sea posible."""
    text = text or ""
    if len(text) <= max_chars:
        return [text]
    paras = re.split(r"(\n\s*\n)", text)
    chunks: list[str] = []
    current = ""
    for part in paras:
        if len(current) + len(part) <= max_chars:
            current += part
            continue
        if current.strip():
            chunks.append(current.strip())
        if len(part) > max_chars:
            for i in range(0, len(part), max_chars):
                piece = part[i:i + max_chars].strip()
                if piece:
                    chunks.append(piece)
            current = ""
        else:
            current = part
    if current.strip():
        chunks.append(current.strip())
    return chunks or [text]


_EVIDENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "ideas_principales": {"type": "array", "items": {"type": "string"}},
        "temas": {"type": "array", "items": {"type": "string"}},
        "posiciones_argumentos": {"type": "array", "items": {"type": "string"}},
        "tensiones_contradicciones": {"type": "array", "items": {"type": "string"}},
        "conceptos_instituciones": {"type": "array", "items": {"type": "string"}},
        "citas_relevantes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "hablante": {"type": "string"},
                    "timecode": {"type": "string"},
                    "texto": {"type": "string"},
                },
                "required": ["texto"],
            },
        },
        "dudas_revision": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["ideas_principales", "temas", "posiciones_argumentos", "tensiones_contradicciones", "conceptos_instituciones", "citas_relevantes", "dudas_revision"],
}


def _ollama_generate(
    provider: TextProvider,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.0,
    timeout: int = 600,
    think: bool | str | None = None,
    response_format: dict | str | None = None,
    num_predict: int | None = None,
    keep_alive: str | int = "15m",
) -> str:
    advanced = getattr(provider, "generate_advanced", None)
    if callable(advanced):
        return advanced(
            messages,
            temperature=temperature,
            timeout=timeout,
            think=think,
            response_format=response_format,
            num_predict=num_predict,
            keep_alive=keep_alive,
        )
    return provider.generate(messages, temperature=temperature, timeout=min(timeout, 180))


def _normalize_json_evidence(raw: str, source_text: str = "") -> str:
    text = str(raw or "").strip()
    if not text:
        return "{}"
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and source_text:
            # Las citas son el punto más sensible del resumen cualitativo. Sólo
            # conservamos citas cuyo texto aparece realmente en el fragmento de
            # transcripción (normalizando únicamente espacios/saltos de línea).
            source_flat = re.sub(r"\s+", " ", source_text).strip()
            quotes = obj.get("citas_relevantes")
            if isinstance(quotes, list):
                valid_quotes = []
                for quote in quotes:
                    if not isinstance(quote, dict):
                        continue
                    quote_text = str(quote.get("texto", "") or "").strip()
                    quote_flat = re.sub(r"\s+", " ", quote_text).strip()
                    if quote_flat and quote_flat in source_flat:
                        valid_quotes.append(quote)
                obj["citas_relevantes"] = valid_quotes
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        # Si el modelo no respetó JSON, conservamos toda la evidencia en vez de
        # descartarla. La fase de integración recibe este texto como fuente.
        return json.dumps({"evidencia_no_estructurada": text}, ensure_ascii=False, indent=2)


def _deep_ollama_summary(
    provider: TextProvider,
    transcript: str,
    system_prompt: str,
    progress: ProgressFn,
    *,
    chunk_chars: int,
) -> str:
    """Análisis jerárquico local de alta fidelidad para entrevistas.

    1) extrae evidencia estructurada por fragmento;
    2) integra temas/tensiones entre fragmentos;
    3) redacta una síntesis analítica global;
    4) verifica la síntesis contra la evidencia antes de devolverla.
    """
    chunks = split_text_for_llm(transcript, max(6000, chunk_chars))
    endpoint = describe_endpoint(TextProviderSettings(
        engine=TEXT_ENGINE_OLLAMA,
        ollama_url=getattr(provider, "base_url", DEFAULT_OLLAMA_URL),
        ollama_model=getattr(provider, "model", ""),
    ))
    locality = "local" if endpoint.scope == "local" else "remoto"
    progress(
        f"Iniciando resumen analítico profundo con Ollama {locality} "
        f"({getattr(provider, 'model', '')}) en {len(chunks)} bloque(s)..."
    )

    evidence: list[str] = []
    extraction_system = (
        "Actúas como asistente de investigación cualitativa. Extrae evidencia de una parte de una entrevista. "
        "No agregues contexto externo, no completes silencios y no conviertas inferencias en hechos. "
        "Conserva etiquetas de hablantes y timecodes cuando estén presentes. Las citas deben ser textuales; "
        "si no puedes garantizar literalidad, no las incluyas. Registra tensiones, cambios de posición y dudas."
    )
    for i, chunk in enumerate(chunks, start=1):
        progress(f"Resumen profundo: extrayendo evidencias {i}/{len(chunks)}...")
        raw = _ollama_generate(
            provider,
            [
                {"role": "system", "content": extraction_system},
                {"role": "user", "content": chunk},
            ],
            temperature=0.0,
            timeout=900,
            think=False,
            response_format=_EVIDENCE_SCHEMA,
            num_predict=3072,
        )
        evidence.append(f"BLOQUE {i}\n" + _normalize_json_evidence(raw, chunk))

    joined_evidence = "\n\n--- BLOQUE DE EVIDENCIA ---\n\n".join(evidence)
    # Modelos menores trabajan mejor si no reciben toda la evidencia de una vez.
    model_ctx = int(getattr(provider, "num_ctx", 8192) or 8192)
    integration_chars = max(18000, min(70000, model_ctx * 3))
    evidence_batches = split_text_for_llm(joined_evidence, integration_chars)
    integrations: list[str] = []
    integration_system = (
        "Integra evidencia estructurada de una entrevista cualitativa. Busca recurrencias, relaciones entre temas, "
        "tensiones, contradicciones, cambios de posición y diferencias entre hablantes. No uses conocimiento externo. "
        "Distingue con claridad lo dicho de cualquier inferencia y conserva referencias a hablantes/timecodes disponibles. "
        "Si reproduces una cita, copia su texto exactamente como aparece en la evidencia: no la parafrasees ni la corrijas. "
        "No redactes todavía una conclusión definitiva: produce un mapa analítico denso que sirva de base a la síntesis."
    )
    for i, batch in enumerate(evidence_batches, start=1):
        progress(f"Ollama resumen: consolidando evidencias {i}/{len(evidence_batches)}...")
        integrations.append(_ollama_generate(
            provider,
            [
                {"role": "system", "content": integration_system},
                {"role": "user", "content": batch},
            ],
            temperature=0.0,
            timeout=1200,
            think=True if getattr(provider, "supports_reasoning", False) else None,
            num_predict=4096,
        ))

    integrated = "\n\n--- MAPA ANALÍTICO ---\n\n".join(integrations)
    progress("Resumen profundo: redactando síntesis global...")
    final_system = (
        (system_prompt or "A partir de la transcripción, elabora un resumen analítico fiel.").strip()
        + "\n\nTrabaja únicamente con el mapa analítico y las evidencias extraídas de la transcripción. "
          "Prioriza fidelidad y densidad analítica sobre brevedad. Explicita recurrencias, matices, tensiones, "
          "contradicciones y cambios de posición cuando la evidencia los sostenga. No agregues contexto externo. "
          "Toda cita debe provenir literalmente de la evidencia y conservar hablante/timecode si existe. "
          "Separa con claridad: resumen analítico, temas centrales, tensiones/contradicciones, conceptos o instituciones, "
          "citas relevantes y dudas para revisión humana."
    )
    draft = _ollama_generate(
        provider,
        [
            {"role": "system", "content": final_system},
            {"role": "user", "content": integrated},
        ],
        temperature=0.0,
        timeout=1500,
        think=True if getattr(provider, "supports_reasoning", False) else None,
        num_predict=6144,
    )

    progress("Resumen profundo: verificando fidelidad contra la evidencia...")
    verification_system = (
        "Revisa un borrador de resumen analítico frente a la evidencia de la entrevista. Corrige o elimina cualquier "
        "afirmación, relación o cita que no esté sostenida por la evidencia. Conserva contradicciones y dudas en vez de "
        "resolverlas artificialmente. No agregues conocimiento externo. Devuelve únicamente la versión final corregida, "
        "completa y legible del resumen."
    )
    verification_source = (
        "EVIDENCIA / MAPA ANALÍTICO:\n" + integrated
        + "\n\nBORRADOR A VERIFICAR:\n" + draft
    )
    # Si la combinación fuera demasiado grande para un modelo pequeño, el mapa
    # integrado sigue siendo la fuente prioritaria y se trunca sólo el borrador.
    max_verify_chars = max(26000, min(90000, model_ctx * 3))
    if len(verification_source) > max_verify_chars:
        verification_source = (
            "EVIDENCIA / MAPA ANALÍTICO:\n" + integrated[: int(max_verify_chars * 0.72)]
            + "\n\nBORRADOR A VERIFICAR:\n" + draft[: int(max_verify_chars * 0.28)]
        )
    final = _ollama_generate(
        provider,
        [
            {"role": "system", "content": verification_system},
            {"role": "user", "content": verification_source},
        ],
        temperature=0.0,
        timeout=1500,
        think=True if getattr(provider, "supports_reasoning", False) else None,
        num_predict=6144,
        keep_alive=0,
    )
    progress("Etapa de resumen analítico profundo con Ollama finalizada.")
    return final


def process_text(
    provider: TextProvider,
    transcript: str,
    system_prompt: str,
    progress: ProgressFn,
    etapa: str,
    *,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    chunk_local_ollama: bool = True,
) -> str:
    """Procesa texto sin cambiar el pipeline ASR.

    Para resúmenes locales usa un análisis jerárquico profundo con extracción de
    evidencia, integración global y verificación final. La limpieza conserva la
    estrategia histórica por fragmentos para no reinterpretar el testimonio.
    """
    if getattr(provider, "engine", "") == TEXT_ENGINE_OLLAMA and chunk_local_ollama:
        if etapa.lower().startswith("resumen"):
            return _deep_ollama_summary(
                provider,
                transcript,
                system_prompt,
                progress,
                chunk_chars=chunk_chars,
            )

        chunks = split_text_for_llm(transcript, chunk_chars)
        endpoint = describe_endpoint(TextProviderSettings(
            engine=TEXT_ENGINE_OLLAMA,
            ollama_url=provider.base_url,
            ollama_model=provider.model,
        ))
        locality = "local" if endpoint.scope == "local" else "remoto"
        if len(chunks) == 1:
            progress(f"Iniciando etapa de {etapa} con Ollama {locality} ({provider.model})...")
            out = _ollama_generate(provider, [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcript},
            ], temperature=0.0, timeout=600, think=False, num_predict=4096)
            progress(f"Etapa de {etapa} con Ollama finalizada.")
            return out

        progress(f"Iniciando etapa de {etapa} con Ollama {locality} ({provider.model}) en {len(chunks)} partes...")
        partials: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            progress(f"Ollama {etapa}: parte {i}/{len(chunks)}...")
            chunk_prompt = (
                system_prompt
                + "\n\nProcesa solo esta parte. Mantén etiquetas de hablantes, timecodes y orden original. "
                + "No inventes contenido ni cierres con comentarios externos."
            )
            partials.append(_ollama_generate(provider, [
                {"role": "system", "content": chunk_prompt},
                {"role": "user", "content": chunk},
            ], temperature=0.0, timeout=600, think=False, num_predict=4096))
        progress(f"Etapa de {etapa} con Ollama finalizada.")
        return "\n\n".join(p.strip() for p in partials if p.strip()).strip()

    label = "OpenAI" if provider.engine == TEXT_ENGINE_OPENAI else "API compatible"
    progress(f"Iniciando etapa de {etapa} con {label} ({provider.model})...")
    out = provider.generate([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": transcript},
    ], temperature=0.0)
    progress(f"Etapa de {etapa} con {label} finalizada.")
    return out


def test_provider(settings: TextProviderSettings) -> dict[str, object]:
    """Prueba el proveedor sin enviar contenido de entrevistas."""
    info = describe_endpoint(settings)
    provider = create_provider(settings)
    if provider is None:
        return {"ok": True, "message": "Procesamiento de texto desactivado.", "endpoint": info}
    test_messages = [
        {"role": "system", "content": "Prueba técnica de conexión. Responde únicamente con OK."},
        {"role": "user", "content": "OK"},
    ]
    advanced = getattr(provider, "generate_advanced", None)
    if callable(advanced) and getattr(provider, "engine", "") == TEXT_ENGINE_OLLAMA:
        result = advanced(test_messages, temperature=0.0, timeout=60, keep_alive=0)
    else:
        result = provider.generate(test_messages, temperature=0.0, timeout=60)
    return {
        "ok": bool(result.strip()),
        "message": result.strip(),
        "endpoint": info,
    }
