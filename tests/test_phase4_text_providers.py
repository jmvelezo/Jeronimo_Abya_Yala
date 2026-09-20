from __future__ import annotations

import hashlib
import inspect
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import core_transcriber as core
import text_providers as tp
from credential_store import environment_secret


ASR_BASELINE_HASHES = {
    "transcribe_audio_local_faster_whisper": "c30b4fa79ca7baac2cbdfbdbed22857ab7fe23497a82af7b2a5860ff5d1db1d1",
    "transcribe_audio": "381f1b26bac90fb8eda53138a570b34edaf5ec41ba2b9b3257aa22f77dbf583c",
    "diarize_and_transcribe_local": "91a86a03fcbef536e5a1085bdfae566118ffaaa876b55b426e7f6944d353bbcf",
}


class _FakeHTTPResponse:
    def __init__(self, payload: dict):
        self.body = json.dumps(payload).encode("utf-8")
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False
    def read(self, *args):
        return self.body


class _FakeProvider:
    engine = tp.TEXT_ENGINE_OLLAMA
    model = "fake"
    base_url = "http://localhost:11434"
    def __init__(self):
        self.calls = []
    def generate(self, messages, temperature=0.0, timeout=180):
        self.calls.append(messages)
        return f"salida-{len(self.calls)}"


class Phase4PrivacyTests(unittest.TestCase):
    def test_localhost_endpoints_are_local(self):
        self.assertEqual(tp.endpoint_scope("http://localhost:11434"), "local")
        self.assertEqual(tp.endpoint_scope("http://127.0.0.1:1234/v1"), "local")
        self.assertEqual(tp.endpoint_scope("http://[::1]:1234/v1"), "local")

    def test_non_loopback_endpoint_is_remote(self):
        self.assertEqual(tp.endpoint_scope("http://192.168.1.5:11434"), "remote")
        self.assertEqual(tp.endpoint_scope("https://example.invalid/v1"), "remote")

    def test_manifest_mode_marks_remote_ollama_hybrid(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            cfg = core.TranscriberConfig(
                api_key="", input_dir=p, output_dir=p, work_dir=p,
                stt_engine=core.STT_ENGINE_LOCAL_FAST_WHISPER,
                text_engine=core.TEXT_ENGINE_OLLAMA,
                ollama_url="http://192.168.1.10:11434",
                ollama_model="qwen3:4b",
                do_summary=True,
                summary_prompt="resume",
            )
            self.assertEqual(core._manifest_privacy_mode(cfg), "hybrid")

    def test_manifest_mode_marks_local_compatible_api_local(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            cfg = core.TranscriberConfig(
                api_key="", input_dir=p, output_dir=p, work_dir=p,
                stt_engine=core.STT_ENGINE_LOCAL_FAST_WHISPER,
                text_engine=core.TEXT_ENGINE_OPENAI_COMPATIBLE,
                compatible_api_url="http://127.0.0.1:1234/v1",
                chat_model="local-model",
                do_summary=True,
                summary_prompt="resume",
            )
            self.assertEqual(core._manifest_privacy_mode(cfg), "local")



    def test_manifest_never_writes_compatible_api_key(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            audio = root / "entrevista.wav"
            audio.write_bytes(b"RIFF")
            cfg = core.TranscriberConfig(
                api_key="", input_dir=root, output_dir=root, work_dir=root,
                stt_engine=core.STT_ENGINE_LOCAL_FAST_WHISPER,
                text_engine=core.TEXT_ENGINE_OPENAI_COMPATIBLE,
                compatible_api_url="https://example.invalid/v1",
                compatible_api_key="super-secret-compatible",
                chat_model="modelo-x", do_summary=True, summary_prompt="resume",
            )
            path = core.write_manifest(
                cfg, input_file=audio, output_base_name="test", generated_files=[], use_ffmpeg=False
            )
            payload = path.read_text(encoding="utf-8")
            self.assertNotIn("super-secret-compatible", payload)
            data = json.loads(payload)
            self.assertEqual(data["text_engine"], core.TEXT_ENGINE_OPENAI_COMPATIBLE)
            self.assertTrue(data["text_sends_data_off_device"])

class Phase4ProviderTests(unittest.TestCase):
    def test_ollama_provider_reads_chat_content(self):
        payload = {"message": {"content": "resultado"}}
        with patch("urllib.request.urlopen", return_value=_FakeHTTPResponse(payload)):
            provider = tp.OllamaProvider("http://localhost:11434", "qwen3:4b")
            out = provider.generate([{"role": "user", "content": "hola"}])
        self.assertEqual(out, "resultado")

    def test_ollama_long_summary_keeps_chunk_and_consolidation_strategy(self):
        provider = _FakeProvider()
        progress = []
        text = "A" * 40 + "\n\n" + "B" * 40 + "\n\n" + "C" * 40
        out = tp.process_text(provider, text, "resume", progress.append, "resumen", chunk_chars=50)
        self.assertGreaterEqual(len(provider.calls), 4)  # >=3 trozos + consolidación
        self.assertTrue(out.startswith("salida-"))
        self.assertTrue(any("consolidando" in p.lower() for p in progress))

    def test_structured_evidence_drops_non_literal_quotes(self):
        raw = json.dumps({
            "ideas_principales": [], "temas": [], "posiciones_argumentos": [],
            "tensiones_contradicciones": [], "conceptos_instituciones": [],
            "citas_relevantes": [
                {"hablante": "H1", "timecode": "00:01", "texto": "frase exacta"},
                {"hablante": "H1", "timecode": "00:02", "texto": "frase inventada"},
            ],
            "dudas_revision": [],
        })
        out = json.loads(tp._normalize_json_evidence(raw, "Aquí aparece una frase exacta en la transcripción."))
        self.assertEqual([q["texto"] for q in out["citas_relevantes"]], ["frase exacta"])

    def test_compatible_api_allows_empty_key_for_local_servers(self):
        fake_client = object()
        with patch.object(tp, "HAS_OPENAI", True), patch.object(tp, "OpenAI", return_value=fake_client) as ctor:
            provider = tp.OpenAICompatibleProvider("http://localhost:1234/v1", "", "modelo-local")
        self.assertIs(provider.client, fake_client)
        kwargs = ctor.call_args.kwargs
        self.assertEqual(kwargs["base_url"], "http://localhost:1234/v1")
        self.assertTrue(kwargs["api_key"])

    def test_compatible_key_environment_name(self):
        with patch.dict(os.environ, {"JERONIMO_COMPATIBLE_API_KEY": "secret-phase4"}, clear=False):
            self.assertEqual(environment_secret("openai_compatible"), "secret-phase4")



    def test_core_chat_process_delegates_to_provider_layer(self):
        fake_provider = object()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = core.TranscriberConfig(
                api_key="", input_dir=root, output_dir=root, work_dir=root,
                text_engine=core.TEXT_ENGINE_OPENAI_COMPATIBLE,
                compatible_api_url="http://localhost:1234/v1",
                chat_model="modelo-local", do_summary=True, summary_prompt="resume",
            )
            with patch.object(core, "create_provider", return_value=fake_provider), \
                 patch.object(core, "_provider_process_text", return_value="resultado") as process:
                out = core.chat_process(None, cfg, "texto", "prompt", lambda _m: None, "resumen")
            self.assertEqual(out, "resultado")
            self.assertIs(process.call_args.args[0], fake_provider)

class Phase4RegressionTests(unittest.TestCase):
    def test_asr_and_diarization_functions_are_unchanged_from_phase3(self):
        for name, expected in ASR_BASELINE_HASHES.items():
            source = inspect.getsource(getattr(core, name)).encode("utf-8")
            self.assertEqual(hashlib.sha256(source).hexdigest(), expected, name)

    def test_phase4_version(self):
        self.assertTrue(bool(core.APP_VERSION))


if __name__ == "__main__":
    unittest.main()
