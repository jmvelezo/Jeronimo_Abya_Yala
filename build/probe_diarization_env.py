from __future__ import annotations

import json
import sys
import warnings

warnings.filterwarnings("ignore")


def main() -> int:
    try:
        import torch
        import whisperx
        from whisperx.diarize import DiarizationPipeline  # noqa: F401
        try:
            from importlib.metadata import version as package_version
            version = package_version("whisperx")
        except Exception:
            version = getattr(whisperx, "__version__", "unknown")
        payload = {
            "ok": True,
            "python": sys.executable,
            "python_version": sys.version.split()[0],
            "whisperx": str(version),
            "torch": str(torch.__version__),
            "cuda": bool(torch.cuda.is_available()),
        }
    except Exception as exc:
        payload = {
            "ok": False,
            "python": sys.executable,
            "python_version": sys.version.split()[0],
            "error": "%s: %s" % (type(exc).__name__, exc),
        }
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return 0 if payload.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
