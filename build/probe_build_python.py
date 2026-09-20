from __future__ import annotations

import json
import sys


def main() -> int:
    version = sys.version_info[:2]
    ok = (3, 10) <= version <= (3, 12)
    print(json.dumps({
        "ok": ok,
        "python": sys.executable,
        "python_version": sys.version.split()[0],
    }), flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
