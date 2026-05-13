"""Verificaciones ligeras del proyecto sin depender de paquetes externos.

Este script sirve cuando aun no se pudieron instalar pandas/streamlit. Revisa que
los archivos Python compilen y que el repositorio pueda empaquetarse como ZIP.
"""
from __future__ import annotations

import compileall
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = ROOT / "dist" / "prediccionchachaia.zip"


def main() -> int:
    print("Verificando compilacion Python...")
    if not compileall.compile_dir(str(ROOT / "football_predictor"), quiet=1):
        return 1
    for target in [ROOT / "app.py", ROOT / "tests", ROOT / "scripts"]:
        if target.is_dir():
            ok = compileall.compile_dir(str(target), quiet=1)
        else:
            ok = compileall.compile_file(str(target), quiet=1)
        if not ok:
            return 1

    print("Creando ZIP de verificacion...")
    ZIP_PATH.parent.mkdir(exist_ok=True)
    result = subprocess.run(
        ["git", "archive", "--format=zip", f"--output={ZIP_PATH}", "HEAD"],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        return result.returncode
    print(f"OK: {ZIP_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
