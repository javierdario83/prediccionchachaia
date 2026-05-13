"""Crea un ZIP descargable del proyecto desde el commit actual."""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "dist" / "prediccionchachaia.zip"


def main() -> int:
    OUTPUT.parent.mkdir(exist_ok=True)
    result = subprocess.run(
        ["git", "archive", "--format=zip", f"--output={OUTPUT}", "HEAD"],
        cwd=ROOT,
        check=False,
    )
    if result.returncode == 0:
        print(f"ZIP creado en: {OUTPUT}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
