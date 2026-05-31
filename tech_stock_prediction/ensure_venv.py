from pathlib import Path
import os
import sys


def restart_with_project_venv():
    project_root = Path(__file__).resolve().parent
    venv_python = project_root / ".venv" / "bin" / "python"

    if not venv_python.exists():
        return

    if Path(sys.prefix).resolve() != (project_root / ".venv").resolve():
        os.execv(str(venv_python), [str(venv_python), *sys.argv])
