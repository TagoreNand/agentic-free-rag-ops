from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class CodeRunResult:
    exit_code: int
    stdout: str
    stderr: str


def run_python_snippet(code: str, timeout_s: int = 20) -> CodeRunResult:
    """
    Runs Python code in a subprocess with a timeout.

    Security note: this is NOT a true sandbox. Only enable if you trust the code.
    """
    # Use a temp file so tracebacks have stable filenames.
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        path = f.name
        f.write(code)

    env = os.environ.copy()
    # Avoid interactive behavior; also reduce risk of leaking secrets via env to child.
    # (Still not a guarantee.)
    env.pop("OPENAI_API_KEY", None)

    proc = subprocess.run(
        [sys.executable, path],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env=env,
    )
    return CodeRunResult(exit_code=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)

