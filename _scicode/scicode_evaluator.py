"""Official-style subprocess evaluator for one SciCode subproblem."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


class SciCodeEvaluator:
    def __init__(self, h5py_file: str | os.PathLike[str], timeout: int = 1800) -> None:
        self.h5py_file = Path(h5py_file).expanduser().resolve()
        self.timeout = timeout

    def ensure_ready(self) -> None:
        if not self.h5py_file.is_file():
            raise FileNotFoundError(
                "SciCode numerical targets are missing. Run "
                "`python dataset/download_scicode_data.py` or set SCICODE_H5_PATH. "
                f"Expected: {self.h5py_file}"
            )

    def evaluate_step(self, step: dict[str, Any], code: str) -> dict[str, Any]:
        self.ensure_ready()
        step_id = str(step["step_number"])
        tests = list(step.get("test_cases") or [])

        with tempfile.TemporaryDirectory(prefix=f"adas_scicode_{step_id}_") as tmp:
            script_path = Path(tmp) / f"{step_id}.py"
            lines = [code.rstrip(), ""]
            lines.extend(
                [
                    "from _scicode.scicode_h5 import process_hdf5_to_tuple",
                    (
                        "targets = process_hdf5_to_tuple("
                        f"{step_id!r}, {len(tests)}, {str(self.h5py_file)!r})"
                    ),
                    "",
                ]
            )
            for index, test in enumerate(tests):
                lines.extend([f"target = targets[{index}]", ""])
                lines.extend(str(test).splitlines())
            script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            env = os.environ.copy()
            project_root = str(Path(__file__).resolve().parents[1])
            existing_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                project_root
                if not existing_pythonpath
                else os.pathsep.join([project_root, existing_pythonpath])
            )
            try:
                completed = subprocess.run(
                    [sys.executable, str(script_path)],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=tmp,
                    env=env,
                )
            except subprocess.TimeoutExpired as exc:
                return {
                    "passed": False,
                    "status": "timeout",
                    "returncode": None,
                    "stdout": exc.stdout or "",
                    "stderr": exc.stderr or "",
                }

        return {
            "passed": completed.returncode == 0,
            "status": "pass" if completed.returncode == 0 else "fail",
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
