from __future__ import annotations

import subprocess
from pathlib import Path


class EngineError(RuntimeError):
    def __init__(self, argv: list[str], returncode: int, stdout: str, stderr: str) -> None:
        self.argv = argv
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        super().__init__(f"engine exited {returncode}: {stderr.strip() or stdout.strip()}")


def _output_from_argv(argv: list[str]) -> Path | None:
    for name in ("--output", "-o"):
        if name in argv:
            return Path(argv[argv.index(name) + 1])
    return None


def run_cli(argv: list[str], workdir: Path) -> list[Path]:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        argv,
        cwd=workdir,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n", flush=True)
    if completed.stderr:
        print(completed.stderr, end="" if completed.stderr.endswith("\n") else "\n", flush=True)
    if completed.returncode != 0:
        raise EngineError(argv, completed.returncode, completed.stdout, completed.stderr)

    output = _output_from_argv(argv)
    if output is not None and output.exists():
        return [output]

    generated = sorted(workdir.glob("output*.png")) + sorted(workdir.glob("output*.jpg"))
    return generated
