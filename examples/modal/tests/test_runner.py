import sys
from pathlib import Path

from sdcpp_hooks.runner import EngineError, run_cli


def test_run_cli_collects_files_written_to_output_path(tmp_path):
    script = tmp_path / "fake_sd.py"
    script.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "out = Path(sys.argv[sys.argv.index('--output') + 1])\n"
        "out.write_bytes(b'image')\n",
        encoding="utf-8",
    )

    files = run_cli(
        [sys.executable, str(script), "--output", str(tmp_path / "out.png")],
        workdir=tmp_path,
    )

    assert [path.read_bytes() for path in files] == [b"image"]


def test_run_cli_raises_engine_error_on_nonzero_exit(tmp_path):
    script = tmp_path / "fail.py"
    script.write_text("import sys\nsys.stderr.write('boom')\nsys.exit(2)\n", encoding="utf-8")

    try:
        run_cli([sys.executable, str(script)], workdir=tmp_path)
    except EngineError as exc:
        assert exc.returncode == 2
        assert "boom" in exc.stderr
    else:
        raise AssertionError("EngineError was not raised")
