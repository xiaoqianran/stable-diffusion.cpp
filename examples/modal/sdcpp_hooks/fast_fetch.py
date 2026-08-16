from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Mapping, Sequence
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen


ARIA2_CONNECTIONS = 16
ARIA2_SPLIT = 16
ARIA2_CHUNK = "1M"


def redact_url_for_log(url: str) -> str:
    try:
        parsed = urlparse(url)
        if not parsed.query:
            return url
        query = parse_qs(parsed.query)
        for key in list(query):
            if key.lower() in {"token", "auth", "authorization"}:
                query[key] = ["***"]
        return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))
    except Exception:
        return url


def normalize_huggingface_url(url: str) -> str:
    parsed = urlparse(url)
    if "/blob/" in parsed.path:
        parsed = parsed._replace(path=parsed.path.replace("/blob/", "/resolve/"))
    return urlunparse(parsed)


def add_token_to_civitai_url(url: str, token: str | None = None) -> str:
    token = token or os.environ.get("CIVITAI_TOKEN")
    if not token or "civitai.com" not in url:
        return url
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    if "token" not in query:
        query["token"] = [token]
    return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))


def parse_hf_url_for_cli(url: str) -> tuple[str, str, str, str] | None:
    parsed = urlparse(normalize_huggingface_url(url))
    host = (parsed.netloc or "").lower()
    if host not in {"huggingface.co", "www.huggingface.co", "hf-mirror.com"}:
        return None
    parts = parsed.path.strip("/").split("/")
    repo_type = "model"
    index = 0
    if parts and parts[0] in {"datasets", "spaces"}:
        repo_type = "dataset" if parts[0] == "datasets" else "space"
        index = 1
    if len(parts) < index + 2:
        return None
    repo_id = f"{parts[index]}/{parts[index + 1]}"
    rest = parts[index + 2 :]
    revision = "main"
    file_in_repo = ""
    if rest and rest[0] == "resolve":
        if len(rest) >= 2:
            revision = rest[1]
            file_in_repo = "/".join(rest[2:])
    else:
        file_in_repo = "/".join(rest)
    if not file_in_repo:
        return None
    return repo_id, file_in_repo, revision, repo_type


def detect_hf_cli() -> str | None:
    return shutil.which("hf") or shutil.which("huggingface-cli")


def argv_for_log(argv: Sequence[str]) -> list[str]:
    logged: list[str] = []
    for item in argv:
        text = str(item)
        lower = text.lower()
        if lower.startswith("--header=") and "authorization" in lower:
            logged.append("--header=Authorization: Bearer ***")
            continue
        if text.startswith("http://") or text.startswith("https://"):
            logged.append(redact_url_for_log(text))
            continue
        logged.append(text)
    return logged


def run_command(argv: Sequence[str]) -> int:
    print(" ".join(argv_for_log(argv)), flush=True)
    process = subprocess.Popen(
        list(argv),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    return process.wait()


def _bearer_token(headers: Mapping[str, str]) -> str | None:
    auth = headers.get("Authorization") or headers.get("authorization")
    if not auth:
        return None
    prefix = "bearer "
    if auth.lower().startswith(prefix):
        return auth[len(prefix) :].strip()
    return None


def prepare_url(url: str, headers: Mapping[str, str] | None = None) -> str:
    url = normalize_huggingface_url(url)
    headers = headers or {}
    if "civitai.com" in url:
        url = add_token_to_civitai_url(url, _bearer_token(headers) or os.environ.get("CIVITAI_TOKEN"))
    return url


def download_with_aria2c(url: str, dest: Path, headers: Mapping[str, str]) -> bool:
    aria2c = shutil.which("aria2c")
    if not aria2c:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        aria2c,
        "-x",
        str(ARIA2_CONNECTIONS),
        "-s",
        str(ARIA2_SPLIT),
        "-c",
        "-k",
        ARIA2_CHUNK,
        "--file-allocation=none",
        "--auto-file-renaming=false",
        "--allow-overwrite=true",
        "--console-log-level=notice",
        "--summary-interval=1",
        "--show-console-readout=true",
        "-d",
        str(dest.parent),
        "-o",
        dest.name,
    ]
    for key, value in headers.items():
        argv.append(f"--header={key}: {value}")
    argv.append(url)
    return run_command(argv) == 0 and dest.exists() and dest.stat().st_size > 0


def download_with_hf_cli(url: str, dest: Path) -> bool:
    cli = detect_hf_cli()
    parsed = parse_hf_url_for_cli(url)
    if not cli or parsed is None:
        return False
    repo_id, file_in_repo, revision, repo_type = parsed
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sdcpp-hf-") as tmp:
        argv = [
            cli,
            "download",
            repo_id,
            file_in_repo,
            "--revision",
            revision,
            "--repo-type",
            repo_type,
            "--local-dir",
            tmp,
        ]
        if run_command(argv) != 0:
            return False
        source = Path(tmp) / file_in_repo
        if not source.is_file():
            matches = [path for path in Path(tmp).rglob(Path(file_in_repo).name) if path.is_file()]
            if len(matches) != 1:
                return False
            source = matches[0]
        if dest.exists():
            dest.unlink()
        shutil.move(str(source), str(dest))
    return dest.exists() and dest.stat().st_size > 0


def download_with_urllib(url: str, dest: Path, headers: Mapping[str, str]) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers=dict(headers))
    with urlopen(request) as response, dest.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    return dest


def fast_fetch(url: str, dest: Path, headers: Mapping[str, str]) -> Path:
    dest = Path(dest)
    url = prepare_url(url, headers)
    print(f"fetch {redact_url_for_log(url)} -> {dest}", flush=True)
    if download_with_aria2c(url, dest, headers):
        return dest
    if "huggingface.co" in url or "hf-mirror.com" in url:
        print("aria2c unavailable or failed; trying Hugging Face CLI", flush=True)
        if download_with_hf_cli(url, dest):
            return dest
    print("falling back to urllib", flush=True)
    return download_with_urllib(url, dest, headers)
