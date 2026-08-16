from pathlib import Path

from sdcpp_hooks.fast_fetch import (
    add_token_to_civitai_url,
    argv_for_log,
    detect_hf_cli,
    download_with_aria2c,
    download_with_hf_cli,
    fast_fetch,
    normalize_huggingface_url,
    parse_hf_url_for_cli,
    prepare_url,
    redact_url_for_log,
)


def test_redact_and_normalize_urls():
    assert "/resolve/" in normalize_huggingface_url(
        "https://huggingface.co/org/repo/blob/main/a.safetensors"
    )
    redacted = redact_url_for_log("https://civitai.com/api/download/models/1?token=secret")
    assert "secret" not in redacted
    assert "token=" in redacted


def test_civitai_token_comes_from_env(monkeypatch):
    monkeypatch.setenv("CIVITAI_TOKEN", "from-env")
    url = add_token_to_civitai_url("https://civitai.com/api/download/models/1")
    assert "token=from-env" in url
    already = add_token_to_civitai_url(
        "https://civitai.com/api/download/models/1?token=kept"
    )
    assert "token=kept" in already


def test_parse_hf_url_for_cli_files_and_datasets():
    parsed = parse_hf_url_for_cli(
        "https://huggingface.co/org/repo/blob/dev/transformer/a.safetensors"
    )
    assert parsed == ("org/repo", "transformer/a.safetensors", "dev", "model")
    dataset = parse_hf_url_for_cli(
        "https://huggingface.co/datasets/org/data/resolve/main/weights.gguf"
    )
    assert dataset == ("org/data", "weights.gguf", "main", "dataset")
    assert parse_hf_url_for_cli("https://example.com/a.bin") is None


def test_argv_for_log_hides_authorization_and_query_tokens():
    logged = argv_for_log(
        [
            "aria2c",
            "--header=Authorization: Bearer secret-token",
            "https://civitai.com/api/download/models/1?token=secret-token",
        ]
    )
    assert "secret-token" not in " ".join(logged)
    assert logged[1] == "--header=Authorization: Bearer ***"


def test_aria2c_uses_argv_not_shell(tmp_path, monkeypatch):
    dest = tmp_path / "model.safetensors"
    seen = {}

    monkeypatch.setattr("sdcpp_hooks.fast_fetch.shutil.which", lambda name: "/usr/bin/aria2c")

    def fake_run(argv):
        seen["argv"] = list(argv)
        dest.write_bytes(b"weights")
        return 0

    monkeypatch.setattr("sdcpp_hooks.fast_fetch.run_command", fake_run)

    assert download_with_aria2c(
        "https://huggingface.co/org/repo/resolve/main/model.safetensors",
        dest,
        {"Authorization": "Bearer secret-token", "User-Agent": "sdcpp-modal-cli/0.1"},
    )
    assert seen["argv"][0] == "/usr/bin/aria2c"
    assert "-x" in seen["argv"]
    assert "16" in seen["argv"]
    assert dest.name in seen["argv"]
    assert any(item.startswith("--header=Authorization: Bearer secret-token") for item in seen["argv"])
    assert not any("shell" in str(item) for item in seen["argv"])


def test_hf_cli_fallback_moves_downloaded_file(tmp_path, monkeypatch):
    dest = tmp_path / "out.safetensors"
    seen = {}

    monkeypatch.setattr("sdcpp_hooks.fast_fetch.detect_hf_cli", lambda: "hf")

    def fake_run(argv):
        seen["argv"] = list(argv)
        local_dir = Path(argv[argv.index("--local-dir") + 1])
        nested = local_dir / "transformer" / "a.safetensors"
        nested.parent.mkdir(parents=True)
        nested.write_bytes(b"hf-cli")
        return 0

    monkeypatch.setattr("sdcpp_hooks.fast_fetch.run_command", fake_run)

    assert download_with_hf_cli(
        "https://huggingface.co/org/repo/resolve/main/transformer/a.safetensors",
        dest,
    )
    assert dest.read_bytes() == b"hf-cli"
    assert seen["argv"][:4] == ["hf", "download", "org/repo", "transformer/a.safetensors"]


def test_fast_fetch_falls_back_aria2_then_hf_then_urllib(tmp_path, monkeypatch):
    dest = tmp_path / "a.bin"
    calls = []

    monkeypatch.setattr(
        "sdcpp_hooks.fast_fetch.download_with_aria2c",
        lambda url, dest, headers: calls.append("aria2") or False,
    )
    monkeypatch.setattr(
        "sdcpp_hooks.fast_fetch.download_with_hf_cli",
        lambda url, dest: calls.append("hf") or False,
    )
    def fake_urllib(url, dest, headers):
        dest.write_bytes(b"urllib")
        return dest

    monkeypatch.setattr("sdcpp_hooks.fast_fetch.download_with_urllib", fake_urllib)

    path = fast_fetch(
        "https://huggingface.co/org/repo/blob/main/a.bin",
        dest,
        {},
    )
    assert path.read_bytes() == b"urllib"
    assert calls == ["aria2", "hf"]


def test_prepare_url_adds_civitai_token_from_header():
    url = prepare_url(
        "https://civitai.com/api/download/models/9",
        {"Authorization": "Bearer abc"},
    )
    assert "token=abc" in url


def test_detect_hf_cli_prefers_hf(monkeypatch):
    monkeypatch.setattr(
        "sdcpp_hooks.fast_fetch.shutil.which",
        lambda name: "/usr/bin/hf" if name == "hf" else None,
    )
    assert detect_hf_cli() == "/usr/bin/hf"
