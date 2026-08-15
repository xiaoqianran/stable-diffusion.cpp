from sdcpp_hooks.artifacts import list_cached_artifacts


def test_list_cached_artifacts_empty_dir(tmp_path):
    assert list_cached_artifacts(tmp_path) == []
    assert list_cached_artifacts(tmp_path / "missing") == []


def test_list_cached_artifacts_returns_relative_files_and_sizes(tmp_path):
    target = tmp_path / "hf" / "org" / "repo" / "main" / "model.safetensors"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"abcd")

    listed = list_cached_artifacts(tmp_path)

    assert listed == [
        {
            "path": "hf/org/repo/main/model.safetensors",
            "bytes": 4,
        }
    ]
