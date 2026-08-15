from sdcpp_hooks.hardware import collect_run_environment, format_host_summary


def test_collect_run_environment_always_records_torch_and_runtime(monkeypatch):
    monkeypatch.setattr("sdcpp_hooks.hardware._nvidia_field", lambda query: "")
    monkeypatch.setattr("sdcpp_hooks.hardware._cuda_from_smi", lambda: "")
    monkeypatch.setattr(
        "sdcpp_hooks.hardware._torch_info",
        lambda: {
            "torch_version": "not installed (sd-cli uses ggml, not PyTorch)",
            "torch_cuda": "",
        },
    )

    host = collect_run_environment(help_text="stable-diffusion.cpp v0", binary="/sd-cli")

    assert host["torch_version"] == "not installed (sd-cli uses ggml, not PyTorch)"
    assert host["runtime"] == "stable-diffusion.cpp / ggml"
    assert host["sd_cli_binary"] == "/sd-cli"
    assert host["sd_cli_version"] == "stable-diffusion.cpp v0"
    assert "gpu_name" in host
    assert "cuda_version" in host


def test_format_host_summary_includes_required_facts():
    summary = format_host_summary(
        {
            "gpu_name": "NVIDIA L4",
            "cuda_version": "12.4",
            "torch_version": "2.4.0",
            "driver_version": "550.90.07",
            "modal_gpu": "L4",
        },
        duration_ms=12400,
    )

    assert summary.startswith("12.4s")
    assert "gpu NVIDIA L4" in summary
    assert "cuda 12.4" in summary
    assert "torch 2.4.0" in summary
    assert "driver 550.90.07" in summary
    assert "modal L4" in summary
