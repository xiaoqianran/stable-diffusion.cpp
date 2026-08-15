from decimal import Decimal

from sdcpp_hooks.cli import parse_argv
from sdcpp_hooks.cost import (
    PriceBook,
    billed_usd,
    default_plan,
    format_usd,
    gpu_rate_key,
    make_event,
)
from sdcpp_hooks.meter import ContainerMeter, Ledger, client_ledger, last_event, span


def test_gpu_rate_key_normalizes_aliases():
    assert gpu_rate_key("L4") == "gpu_hour_cost_l4"
    assert gpu_rate_key("A10") == "gpu_hour_cost_a10g"
    assert gpu_rate_key("A100-80GB") == "gpu_hour_cost_a100_80gb"


def test_price_book_estimates_l4_and_cpu():
    book = PriceBook()
    gpu_total, gpu_parts = book.estimate(default_plan("generate", gpu="L4"), 10)
    cpu_total, cpu_parts = book.estimate(default_plan("pull"), 10)

    assert "gpu" in gpu_parts
    assert "gpu" not in cpu_parts
    assert gpu_total > cpu_total
    assert gpu_parts["gpu"] == Decimal("0.8") * Decimal(10) / Decimal(3600)
    assert format_usd(gpu_total).startswith("$")


def test_billed_usd_does_not_double_count_session_and_remote():
    book = PriceBook()
    remote = make_event(
        event_id="r1",
        plan=default_plan("generate", gpu="L4"),
        duration_ms=10_000,
        book=book,
        phase="remote",
        trace_id="t1",
    )
    session = make_event(
        event_id="s1",
        plan=default_plan("session:gpu"),
        duration_ms=12_000,
        book=book,
        phase="session",
        trace_id="t1",
    )

    combined = billed_usd([session, remote])
    remote_only = billed_usd([remote])
    assert combined >= remote_only
    assert combined < Decimal(remote.usd) + Decimal(session.usd)


def test_span_and_ledger_record_a_billed_window(tmp_path, monkeypatch):
    monkeypatch.setenv("SDCPP_COST_LOG", str(tmp_path / "cost.jsonl"))
    with span(default_plan("ls"), phase="remote", ledger=client_ledger()):
        pass

    event = last_event()
    assert event is not None
    assert event.phase == "remote"
    assert event.duration_ms >= 0
    stored = client_ledger().read()
    assert stored[-1].name == "ls"


def test_container_meter_writes_worker_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("SDCPP_COST_WORKER_LOG", str(tmp_path / "worker.jsonl"))
    meter = ContainerMeter.start("SDEngine", gpu="L4")
    event = meter.stop()
    assert event.phase == "container"
    assert event.gpu == "L4"
    assert Ledger(tmp_path / "worker.jsonl").read()[0].name == "SDEngine"


def test_parse_cost_command():
    plain = parse_argv(["cost"])
    assert plain.action == "cost"
    assert plain.official is False
    assert parse_argv(["cost", "--official"]).official is True
