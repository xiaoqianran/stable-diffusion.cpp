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
from sdcpp_hooks.cost_view import build_traces, ledger_report
from sdcpp_hooks.meter import ContainerMeter, Ledger, bind_task, client_ledger, last_event, span


def test_default_gpu_is_l40s(monkeypatch):
    monkeypatch.delenv("SDCPP_GPU", raising=False)
    from sdcpp_hooks.modal_meter import _gpu_name

    assert _gpu_name() == "L40S"


def test_gpu_rate_key_normalizes_aliases():
    assert gpu_rate_key("L4") == "gpu_hour_cost_l4"
    assert gpu_rate_key("L40S") == "gpu_hour_cost_l40s"
    assert gpu_rate_key("RTX-PRO-6000") == "gpu_hour_cost_rtx6000"
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


def test_make_event_records_per_second_and_task():
    book = PriceBook()
    event = make_event(
        event_id="e1",
        plan=default_plan("generate", gpu="L40S"),
        duration_ms=1000,
        book=book,
        phase="remote",
        job_id="job_1",
        recipe="sd15",
        image_id="img_1",
    )
    assert event.job_id == "job_1"
    assert event.image_id == "img_1"
    assert event.recipe == "sd15"
    assert Decimal(event.usd_per_second) > 0
    assert "gpu" in event.rates
    assert abs(Decimal(event.usd) - Decimal(event.usd_per_second)) < Decimal("0.000001")


def test_bind_task_attaches_job_to_span(tmp_path, monkeypatch):
    monkeypatch.setenv("SDCPP_COST_LOG", str(tmp_path / "cost.jsonl"))
    with bind_task(job_id="job_abc", recipe="sd15"):
        with span(default_plan("generate", gpu="L4"), phase="remote", ledger=client_ledger()):
            pass
    event = last_event()
    assert event is not None
    assert event.job_id == "job_abc"
    assert event.recipe == "sd15"


def test_build_traces_nests_remote_under_session():
    book = PriceBook()
    session = make_event(
        event_id="sess",
        plan=default_plan("gpu"),
        duration_ms=12_000,
        book=book,
        phase="session",
        trace_id="t-nest",
        parent_id="",
    )
    remote = make_event(
        event_id="rem",
        plan=default_plan("generate", gpu="L4"),
        duration_ms=10_000,
        book=book,
        phase="remote",
        trace_id="t-nest",
        parent_id="sess",
        job_id="job_1",
        image_id="img_9",
    )
    traces = build_traces([remote, session])
    assert traces[0]["chain"][0]["id"] == "sess"
    assert traces[0]["chain"][0]["depth"] == 0
    assert traces[0]["chain"][1]["id"] == "rem"
    assert traces[0]["chain"][1]["depth"] == 1
    assert traces[0]["job_ids"] == ["job_1"]
    assert any("gpu" in line for line in traces[0]["chain"][1]["breakdown_lines"])
    assert "×" in traces[0]["chain"][1]["line"]


def test_billed_app_session_is_not_its_own_parent(tmp_path, monkeypatch):
    from contextlib import contextmanager

    from sdcpp_hooks.modal_meter import billed_app, billed_remote

    class App:
        @contextmanager
        def run(self):
            yield

    class Fn:
        def remote(self, *args, **kwargs):
            return {"ok": True}

    monkeypatch.setenv("SDCPP_COST_LOG", str(tmp_path / "cost.jsonl"))
    with billed_app(App(), "gpu"):
        billed_remote(Fn(), name="generate", gpu=True)
    events = client_ledger().read()
    session = next(item for item in events if item.phase == "session")
    remote = next(item for item in events if item.phase == "remote")
    assert session.parent_id == ""
    assert remote.parent_id == session.id
    assert Decimal(remote.usd_per_second) > 0


def test_ledger_report_filters_job(tmp_path, monkeypatch):
    monkeypatch.setenv("SDCPP_COST_LOG", str(tmp_path / "cost.jsonl"))
    book = PriceBook()
    client_ledger().append(
        make_event(
            event_id="a",
            plan=default_plan("generate", gpu="L4"),
            duration_ms=1000,
            book=book,
            phase="remote",
            job_id="job_keep",
            trace_id="t-a",
        )
    )
    client_ledger().append(
        make_event(
            event_id="b",
            plan=default_plan("generate", gpu="L4"),
            duration_ms=1000,
            book=book,
            phase="remote",
            job_id="job_drop",
            trace_id="t-b",
        )
    )
    report = ledger_report(job_id="job_keep")
    assert report["event_count"] == 1
    assert report["jobs"]["job_keep"]["event_count"] == 1
    assert "job_drop" not in report["jobs"]
    assert report["billed"]["event_count"] == 1


def test_parse_cost_command():
    plain = parse_argv(["cost"])
    assert plain.action == "cost"
    assert plain.official is False
    assert parse_argv(["cost", "--official"]).official is True
