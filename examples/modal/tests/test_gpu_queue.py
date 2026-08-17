from sdcpp_hooks.gpu_queue import GPUQueue


def test_single_gpu_queue_reports_running_and_waiting_positions():
    queue = GPUQueue(max_active=1)

    queue.enqueue("job-a")
    queue.enqueue("job-b")
    assert queue.acquire("job-a") is True

    snapshot = queue.snapshot("job-b")
    assert snapshot["running_job_ids"] == ["job-a"]
    assert snapshot["waiting_job_ids"] == ["job-b"]
    assert snapshot["queue_length"] == 1
    assert snapshot["job"]["state"] == "waiting"
    assert snapshot["job"]["ahead"] == 1
    assert snapshot["job"]["position"] == 2

    queue.release("job-a")
    assert queue.acquire("job-b") is True
    running = queue.snapshot("job-b")
    assert running["job"]["state"] == "running"
    assert running["job"]["ahead"] == 0
    queue.release("job-b")
    assert queue.snapshot()["state"] == "idle"


def test_cancel_removes_waiting_job_without_releasing_running_job():
    queue = GPUQueue(max_active=1)
    queue.enqueue("job-a")
    queue.enqueue("job-b")
    assert queue.acquire("job-a") is True

    queue.cancel("job-b")
    snapshot = queue.snapshot()
    assert snapshot["running_job_ids"] == ["job-a"]
    assert snapshot["waiting_job_ids"] == []


def test_queue_respects_configured_parallel_slots():
    queue = GPUQueue(max_active=2)
    queue.enqueue("job-a")
    queue.enqueue("job-b")
    queue.enqueue("job-c")

    assert queue.acquire("job-a") is True
    assert queue.acquire("job-b") is True
    snapshot = queue.snapshot("job-c")
    assert snapshot["running_count"] == 2
    assert snapshot["job"]["ahead"] == 2

    queue.release("job-a")
    assert queue.acquire("job-c") is True


def test_same_model_affinity_pulls_warm_job_forward():
    queue = GPUQueue(max_active=1, affinity_window=8, max_affinity_streak=4)
    queue.enqueue("warm", affinity_key="L40S::z-image-turbo")
    assert queue.acquire("warm") is True

    # FIFO would be flux first. While z-image is warm, the matching z-image job
    # should move ahead within the bounded affinity window.
    queue.enqueue("flux", affinity_key="L40S::flux2-klein")
    queue.enqueue("z-next", affinity_key="L40S::z-image-turbo")
    queue.release("warm")

    snapshot = queue.snapshot("z-next")
    assert snapshot["waiting_job_ids"] == ["z-next", "flux"]
    assert snapshot["job"]["position"] == 1
    assert snapshot["affinity"]["preferred"] == "L40S::z-image-turbo"


def test_affinity_streak_cap_returns_to_fifo_for_fairness():
    queue = GPUQueue(max_active=1, max_affinity_streak=2)

    queue.enqueue("a1", affinity_key="A")
    assert queue.acquire("a1") is True
    queue.release("a1")
    queue.enqueue("a2", affinity_key="A")
    assert queue.acquire("a2") is True
    queue.release("a2")

    # The warm affinity has reached its streak cap, so an older B stays ahead.
    queue.enqueue("b", affinity_key="B")
    queue.enqueue("a3", affinity_key="A")
    snapshot = queue.snapshot()
    assert snapshot["waiting_job_ids"] == ["b", "a3"]
    assert snapshot["affinity"]["preferred"] == ""


def test_job_snapshot_reports_affinity_key():
    queue = GPUQueue(max_active=1)
    queue.enqueue("job-a", affinity_key="RTX-PRO-6000::ideogram4")
    snapshot = queue.snapshot("job-a")
    assert snapshot["job"]["affinity_key"] == "RTX-PRO-6000::ideogram4"
