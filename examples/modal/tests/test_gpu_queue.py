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
