from pathlib import Path

import pytest

from sdcpp_hooks.web_jobs import JobService


def test_job_service_stores_parallelism(tmp_path: Path):
    service = JobService(tmp_path)
    job = service.create_job(
        [{"prompt": "a cat"}, {"prompt": "a dog"}],
        {"recipe": "sd15", "parallelism": 4},
    )

    assert job["parallelism"] == 4
    assert job["config"]["parallelism"] == 4
    assert job["total_images"] == 2


def test_job_service_rejects_unsupported_parallelism(tmp_path: Path):
    service = JobService(tmp_path)

    with pytest.raises(ValueError, match="parallelism"):
        service.create_job(
            [{"prompt": "a cat"}],
            {"recipe": "sd15", "parallelism": 3},
        )


def test_parallelism_is_not_part_of_model_affinity(tmp_path: Path):
    service = JobService(tmp_path)
    one = service.create_job(
        [{"prompt": "one"}],
        {"recipe": "z-image-turbo", "gpu": "L40S", "parallelism": 1},
    )
    four = service.create_job(
        [{"prompt": "four"}],
        {"recipe": "z-image-turbo", "gpu": "L40S", "parallelism": 4},
    )

    assert one["affinity_key"] == four["affinity_key"]
    assert one["parallelism"] == 1
    assert four["parallelism"] == 4
