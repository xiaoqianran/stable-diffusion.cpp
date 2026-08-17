import pytest
from pydantic import ValidationError

from web.api import CreateJobBody, _config_from_body


def test_create_job_body_accepts_supported_parallelism():
    for value in (1, 2, 4):
        body = CreateJobBody(prompt="a cat", recipe="sd15", parallelism=value)
        assert _config_from_body(body)["parallelism"] == value


def test_create_job_body_rejects_other_parallelism():
    with pytest.raises(ValidationError):
        CreateJobBody(prompt="a cat", recipe="sd15", parallelism=3)
