"""Shared "skip if the Hugging Face Hub is unreachable" helper for the tests
that load the tiny real checkpoint. execute_prompt() wraps a node's
exception in NodeExecutionError, and the checkpoint loader wraps network
errors in a RuntimeError ("can't reach the Hub"), so the whole __cause__
chain is checked.

Set PIXELFLOW_REQUIRE_HUB=1 (CI does) to make these tests fail instead of
skip, so a green run can't hide missing diffusion coverage."""

import os

import pytest
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
from requests.exceptions import ConnectionError as RequestsConnectionError

try:
    import httpx

    _HTTPX_ERRORS = (httpx.TransportError,)
except ImportError:  # pragma: no cover
    _HTTPX_ERRORS = ()

HUB_ERRORS = (HfHubHTTPError, LocalEntryNotFoundError, RequestsConnectionError) + _HTTPX_ERRORS


def skip_if_hub_unreachable(exc: BaseException) -> None:
    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, HUB_ERRORS):
            if os.environ.get("PIXELFLOW_REQUIRE_HUB") == "1":
                return  # let the caller re-raise: a hard failure
            pytest.skip(f"Hugging Face Hub unreachable: {current}")
        current = current.__cause__ or current.__context__
