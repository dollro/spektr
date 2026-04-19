"""Tests for config.observability.

We don't exercise Logfire end-to-end (that needs a running collector);
we verify the important contracts: idempotency, local-only default,
and that logfire.configure gets the right kwargs.
"""

from __future__ import annotations

from unittest.mock import patch

import config.observability as obs


def _reset_obs() -> None:
    """Reset the module-level initialised flag between tests."""
    obs._initialized = False


class TestSetupObservability:
    def test_is_idempotent(self) -> None:
        _reset_obs()
        with patch("logfire.configure") as mock_cfg:
            obs.setup_observability()
            obs.setup_observability()
            obs.setup_observability()
        assert mock_cfg.call_count == 1

    def test_local_only_blocks_network_send(self) -> None:
        _reset_obs()
        with (
            patch("logfire.configure") as mock_cfg,
            patch("logfire.instrument_pydantic_ai"),
            patch("logfire.instrument_httpx"),
            patch("config.settings.settings") as mock_settings,
        ):
            mock_settings.logfire_token = ""
            mock_settings.observability_local_only = True
            mock_settings.service_name = "spektr"
            obs.setup_observability()

        assert mock_cfg.call_args.kwargs["send_to_logfire"] is False
        assert mock_cfg.call_args.kwargs["service_name"] == "spektr"

    def test_token_plus_cloud_mode_ships(self) -> None:
        _reset_obs()
        with (
            patch("logfire.configure") as mock_cfg,
            patch("logfire.instrument_pydantic_ai"),
            patch("logfire.instrument_httpx"),
            patch("config.settings.settings") as mock_settings,
        ):
            mock_settings.logfire_token = "secret"
            mock_settings.observability_local_only = False
            mock_settings.service_name = "spektr"
            obs.setup_observability()

        assert mock_cfg.call_args.kwargs["send_to_logfire"] is True
        assert mock_cfg.call_args.kwargs["token"] == "secret"

    def test_missing_token_forces_local(self) -> None:
        """Even with local_only=false, no token means no shipping."""
        _reset_obs()
        with (
            patch("logfire.configure") as mock_cfg,
            patch("logfire.instrument_pydantic_ai"),
            patch("logfire.instrument_httpx"),
            patch("config.settings.settings") as mock_settings,
        ):
            mock_settings.logfire_token = ""
            mock_settings.observability_local_only = False
            mock_settings.service_name = "spektr"
            obs.setup_observability()

        assert mock_cfg.call_args.kwargs["send_to_logfire"] is False


class TestInstrumentFastAPI:
    def test_swallows_errors_silently(self) -> None:
        with patch("logfire.instrument_fastapi", side_effect=RuntimeError("boom")):
            obs.instrument_fastapi(object())  # no exception
