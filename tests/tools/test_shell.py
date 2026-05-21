from __future__ import annotations

from unittest.mock import patch


def test_run_command_returns_output():
    from agent.lc_tools import run_command
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = "hello"
        mock_run.return_value.stderr = ""
        result = run_command.invoke({"command": "echo hello"})
        assert result == "hello"


def test_run_command_returns_stderr_on_empty_stdout():
    from agent.lc_tools import run_command
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = "error"
        result = run_command.invoke({"command": "bad"})
        assert result == "error"
