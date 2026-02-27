"""Tests for ClaudeCliResponse."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest
from mloda.user import Options

from rag_integration.feature_groups.rag_pipeline.llm_response.base import BaseLLMResponse
from rag_integration.feature_groups.rag_pipeline.llm_response.claude_cli import ClaudeCliResponse


class TestBuildPrompt:
    """Tests for _build_prompt assembly."""

    def test_with_context(self) -> None:
        """Should include context section when context is provided."""
        result = ClaudeCliResponse._build_prompt("What is X?", "X is a letter.", "You are helpful.")
        assert "Context:" in result
        assert "X is a letter." in result
        assert "Question:" in result
        assert "What is X?" in result
        assert "You are helpful." in result

    def test_without_context(self) -> None:
        """Should omit context section when context is empty."""
        result = ClaudeCliResponse._build_prompt("What is X?", "", "You are helpful.")
        assert "Context:" not in result
        assert "Question:" in result
        assert "What is X?" in result


class TestFeatureMatching:
    """Tests for match_feature_group_criteria."""

    def test_matches_llm_response(self) -> None:
        """Should match 'llm_response' feature name."""
        assert ClaudeCliResponse.match_feature_group_criteria("llm_response", Options())

    def test_rejects_other_names(self) -> None:
        """Should reject non-llm_response feature names."""
        assert not ClaudeCliResponse.match_feature_group_criteria("retrieved", Options())
        assert not ClaudeCliResponse.match_feature_group_criteria("docs", Options())
        assert not ClaudeCliResponse.match_feature_group_criteria("llm", Options())


class TestGetQuery:
    """Tests for _get_query extraction."""

    def test_missing_query_raises(self) -> None:
        """Should raise ValueError when query is missing."""
        with pytest.raises(ValueError, match="query"):
            BaseLLMResponse._get_query(Options())

    def test_present_query(self) -> None:
        """Should return query string."""
        assert BaseLLMResponse._get_query(Options({"query": "hello"})) == "hello"


class TestGetContext:
    """Tests for _get_context extraction."""

    def test_list_context(self) -> None:
        """Should join list items with newlines."""
        result = BaseLLMResponse._get_context(Options({"context": ["a", "b", "c"]}))
        assert result == "a\nb\nc"

    def test_string_context(self) -> None:
        """Should return string as-is."""
        result = BaseLLMResponse._get_context(Options({"context": "some text"}))
        assert result == "some text"

    def test_empty_context(self) -> None:
        """Should return empty string when context is missing."""
        result = BaseLLMResponse._get_context(Options())
        assert result == ""


class TestCallClaudeCli:
    """Tests for _call_claude_cli with mocked subprocess."""

    def test_success(self) -> None:
        """Should parse JSON result on success."""
        mock_output = json.dumps({"result": "The answer is 42."})
        mock_completed: Any = type("CompletedProcess", (), {"returncode": 0, "stdout": mock_output, "stderr": ""})()

        with patch("rag_integration.feature_groups.rag_pipeline.llm_response.claude_cli.subprocess.run") as mock_run:
            mock_run.return_value = mock_completed
            result = ClaudeCliResponse._call_claude_cli("test prompt", "", 1)

        assert result == "The answer is 42."
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[0][0] == ["claude", "-p", "--output-format", "json", "--max-turns", "1"]
        assert call_args[1]["input"] == "test prompt"

    def test_with_allowed_tools(self) -> None:
        """Should pass --allowedTools when provided."""
        mock_output = json.dumps({"result": "done"})
        mock_completed: Any = type("CompletedProcess", (), {"returncode": 0, "stdout": mock_output, "stderr": ""})()

        with patch("rag_integration.feature_groups.rag_pipeline.llm_response.claude_cli.subprocess.run") as mock_run:
            mock_run.return_value = mock_completed
            ClaudeCliResponse._call_claude_cli("prompt", "Bash", 3)

        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert "--allowedTools" in cmd
        assert "Bash" in cmd
        assert "--max-turns" in cmd
        assert "3" in cmd

    def test_failure_raises(self) -> None:
        """Should raise ValueError on non-zero exit code."""
        mock_completed: Any = type("CompletedProcess", (), {"returncode": 1, "stdout": "", "stderr": "error message"})()

        with patch("rag_integration.feature_groups.rag_pipeline.llm_response.claude_cli.subprocess.run") as mock_run:
            mock_run.return_value = mock_completed
            with pytest.raises(ValueError, match="claude -p failed"):
                ClaudeCliResponse._call_claude_cli("test prompt", "", 1)
