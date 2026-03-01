"""Claude CLI-based LLM response feature group."""

from __future__ import annotations

import json
import subprocess  # nosec B404
from typing import Any, Dict, List

from mloda.user import Options
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

from rag_integration.feature_groups.rag_pipeline.llm_response.base import BaseLLMResponse


class ClaudeCliResponse(BaseLLMResponse):
    """
    LLM response via the Claude CLI (claude -p).

    Calls 'claude -p --output-format json' as a subprocess. Works with a Claude
    subscription (no API key needed).

    Config-based matching:
        llm_method="claude_cli"

    Additional options:
        - allowed_tools: Comma-separated tools to allow (default: none)
        - max_turns: Maximum conversation turns (default: 1)
    """

    ALLOWED_TOOLS = "allowed_tools"
    MAX_TURNS = "max_turns"

    LLM_METHODS = {
        "claude_cli": "Claude CLI (claude -p) response generation",
    }

    PROPERTY_MAPPING = {
        BaseLLMResponse.LLM_METHOD: {
            "claude_cli": "Claude CLI (claude -p) response generation",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        BaseLLMResponse.QUERY: {
            "explanation": "The user question to answer",
            DefaultOptionKeys.context: True,
        },
        BaseLLMResponse.CONTEXT: {
            "explanation": "Retrieved context to include in the prompt (list or string)",
            DefaultOptionKeys.context: True,
        },
        BaseLLMResponse.SYSTEM_PROMPT: {
            "explanation": "System prompt for the LLM",
            DefaultOptionKeys.context: True,
        },
        ALLOWED_TOOLS: {
            "explanation": "Comma-separated tools to allow for Claude CLI",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.default: "",
        },
        MAX_TURNS: {
            "explanation": "Maximum conversation turns for Claude CLI",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.default: 1,
        },
    }

    @classmethod
    def _get_allowed_tools(cls, options: Options) -> str:
        """Get allowed_tools from options, default empty string."""
        val = options.get(cls.ALLOWED_TOOLS)
        if val is None:
            return ""
        return str(val)

    @classmethod
    def _get_max_turns(cls, options: Options) -> int:
        """Get max_turns from options, default 1."""
        val = options.get(cls.MAX_TURNS)
        if val is None:
            return 1
        return int(val)

    @classmethod
    def _build_prompt(cls, query: str, context: str, system_prompt: str) -> str:
        """Assemble the full prompt text for Claude CLI."""
        parts: List[str] = [system_prompt, ""]
        if context:
            parts.append("Context:")
            parts.append(context)
            parts.append("")
        parts.append("Question:")
        parts.append(query)
        return "\n".join(parts)

    @classmethod
    def _call_claude_cli(cls, prompt: str, allowed_tools: str, max_turns: int) -> str:
        """
        Run claude -p and return the result text.

        Raises ValueError on non-zero exit code.
        """
        cmd: List[str] = ["claude", "-p", "--output-format", "json", "--max-turns", str(max_turns)]
        if allowed_tools:
            cmd.extend(["--allowedTools", allowed_tools])

        result = subprocess.run(cmd, input=prompt, capture_output=True, text=True)  # nosec B603
        if result.returncode != 0:
            raise ValueError(f"claude -p failed (exit {result.returncode}): {result.stderr}")

        parsed: Dict[str, Any] = json.loads(result.stdout)
        return str(parsed.get("result", result.stdout))

    @classmethod
    def _generate(cls, query: str, context: str, system_prompt: str, options: Options) -> str:
        """Generate a response by calling Claude CLI."""
        allowed_tools = cls._get_allowed_tools(options)
        max_turns = cls._get_max_turns(options)
        prompt = cls._build_prompt(query, context, system_prompt)
        return cls._call_claude_cli(prompt, allowed_tools, max_turns)
