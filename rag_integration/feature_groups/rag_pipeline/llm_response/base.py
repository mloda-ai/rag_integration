"""Base class for LLM response feature groups."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Set, Type, Union

from mloda.provider import DataCreator, FeatureGroup, ComputeFramework, FeatureSet
from mloda.user import Options, FeatureName
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Answer the question using the provided context."


class BaseLLMResponse(FeatureGroup):
    """
    Base class for LLM response feature groups.

    This is a ROOT feature (like BaseRetriever): it has no input features
    and produces an LLM-generated response from a query and optional context.

    Configuration via Options:
        - query: The user question (required)
        - context: Retrieved context to include in the prompt (optional, list or string)
        - system_prompt: System prompt for the LLM (optional, has default)
        - llm_method: Which LLM implementation to use (discriminator)

    Output rows contain: llm_response (the generated text)
    """

    QUERY = "query"
    CONTEXT = "context"
    SYSTEM_PROMPT = "system_prompt"
    LLM_METHOD = "llm_method"

    LLM_METHODS: Dict[str, str] = {}

    PROPERTY_MAPPING = {
        LLM_METHOD: {
            "explanation": "Which LLM implementation to use",
            DefaultOptionKeys.context: True,
        },
        QUERY: {
            "explanation": "The user question to answer",
            DefaultOptionKeys.context: True,
        },
        CONTEXT: {
            "explanation": "Retrieved context to include in the prompt (list or string)",
            DefaultOptionKeys.context: True,
        },
        SYSTEM_PROMPT: {
            "explanation": "System prompt for the LLM",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.default: DEFAULT_SYSTEM_PROMPT,
        },
    }

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PythonDictFramework}

    @classmethod
    def input_data(cls) -> DataCreator:
        return DataCreator({"llm_response"})

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Any = None,
    ) -> bool:
        """Match features named 'llm_response' exactly."""
        if isinstance(feature_name, FeatureName):
            feature_name = feature_name.name
        return feature_name == "llm_response"

    def input_features(self, options: Options, feature_name: FeatureName) -> None:
        """Root feature: no input features."""
        return None

    @classmethod
    def _get_query(cls, options: Options) -> str:
        """Get query from options. Raises ValueError if missing."""
        val = options.get(cls.QUERY)
        if val is None:
            raise ValueError("LLM response requires 'query' in options.")
        return str(val)

    @classmethod
    def _get_context(cls, options: Options) -> str:
        """Get context from options. Returns empty string if missing."""
        val = options.get(cls.CONTEXT)
        if val is None:
            return ""
        if isinstance(val, list):
            return "\n".join(str(item) for item in val)
        return str(val)

    @classmethod
    def _get_system_prompt(cls, options: Options) -> str:
        """Get system prompt from options, with default."""
        val = options.get(cls.SYSTEM_PROMPT)
        if val is None:
            return DEFAULT_SYSTEM_PROMPT
        return str(val)

    @classmethod
    @abstractmethod
    def _generate(cls, query: str, context: str, system_prompt: str, options: Options) -> str:
        """
        Generate a response from the LLM.

        Args:
            query: The user question
            context: Retrieved context (may be empty)
            system_prompt: System prompt for the LLM
            options: Full options for additional configuration

        Returns:
            The generated response text
        """
        ...

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> List[Dict[str, Any]]:
        """Run LLM generation: extract options, delegate to _generate, return result."""
        for feature in features.features:
            options = feature.options

            query = cls._get_query(options)
            context = cls._get_context(options)
            system_prompt = cls._get_system_prompt(options)

            response = cls._generate(query, context, system_prompt, options)
            return [{"llm_response": response}]

        return []
