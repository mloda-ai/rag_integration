"""Knowledge-graph source feature groups for the ``graph_rag`` family (issue #45).

A KG source is a ROOT FeatureGroup that emits a passage graph,
``{"nodes": [{doc_id, text}, ...], "edges": [[doc_id_a, doc_id_b], ...]}``,
under the ``knowledge_graph`` feature name. A graph_rag connector with the
``graph_source`` option set consumes this feature as its corpus instead of
carrying inline ``nodes`` + ``edges``, so an existing graph source is reused
rather than duplicated. It is a corpus source inside the family, not a seventh
connector family: it answers no query and returns no ranking.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

from mloda.provider import DataCreator, FeatureGroup, ComputeFramework, FeatureSet
from mloda.user import Options, FeatureName
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)

from rag_integration.feature_groups.connectors.errors import InvalidOptionError
from rag_integration.feature_groups.connectors.mixins import OptionsMixin


class BaseKnowledgeGraphSource(OptionsMixin, FeatureGroup):
    """Root FeatureGroup for knowledge-graph source backends.

    Mirrors the family-base shape (selector dict, criteria gating, one hook):
    a concrete declares its selector value in ``KG_BACKENDS`` and implements
    :meth:`_build_graph`. The output payload keys (``nodes``, ``edges``) are
    exactly what ``BaseGraphRagConnector`` reads from a graph-source row.
    """

    ROOT_FEATURE_NAME = "knowledge_graph"

    KG_BACKEND = "kg_backend"

    NODES_KEY = "nodes"
    EDGES_KEY = "edges"

    # Filled per concrete; empty on the base so it never matches.
    KG_BACKENDS: Dict[str, str] = {}

    PROPERTY_MAPPING = {
        KG_BACKEND: {"explanation": "Which knowledge-graph source backend to use"},
    }

    @classmethod
    def compute_framework_rule(cls) -> Optional[Set[Type[ComputeFramework]]]:
        return {PythonDictFramework}

    @classmethod
    def input_data(cls) -> DataCreator:
        return DataCreator({cls.ROOT_FEATURE_NAME})

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Any = None,
    ) -> bool:
        """Match the root feature name only for a backend this concrete declares."""
        if str(feature_name) != cls.ROOT_FEATURE_NAME:
            return False
        backend = options.get(cls.KG_BACKEND)
        return backend in cls.KG_BACKENDS

    def input_features(self, options: Options, feature_name: FeatureName) -> None:
        """Root feature: no input features (the graph definition arrives via Options)."""
        return None

    @classmethod
    @abstractmethod
    def _build_graph(cls, options: Options) -> Dict[str, Any]:
        """Build the ``{nodes, edges}`` payload from this backend's options."""
        ...

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> List[Dict[str, Any]]:
        """Emit the graph payload as a single row under the root feature name."""
        for feature in features.features:
            graph = cls._build_graph(feature.options)
            return [{cls.ROOT_FEATURE_NAME: graph}]
        return []


class TriplesKnowledgeGraph(BaseKnowledgeGraphSource):
    """Passage graph built from subject-predicate-object triples (``kg_backend="triples"``).

    ``triples`` is a list of ``[subject, predicate, object]``. Every distinct
    subject or object becomes a node (``doc_id`` = the entity, first-appearance
    order); a node's text is the entity followed by every triple it
    participates in, rendered as ``"subject predicate object"`` sentences, so
    lexical graph_rag scoring sees the relations. Each triple contributes one
    ``[subject, object]`` edge; self-loops are skipped and repeated pairs are
    emitted once.
    """

    TRIPLES = "triples"

    KG_BACKENDS = {
        "triples": "Passage graph from subject-predicate-object triples (pure Python, zero-download)",
    }

    PROPERTY_MAPPING = {
        BaseKnowledgeGraphSource.KG_BACKEND: {"explanation": "Use 'triples' for a triple-built passage graph"},
        TRIPLES: {"explanation": "Knowledge-graph triples: a list of [subject, predicate, object]"},
    }

    @classmethod
    def _resolve_triples(cls, options: Options) -> List[Tuple[str, str, str]]:
        raw = cls._require_option(options, cls.TRIPLES)
        if not isinstance(raw, (list, tuple)):
            raise InvalidOptionError(
                f"{cls.__name__} '{cls.TRIPLES}' must be a list of [subject, predicate, object], "
                f"got {type(raw).__name__}."
            )
        triples: List[Tuple[str, str, str]] = []
        for i, triple in enumerate(raw):
            if not isinstance(triple, (list, tuple)) or len(triple) != 3:
                raise InvalidOptionError(
                    f"{cls.__name__} triple at index {i} is not a [subject, predicate, object] item: {triple!r}."
                )
            triples.append((str(triple[0]), str(triple[1]), str(triple[2])))
        return triples

    @classmethod
    def _build_graph(cls, options: Options) -> Dict[str, Any]:
        triples = cls._resolve_triples(options)

        entities: List[str] = []
        sentences: Dict[str, List[str]] = {}
        for subject, predicate, obj in triples:
            sentence = f"{subject} {predicate} {obj}"
            # dict.fromkeys: a self-loop contributes its sentence once, not twice.
            for entity in dict.fromkeys((subject, obj)):
                if entity not in sentences:
                    entities.append(entity)
                    sentences[entity] = []
                if sentence not in sentences[entity]:
                    sentences[entity].append(sentence)

        nodes = [{"doc_id": entity, "text": " ".join([entity] + sentences[entity])} for entity in entities]

        edges: List[List[str]] = []
        seen_pairs: Set[Tuple[str, str]] = set()
        for subject, _, obj in triples:
            if subject == obj or (subject, obj) in seen_pairs:
                continue
            seen_pairs.add((subject, obj))
            edges.append([subject, obj])

        return {cls.NODES_KEY: nodes, cls.EDGES_KEY: edges}
