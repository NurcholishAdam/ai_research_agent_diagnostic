# -*- coding: utf-8 -*-
"""
LIMIT-GRAPH Diagnostic Module: Long-Context Robustness
Evaluates multilingual agents' ability to handle long contexts with various challenges
"""

from .evaluate_similarity import NeedleQuestionSimilarityEvaluator
from .inject_distractors import DistractorInjectionFramework
from .compute_needle_haystack_similarity import NeedleHaystackSemanticAlignment
from .shuffle_haystack import HaystackStructureSensitivity
from .repeat_task import RepeatedWordsReplicationTask
from .diagnostic_dashboard import DiagnosticDashboard

__all__ = [
    'NeedleQuestionSimilarityEvaluator',
    'DistractorInjectionFramework',
    'NeedleHaystackSemanticAlignment',
    'HaystackStructureSensitivity',
    'RepeatedWordsReplicationTask',
    'DiagnosticDashboard'
]

__version__ = "1.0.0"
__description__ = "Long-context robustness evaluation for multilingual agents"
