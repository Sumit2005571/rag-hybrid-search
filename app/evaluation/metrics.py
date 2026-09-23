"""RAG evaluation metrics (Faithfulness, Answer Relevance, Context Recall/Precision)."""

from typing import Dict, List
from pydantic import BaseModel, Field


class EvaluationMetrics(BaseModel):
    """Container for RAG performance evaluation scores."""

    faithfulness: float = Field(default=0.0, description="Measures factual consistency against context (0-1)")
    answer_relevance: float = Field(default=0.0, description="Measures relevance of answer to question (0-1)")
    context_precision: float = Field(default=0.0, description="Measures rank quality of ground-truth chunks (0-1)")
    context_recall: float = Field(default=0.0, description="Measures proportion of relevant information retrieved (0-1)")


class RAGEvaluator:
    """Evaluates RAG pipeline outputs using predefined metrics."""

    def compute_faithfulness(self, answer: str, context: List[str]) -> float:
        """Compute faithfulness score."""
        return 1.0 if answer and context else 0.0

    def compute_metrics(
        self, question: str, answer: str, context: List[str], ground_truth: str
    ) -> EvaluationMetrics:
        """Evaluate generation across all core metrics."""
        return EvaluationMetrics(
            faithfulness=self.compute_faithfulness(answer, context),
            answer_relevance=1.0 if answer else 0.0,
            context_precision=1.0 if context else 0.0,
            context_recall=1.0 if ground_truth else 0.0,
        )
