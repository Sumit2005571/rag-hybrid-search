"""Evaluation runner executing test cases against golden datasets."""

import json
from pathlib import Path
from typing import Any, Dict, List, Union
from app.evaluation.metrics import EvaluationMetrics, RAGEvaluator


class EvaluationRunner:
    """Orchestrates golden Q&A dataset benchmark runs."""

    def __init__(self):
        self.evaluator = RAGEvaluator()

    def run_benchmark(self, dataset_path: Union[str, Path]) -> Dict[str, Any]:
        """Load benchmark dataset and compute aggregate metrics."""
        path = Path(dataset_path)
        if not path.exists():
            return {"error": f"Dataset file {dataset_path} not found", "items_evaluated": 0}

        try:
            records = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            return {"error": str(e), "items_evaluated": 0}

        results: List[Dict[str, Any]] = []
        for item in records:
            metrics = self.evaluator.compute_metrics(
                question=item.get("question", ""),
                answer=item.get("answer", ""),
                context=item.get("contexts", []),
                ground_truth=item.get("ground_truth", ""),
            )
            results.append({"question": item.get("question"), "metrics": metrics.model_dump()})

        return {
            "items_evaluated": len(results),
            "results": results,
        }
