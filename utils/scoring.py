from typing import Dict, Any


def avg_score(scores: Dict[str, Any]) -> float:
    vals = [v for v in scores.values() if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else 0.0