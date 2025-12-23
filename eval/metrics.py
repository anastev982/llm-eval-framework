from typing import Tuple
from utils.helpers import normalize_text

def accuracy(preds, labels):
    correct = sum(p == l for p, l in zip(preds, labels))
    return correct / len(labels) if labels else 0.0

def token_f1(pred, ref):
    """Simplified token-level F1 for summarization/extraction tasks."""
    pred_tokens = set(pred.lower().split())
    ref_tokens = set(ref.lower().split())

    if not pred_tokens or not ref_tokens:
        return 0.0

    tp = len(pred_tokens & ref_tokens)
    precision = tp / len(pred_tokens)
    recall = tp / len(ref_tokens)
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

def rouge1(pred: str, ref: str) -> Tuple[float, float, float]:
    """
    Returns: (precision, recall, f1)
    ROUGE-1 = unigram overlap.
    """
    ref_norm = normalize_text(ref)
    pred_norm = normalize_text(pred)

    ref_words = ref_norm.split()
    pred_words = pred_norm.split()

    if not ref_words or not pred_words:
        return 0.0, 0.0, 0.0

    ref_set = set(ref_words)
    pred_set = set(pred_words)

    overlap = ref_set & pred_set
    overlap_count = len(overlap)

    recall = overlap_count / len(ref_set)
    precision = overlap_count / len(pred_set)

    if recall + precision == 0:
        f1 = 0.0
    else:
        f1 = 2 * recall * precision / (recall + precision)

    return precision, recall, f1