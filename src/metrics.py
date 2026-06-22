"""
Модуль подсчёта метрик качества поиска.

Реализует стандартные метрики ранжирования: Precision@K и MRR,
утилиты для построения сводной таблицы результатов поиска.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

@dataclass
class Question:
    question_id: str
    query: str
    correct_chunk_ids: list[str]

@dataclass
class Prediction:
    question_id: str
    top_5_chunks: list[str]

def _normalize_chunk_id(chunk_id: str) -> str:
    return chunk_id.strip().replace("\\", "/")

def _chunks_match(a: str, b: str, line_tolerance: int = 2) -> bool:
    a = _normalize_chunk_id(a)
    b = _normalize_chunk_id(b)
    if a == b:
        return True

    a_parts = a.rsplit(":", 2)
    b_parts = b.rsplit(":", 2)
    
    if len(a_parts) != 3 or len(b_parts) != 3:
        return False

    a_path, a_name, a_line = a_parts
    b_path, b_name, b_line = b_parts

    if a_name != b_name:
        return False

    if not (a_path.endswith(b_path) or b_path.endswith(a_path)):
        return False

    try:
        return abs(int(a_line) - int(b_line)) <= line_tolerance
    except ValueError:
        return False

def precision_at_5(
    predictions: Sequence[Prediction],
    questions: Sequence[Question],
) -> float:
    if not predictions or not questions:
        return 0.0

    questions_map = {q.question_id: q for q in questions}
    scores: list[float] = []

    for pred in predictions:
        q = questions_map.get(pred.question_id)
        if q is None:
            continue

        top5 = [_normalize_chunk_id(c) for c in pred.top_5_chunks[:5]]
        matched = sum(
            1
            for correct in q.correct_chunk_ids
            if any(_chunks_match(correct, c) for c in top5)
        )
        denom = min(5, len(q.correct_chunk_ids))
        scores.append(matched / denom if denom > 0 else 0.0)

    return sum(scores) / len(scores) if scores else 0.0

def mean_reciprocal_rank(
    predictions: Sequence[Prediction],
    questions: Sequence[Question],
) -> float:
    if not predictions or not questions:
        return 0.0

    questions_map = {q.question_id: q for q in questions}
    rr_scores: list[float] = []

    for pred in predictions:
        q = questions_map.get(pred.question_id)
        if q is None:
            continue

        top5 = [_normalize_chunk_id(c) for c in pred.top_5_chunks[:5]]
        rr = 0.0
        for rank, chunk in enumerate(top5, start=1):
            if any(_chunks_match(correct, chunk) for correct in q.correct_chunk_ids):
                rr = 1.0 / rank
                break
        rr_scores.append(rr)

    return sum(rr_scores) / len(rr_scores) if rr_scores else 0.0

def evaluate(
    predictions: Sequence[Prediction],
    questions: Sequence[Question],
) -> dict[str, float]:
    return {
        "precision_at_5": precision_at_5(predictions, questions),
        "mrr": mean_reciprocal_rank(predictions, questions),
    }