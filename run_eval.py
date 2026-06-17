"""Скрипт для оценки качества поиска на эталонном датасете."""

import json
from pathlib import Path

from src.metrics import evaluate, Prediction, Question
from src.searcher import search, init_searcher


def main() -> None:
    dataset_path = Path("data/eval_questions.json")
    
    if not dataset_path.exists():
        print(f"Файл {dataset_path} не найден!")
        return

    # Загружаем датасет
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Инициализируем объекты Question, игнорируя лишние поля
    questions = [
        Question(
            question_id=item["question_id"],
            query=item["query"],
            correct_chunk_ids=item["correct_chunk_ids"]
        )
        for item in raw_data
    ]

    print("=" * 60)
    print("Подготовка к тестированию...")
    
    # Warm up, вызываем init_searcher, чтобы загрузить модель в RAM
    # Делаем холостой запрос, чтобы инициализировать все графы вычислений
    init_searcher() 
    search("warmup query", top_k=1) 
    print("Модель и БД прогреты. Начинаем замеры на 'горячую'.\n")
    print("=" * 60)

    predictions = []
    total_latency = 0.0

    # Прогон вопросов
    for i, q in enumerate(questions, start=1):
        results, latency = search(q.query, top_k=5)
        total_latency += latency
        
        top_5_ids = [res["id"] for res in results]
        predictions.append(Prediction(question_id=q.question_id, top_5_chunks=top_5_ids))
        
        print(f"[{i:02d}/{len(questions)}] Вопрос: {q.query[:50]}... | {latency:.4f} сек.")

    # Подсчет метрик
    avg_latency = total_latency / len(questions)
    metrics_result = evaluate(predictions, questions)

    print("\n" + "=" * 60)
    print(" РЕЗУЛЬТАТЫ ОЦЕНКИ (EVALUATION)")
    print("=" * 60)
    print(f"Precision@5:  {metrics_result['precision_at_5'] * 100:.2f}%")
    print(f"MRR:          {metrics_result['mrr']:.4f}")
    print(f"Avg Latency:  {avg_latency:.4f} сек.")
    print("=" * 60)


if __name__ == "__main__":
    main()