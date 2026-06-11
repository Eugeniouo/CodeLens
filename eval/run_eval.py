"""Скрипт оценки качества"""

import json
import argparse #cдля обработки командной строки
from pathlib import Path
import sys

# python eval/run_eval.py
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.metrics import Question, Prediction, evaluate

# загрузка вопросов
def load_questions(path: str) -> list[Question]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        Question(
            question_id=q["question_id"],
            query=q["query"],
            correct_chunk_ids=q["correct_chunk_ids"]
        )
        for q in data
    ]

# загрузка предсказания модели (results.json)
def load_predictions(path: str) -> list[Prediction]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        Prediction(
            question_id=p["question_id"],
            top_5_chunks=p["top_5_chunks"]
        )
        for p in data
    ]


def main():
    # description - текст, который покажется при вызове в консоли команды --help
    parser = argparse.ArgumentParser(description="Оценка качества поиска CodeLens")
    parser.add_argument(
        "--questions", 
        default="data/eval_questions.json",
        help="Путь к файлу с эталонными вопросами"
    )
    parser.add_argument(
        "--predictions", 
        default="results.json", 
        help="Путь к файлу с предсказаниями (results.json)"
    )
    args = parser.parse_args()
    # т.е. при запуске, если пользователь напишет python eval/run_eval.py, то скрипт увидит, что польз. ничего не указал и
    # по умолчанию возьмёт data/eval_questions.json и results.json
    print(f"DEBUG: Questions path: {args.questions}")
    print(f"DEBUG: Predictions path: {args.predictions}")
    print(f"DEBUG: Questions file exists: {Path(args.questions).exists()}")
    print(f"DEBUG: Predictions file exists: {Path(args.predictions).exists()}")

    if not Path(args.questions).exists():
        print(f"Ошибка: файл вопросов не найден по пути {args.questions}")
        sys.exit(1)
    if not Path(args.predictions).exists():
        print(f"Ошибка: файл предсказаний не найден по пути {args.predictions}")
        sys.exit(1)

    questions = load_questions(args.questions)
    predictions = load_predictions(args.predictions)

    metrics = evaluate(predictions, questions)

    print("\nCodeLens evaluation")
    print(f"Questions evaluated: {len(questions)}")
    print(f"Precision@5: {metrics['precision_at_5']:.4f}")
    print(f"MRR: {metrics['mrr']:.4f}")

if __name__ == "__main__":
    main()