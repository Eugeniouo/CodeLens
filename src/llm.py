"""Модуль для подключения AI-агента к ответу на запрос."""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY", os.getenv("API_KEY")), 
    base_url="https://api.groq.com/openai/v1"
)

SYSTEM_PROMPT = """Твоя задача - ответить на технический вопрос пользователя, опираясь ТОЛЬКО на предоставленные фрагменты кодовой базы.
Для каждого утверждения указывай файл и название сущности, откуда взята информация.
Если в предоставленном контексте нет ответа на вопрос, прямо скажи: "В найденных фрагментах кода нет информации для точного ответа".
Категорически запрещено придумывать реализацию, классы или переменные, которых нет в контексте.
Отвечай структурировано и используй Markdown для форматирования кода."""

def stream_answer(query: str, retrieved_chunks: list[dict]):
    """
    Генерирует ответ LLM в потоковом режиме
    """
    if not retrieved_chunks:
        yield "Не найдено релевантных фрагментов кода для ответа."
        return

    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        file_path = chunk.get("file_path", "unknown")
        name = chunk.get("name", "unknown")
        code = chunk.get("source_code", "")
        context_parts.append(f"--- Фрагмент {i} ---\nФайл: {file_path}\nСущность: {name}\nКод:\n{code}\n")

    context_text = "\n".join(context_parts)
    user_prompt = f"Вопрос пользователя: {query}\n\nКонтекст кодовой базы:\n{context_text}"

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            stream=True 
        )
        
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
         yield f"Произошла ошибка при обращении к LLM: {e}"