import importlib
import json
import time
from pathlib import Path

import streamlit as st

import src.searcher
from src.embedder import load_model
from src.indexer import COLLECTION_NAME, DB_PATH, index_chunks
from src.metrics import Prediction, Question, evaluate
from src.parser import parse_directory
from src.searcher import init_searcher, search

st.set_page_config(page_title="CodeLens | Внутренний портал", page_icon="🔍", layout="wide")

if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Главная"

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght=400;500;600;700&display=swap');

    html, body, [data-testid="stAppViewContainer"], .stApp, * {
        font-family: 'Inter', sans-serif !important;
    }

    /* Строгий светло-серый фон приложения */
    html body .stApp {
        background-color: #F4F5F7 !important;
    }

    /* Скрытие служебных элементов Streamlit */
    header, [data-testid="stHeader"], [data-testid="stDecoration"], .stDeployButton, #MainMenu {
        display: none !important;
        height: 0 !important;
    }

    /* Типографика */
    h1, h2, h3, h4 {
        color: #101828 !important;
        font-weight: 600;
    }

    /* Универсальная белая карточка (B2B стиль) */
    .rt-card {
        background-color: #FFFFFF !important;
        border: 1px solid #E4E7EC !important;
        border-radius: 8px !important;
        padding: 32px;
        margin-bottom: 24px;
        box-shadow: 0 1px 3px rgba(16, 24, 40, 0.05) !important;
    }

    /* Заголовки страниц */
    .page-header {
        margin-bottom: 24px;
        padding-bottom: 16px;
        border-bottom: 1px solid #E4E7EC;
    }
    .page-header h1 {
        font-size: 28px;
        margin-bottom: 8px;
    }
    .page-header p {
        color: #475467;
        font-size: 15px;
        margin: 0;
    }

    /* Базовые кнопки */
    div.stButton > button {
        background-color: #FFFFFF !important;
        color: #344054 !important;
        border: 1px solid #D0D5DD !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        padding: 0.5rem 1rem !important;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.05) !important;
        transition: all 0.2s ease-in-out !important;
        width: 100%;
    }

    /* Акцентное наведение (Корпоративный оранжевый) */
    div.stButton > button:hover {
        border-color: #FF4F12 !important;
        color: #FF4F12 !important;
        background-color: #FFF9F6 !important;
    }
    
    /* Специфичные настройки для больших кнопок-плиток на Главной */
    .menu-tile div.stButton > button {
        min-height: 120px !important;
        text-align: left !important;
        white-space: pre-wrap !important;
        justify-content: flex-start !important;
        padding: 24px !important;
        font-size: 1.05rem !important;
        line-height: 1.5 !important;
    }

    /* Поля ввода */
    div[data-testid="stTextInput"] input {
        background-color: #FFFFFF !important;
        color: #101828 !important;
        border: 1px solid #D0D5DD !important;
        border-radius: 6px !important;
        padding: 10px 14px !important;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.05) !important;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #FF4F12 !important;
        box-shadow: 0 0 0 4px rgba(255, 79, 18, 0.1) !important;
    }

    /* Стилизация алертов */
    div[data-testid="stAlert"] {
        background-color: #F9FAFB !important;
        border: 1px solid #E4E7EC !important;
        border-left: 4px solid #FF4F12 !important;
        border-radius: 6px;
        color: #344054 !important;
    }

    /* Очистка блоков кода (чтобы не перебивалась подсветка) */
    pre, code {
        font-family: 'JetBrains Mono', 'Courier New', monospace !important;
        font-size: 0.9rem !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Прогрев модели и БД
@st.cache_resource
def warmup_backend():
    """Загружает модель и БД в память."""
    try:
        init_searcher()
        search("warmup query", top_k=1)
        return True, None
    except Exception as e:
        return False, str(e)


backend_ready, backend_error = warmup_backend()


if st.session_state["current_page"] == "Главная":
    st.markdown(
        """
        <div class="rt-card" style="text-align: center; padding: 48px 24px;">
            <h1 style="font-size: 3rem; margin-bottom: 12px; color: #101828;">Платформа CodeLens</h1>
            <p style="color: #475467; font-size: 1.1rem; max-width: 600px; margin: 0 auto;">
                Внутренний инструмент семантического поиска и архитектурного аудита кодовой базы
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<h3 style='margin-bottom: 20px; font-size: 1.2rem;'>Доступные модули:</h3>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown('<div class="menu-tile">', unsafe_allow_html=True)
        if st.button("🔍 Семантический поиск\nПоиск фрагментов кода по текстовому описанию логики", use_container_width=True):
            st.session_state["current_page"] = "Поиск"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="menu-tile">', unsafe_allow_html=True)
        if st.button("📊 Валидация системы\nРасчет метрик качества RAG (Precision@5, MRR)", use_container_width=True):
            st.session_state["current_page"] = "Метрики"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="menu-tile">', unsafe_allow_html=True)
        if st.button("⚙️ Индексация базы\nПарсинг проекта и обновление векторной БД (ChromaDB)", use_container_width=True):
            st.session_state["current_page"] = "Индексация"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


elif st.session_state["current_page"] == "Поиск":
    if st.button("← На главную", key="back_search"):
        st.session_state["current_page"] = "Главная"
        st.rerun()

    st.markdown(
        """
        <div class="page-header">
            <h1>Семантический поиск</h1>
            <p>Локализация бизнес-логики в исходном коде по запросам на естественном языке.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not backend_ready:
        st.error("Векторная база данных не инициализирована. Пожалуйста, выполните индексацию кодовой базы.")
    else:
        col1, col2 = st.columns([5, 1], gap="medium")
        with col1:
            query = st.text_input("Поисковый запрос:", placeholder="Например: реализация генерации JWT токена и сроки его жизни")
        with col2:
            st.write("")
            st.write("")
            search_button = st.button("Выполнить поиск", use_container_width=True)

        use_llm = st.checkbox("Запросить аналитический ответ LLM", value=True)

        if search_button and query:
            with st.spinner("Выполнение векторного поиска..."):
                results, latency = search(query, top_k=5)

            st.caption(f"Время выполнения запроса: {latency:.3f} сек.")
            st.divider()

            if use_llm:
                st.markdown("### Анализ кодовой базы (LLM-ассистент)")
                
                from src.llm import stream_answer
                
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(f"**Анализ по запросу:** *«{query}»*")
                    
                    if results:
                        stream_generator = stream_answer(query, results)
                        st.write_stream(stream_generator) 
                    else:
                        st.write("К сожалению, релевантные фрагменты кода для анализа не найдены.")
                st.divider()

            st.markdown(f"### Найденные фрагменты (Топ-{len(results)})")

            if not results:
                st.info("По вашему запросу ничего не найдено.")

            for item in results:
                with st.container(border=True):
                    rank = item.get("rank", 1)
                    score = item.get("score", 0.0)
                    chunk_id = item.get("id", "unknown")
                    file_path = item.get("file_path", "unknown")
                    name = item.get("name", "unknown")
                    source_code = item.get("source_code", "")

                    c1, c2 = st.columns([4, 1])
                    with c1:
                        st.markdown(f"**{rank}. `{file_path}`** →  `{name}`")
                        st.caption(f"ID фрагмента: {chunk_id}")
                    with c2:
                        st.markdown(
                            f"<div style='text-align: right; color: #475467; font-weight: 600;'>Совпадение: {score * 100:.1f}%</div>",
                            unsafe_allow_html=True,
                        )

                    # Используем нативную подсветку синтаксиса
                    st.code(source_code, language="python")


# Навигация: Расчет метрик
elif st.session_state["current_page"] == "Метрики":
    if st.button("← На главную", key="back_metrics"):
        st.session_state["current_page"] = "Главная"
        st.rerun()

    st.markdown(
        """
        <div class="page-header">
            <h1>Валидация системы</h1>
            <p>Автоматическое тестирование RAG-алгоритма на размеченном датасете (eval_questions.json).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not backend_ready:
        st.error("Расчет метрик невозможен: векторная база данных пуста. Требуется индексация.")
    else:
        dataset_path = Path("data/eval_questions.json")
        if not dataset_path.exists():
            dataset_path = Path("eval_questions.json")

        if not dataset_path.exists():
            st.error("Файл валидационного датасета `eval_questions.json` не найден.")
        else:
            with open(dataset_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            st.info(f"Доступно тестовых сценариев: {len(raw_data)}")

            if st.button("Запустить тестирование"):
                questions = [
                    Question(
                        question_id=item["question_id"],
                        query=item["query"],
                        correct_chunk_ids=item["correct_chunk_ids"],
                    )
                    for item in raw_data
                ]

                progress_bar = st.progress(0)
                status_text = st.empty()

                predictions = []
                total_latency = 0.0

                for i, q in enumerate(questions):
                    status_text.text(f"Обработка [{i + 1}/{len(questions)}]: {q.query[:80]}...")
                    results, latency = search(q.query, top_k=5)
                    total_latency += latency
                    top_5_ids = [res["id"] for res in results]
                    predictions.append(
                        Prediction(question_id=q.question_id, top_5_chunks=top_5_ids)
                    )
                    progress_bar.progress((i + 1) / len(questions))

                status_text.success("Прогон валидационных тестов завершен.")

                metrics_dict = evaluate(predictions, questions)
                precision_5 = metrics_dict.get("precision_at_5", 0.0)
                mrr_score = metrics_dict.get("mrr", 0.0)
                avg_latency = total_latency / len(questions)

                st.markdown("### Итоговые метрики качества")
                m1, m2, m3 = st.columns(3)
                m1.metric(label="Precision@5", value=f"{precision_5 * 100:.1f}%")
                m2.metric(label="Mean Reciprocal Rank (MRR)", value=f"{mrr_score:.3f}")
                m3.metric(label="Средняя задержка (Latency)", value=f"{avg_latency:.3f} s")


# Навигация: Индексация базы
elif st.session_state["current_page"] == "Индексация":
    if st.button("← На главную", key="back_indexing"):
        st.session_state["current_page"] = "Главная"
        st.rerun()

    st.markdown(
        """
        <div class="page-header">
            <h1>Индексация кодовой базы</h1>
            <p>Анализ AST-деревьев Python-проектов и формирование векторного представления в ChromaDB.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "indexing_result" in st.session_state:
        res = st.session_state["indexing_result"]
        st.success(f"Индексация успешно завершена. Обработано фрагментов: {res['chunks']}. Затрачено времени: {res['time']:.2f} сек.")
        del st.session_state["indexing_result"]

    with st.container(border=True):
        repo_path = st.text_input("Абсолютный или относительный путь к директории проекта:", value="data/codebase_python")
        reset_db = st.checkbox("Принудительная очистка существующей БД (Reset ChromaDB)", value=True)

        if st.button("Запустить индексацию"):
            if not Path(repo_path).exists() or not Path(repo_path).is_dir():
                st.error(f"Указанная директория не найдена: '{repo_path}'")
            else:
                indexing_successful = False
                num_chunks = 0
                total_time = 0.0
                start_time = time.perf_counter()

                with st.status("Процесс индексации...", expanded=True) as status:
                    status.update(label="Парсинг структуры проекта и построение AST...")
                    chunks = parse_directory(str(repo_path))

                    if not chunks:
                        status.update(label="Ошибка: Пригодные для индексации фрагменты кода не найдены.", state="error")
                    else:
                        status.update(label=f"Загрузка модели эмбеддингов. Очередь чанков: {len(chunks)}...")
                        model = load_model()

                        status.update(label="Векторизация и запись данных в коллекцию...")
                        try:
                            index_chunks(
                                chunks=chunks,
                                db_path=DB_PATH,
                                collection_name=COLLECTION_NAME,
                                model=model,
                                batch_size=128,
                                reset=reset_db,
                            )
                            num_chunks = len(chunks)
                            total_time = time.perf_counter() - start_time

                            status.update(label=f"Индексация завершена. Записано чанков: {num_chunks}", state="complete")
                            indexing_successful = True

                        except Exception as e:
                            status.update(label="Сбой при записи в векторную БД.", state="error")
                            st.error(f"Системная ошибка: {e}")

                if indexing_successful:
                    st.session_state["indexing_result"] = {
                        "chunks": num_chunks,
                        "time": total_time,
                    }
                    importlib.reload(src.searcher)
                    st.cache_resource.clear()
                    st.rerun()