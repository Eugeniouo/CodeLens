import importlib  # Добавляем для жесткой перезагрузки модулей
import json
import time
from pathlib import Path

import streamlit as st

import src.searcher  # Импортируем сам модуль для возможности его reload
from src.embedder import load_model
from src.indexer import COLLECTION_NAME, DB_PATH, index_chunks
from src.metrics import Prediction, Question, evaluate
from src.parser import parse_directory
from src.searcher import init_searcher, search

# Настройка конфигурации страницы Streamlit
st.set_page_config(page_title="CodeLens — Поиск по коду", page_icon="🔎", layout="wide")


# Инициализация единого состояния страницы
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "🏠 Главная"


# UI-дизайн: стиль Ростелеком (чистые белые плитки, оранжевые акценты, без лишних рамок)
st.markdown(
    """
<style>

    /* Импорт красивого шрифта Inter */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght=400;500;600;700;900&display=swap');

    /* Глобальный шрифт для абсолютно всех элементов на странице */
    html, body, [data-testid="stAppViewContainer"], .stApp, * {
        font-family: 'Inter', sans-serif !important;
    }

    /* Сочный фирменный фиолетово-синий фон приложения */
    html body .stApp {
        background-color: #b46dfe !important;
    }


    /* Вырезаем служебные надписи */

    html body header,
    html body [data-testid="stHeader"],
    html body [data-testid="stDecoration"],
    html body .stDeployButton,
    html body #MainMenu {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        height: 0 !important;
        width: 0 !important;
    }

    /* Дополнительно глушим любые дочерние элементы в шапке (тексты, индикаторы) */
    html body [data-testid="stHeader"] * {
        display: none !important;
    }

    /* Прячем технические ссылки-якоря (#) у заголовков */
    h1 a, h2 a, h3 a, h4 a, h5 a, h6 a {
        display: none !important;
    }

    /* Стилизация заголовков */
    h1, h2, h3, h4 {
        color: #1D2053 !important;
        font-weight: 800;
    }
    /* Отдельный класс для белого подзаголовка, который перебьет фиолетовый цвет за счет специфичности */
    h3.rt-white-text {
        color: #FFFFFF !important;
    }
    /* Крупные фоновые блоки БЕЗ КОНТУРА  */
    .rt-hero-block {
        background-color: #FFFFFF !important;
        padding: 3.5rem 2rem;
        border-radius: 24px;
        text-align: center;
        margin-bottom: 2.5rem;
        box-shadow: 0 20px 40px rgba(29, 32, 83, 0.15);
        border: none !important;
    }

    .rt-subpage-header {
        background-color: #FFFFFF !important;
        padding: 1.8rem 2.5rem;
        border-radius: 20px;
        margin-top: 1rem;
        margin-bottom: 2.5rem;
        box-shadow: 0 15px 35px rgba(29, 32, 83, 0.12);
        border: none !important;
    }

    /* Контрастный темный терминал внутри главного блока */
    .terminal-line {
        background: #110C35 !important;
        padding: 12px 26px;
        border-radius: 12px;
        font-family: 'Courier New', monospace !important;
        font-size: 0.95rem;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-top: 1.5rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.2);
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    .terminal-line span, .terminal-line p, .terminal-line div {
        color: #FFFFFF !important;
        font-family: 'Courier New', monospace !important;
    }
    .terminal-line .terminal-prompt {
        color: #FF5A00 !important;
        font-weight: bold;
        margin-right: 8px;
    }

    /* Эффект анимации печатающегося текста в терминале */
    .typing-effect {
        display: inline-block;
        overflow: hidden;
        white-space: nowrap;
        border-right: 2px solid #FF5A00;
        width: 0;
        animation: typing 3.5s steps(49, end) forwards, blink 0.75s step-end infinite;
    }
    @keyframes typing {
        from { width: 0; }
        to { width: 49ch; }
    }
    @keyframes blink {
        from, to { border-color: transparent; }
        50% { border-color: #FF5A00; }
    }


    /* Редизайн всех кнопок на сайте */

    div.stButton > button {
        background-color: #FFFFFF !important;
        color: #1D2053 !important;
        border: 2px solid #FFFFFF !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 0.6rem 2rem !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100%;
    }

    /* Эффект наведения для ВСЕХ кнопок: становятся ярко-оранжевыми */
    div.stButton > button:hover {
        background-color: #FF5A00 !important;
        border-color: #FF5A00 !important;
        color: #FFFFFF !important;
        box-shadow: 0 10px 25px rgba(255, 90, 0, 0.35) !important;
        transform: translateY(-2px);
    }
    div.stButton > button:hover * {
        color: #FFFFFF !important;
    }

    /* Специфичные настройки для больших плиток-карточек на Главной */
    .rt-card-button > div > button {
        min-height: 140px !important;
        font-size: 1.1rem !important;
        line-height: 1.4 !important;
        white-space: pre-line !important;
        border-radius: 16px !important;
    }

    /* Системная кнопка "Назад на главную" */
    div.stButton > button[id*="back"], div.stButton > button:contains("Вернуться") {
        border: 2px solid #FFFFFF !important;
    }

    /* Красим белые блоки */

    /* 1. Цепляем внешний контейнер, если он доступен в текущей версии */
    html body .stApp div[data-testid="stVerticalBlockBorderWrapper"]:has(.rt-card-marker) {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
        border: 2px solid #FF5A00 !important;
        border-radius: 20px !important;
        padding: 1.8rem !important;
        box-shadow: 0 12px 36px rgba(0, 0, 0, 0.15) !important;
    }

    /* Прячем дефолтный фон внутреннего контейнера, чтобы не перекрывал белый цвет */
    html body .stApp div[data-testid="stVerticalBlockBorderWrapper"]:has(.rt-card-marker) > div[data-testid="stVerticalBlock"] {
        background-color: transparent !important;
        background: transparent !important;
    }

    /* 2. Запасной вариант: если разметка плоская, красим сам вертикальный блок */
    html body .stApp div[data-testid="stVerticalBlock"]:has(.rt-card-marker) {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
        border: 2px solid #FF5A00 !important;
        border-radius: 20px !important;
        padding: 1.8rem !important;
        box-shadow: 0 12px 36px rgba(0, 0, 0, 0.15) !important;
    }

    /* Сброс конфликтов: если сработали оба селектора, убираем внутреннее дублирование */
    html body .stApp div[data-testid="stVerticalBlockBorderWrapper"]:has(.rt-card-marker) div[data-testid="stVerticalBlock"]:has(.rt-card-marker) {
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        background: transparent !important;
    }

    /* Высокоточный перекрас текстов внутри созданных белых карточек */
    html body .stApp div:has(.rt-card-marker) p,
    html body .stApp div:has(.rt-card-marker) label,
    html body .stApp div:has(.rt-card-marker) span:not([class*="Token"]):not([class*="token"]),
    html body .stApp div:has(.rt-card-marker) h1,
    html body .stApp div:has(.rt-card-marker) h2,
    html body .stApp div:has(.rt-card-marker) h3,
    html body .stApp div:has(.rt-card-marker) h4 {
        color: #1D2053 !important;
        font-weight: 600 !important;
    }

    /* Красивые поля ввода (Inputs) внутри белых карточек */
    div[data-testid="stTextInput"] input {
        background-color: #F4F5F7 !important;
        color: #1D2053 !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
    }

    /* Фикс встроенных алертов (st.info, st.warning), чтобы они не сливались внутри белых блоков */
    div[data-testid="stAlert"] {
        background-color: #F0EDFF !important;
        border-left: 5px solid #7A5CFF !important;
        border-radius: 8px;
    }
    div[data-testid="stAlert"] * {
        color: #7A5CFF !important;
    }

    /* Отключаем глобальное перекрашивание для окон вывода кода, чтобы сохранить подсветку синтаксиса */
    div:has(.rt-card-marker) pre,
    div:has(.rt-card-marker) code,
    div:has(.rt-card-marker) pre span {
        color: inherit !important;
        font-family: monospace !important;
        font-weight: normal !important;
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


# Получаем статус бэкенда
backend_ready, backend_error = warmup_backend()


# Страница 0: Главная страница
if st.session_state["current_page"] == "🏠 Главная":
    st.markdown(
        """
    <div class="rt-hero-block">
        <h1 style="font-size: 4.5rem; font-weight: 900; margin: 0; letter-spacing: -1.5px; color: #1D2053 !important;">CodeLens</h1>
        <p style="color: #51567A; font-size: 1.25rem; margin-top: 0.6rem; font-weight: 600;">Интеллектуальный поиск и аудит кодовой базы RAG-системы</p>
        <div class="terminal-line">
            <span class="terminal-prompt">codelens-bot:~$</span>
            <span class="typing-effect">Анализ абстрактных деревьев AST и векторизация...</span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown(
        "<h3 class='rt-white-text' style='text-align: center; margin-bottom: 2.5rem; font-size: 1.4rem;'>Выберите необходимый модуль системы:</h3>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown('<div class="rt-card-button">', unsafe_allow_html=True)
        if st.button(
            "⌕ Семантический поиск\n\nПоиск по логике на естественном языке",
            use_container_width=True,
        ):
            st.session_state["current_page"] = "⌕ Семантический поиск"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="rt-card-button">', unsafe_allow_html=True)
        if st.button(
            "⏳ Расчет метрик\n\nПрогон валидации и расчет Precision@5",
            use_container_width=True,
        ):
            st.session_state["current_page"] = "⏳ Живой расчет метрик"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="rt-card-button">', unsafe_allow_html=True)
        if st.button(
            "⚙ Индексация базы\n\nПарсинг проекта и сборка ChromaDB",
            use_container_width=True,
        ):
            st.session_state["current_page"] = "⚙️ Индексация базы"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


# Страница 1: Семантический поиск
elif st.session_state["current_page"] == "⌕ Семантический поиск":
    if st.button("← Вернуться на главную", key="back_search"):
        st.session_state["current_page"] = "🏠 Главная"
        st.rerun()

    st.markdown(
        """
    <div class="rt-subpage-header">
        <h2 style="color: #1D2053 !important; margin:0; font-size: 2.3rem; font-weight: 900;">⌕ Семантический поиск по коду</h2>
        <p style="color: #51567A; margin: 0.4rem 0 0 0; font-weight: 600; font-size: 1.1rem;">Введите поисковый запрос на естественном языке для локализации бизнес-логики.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown('<div class="rt-card-marker"></div>', unsafe_allow_html=True)
        if not backend_ready:
            st.warning(
                "⚠️ **Векторная база данных еще не инициализирована.** Коллекция `code_chunks` не найдена. Пожалуйста, перейдите во вкладку Индексации."
            )
        else:
            col1, col2 = st.columns([5, 1])
            with col1:
                query = st.text_input(
                    "Что вы хотите найти?",
                    placeholder="например: как в проекте создаётся токен доступа и какой срок его жизни?",
                )
            with col2:
                st.write("")
                st.write("")
                search_button = st.button("Найти", use_container_width=True)

            use_llm = st.checkbox(
                "Включить генеративный ответ ИИ (LLM-режим)", value=False
            )

            if search_button and query:
                with st.spinner("Поиск по..."):
                    results, latency = search(query, top_k=5)

                st.caption(
                    f"⏱️ Время ответа системы: {latency:.4f} сек. (Лимит ТЗ: <= 3 сек.)"
                )

                st.write("")
                st.markdown(f"#### Результаты поиска (Топ-{len(results)})")

                if not results:
                    st.info("Ничего не найдено.")

                for item in results:
                    with st.container(border=True):
                        st.markdown(
                            '<div class="rt-card-marker"></div>', unsafe_allow_html=True
                        )
                        rank = item.get("rank", 1)
                        score = item.get("score", 0.0)
                        chunk_id = item.get("id", "unknown")
                        file_path = item.get("file_path", "unknown")
                        name = item.get("name", "unknown")
                        source_code = item.get("source_code", "")

                        # FIX: Все элементы вывода теперь строго внутри контейнера
                        c1, c2 = st.columns([4, 1])
                        with c1:
                            st.markdown(
                                f"**{rank}. Файл:** `{file_path}` | **Объект:** `{name}`"
                            )
                            st.caption(f"ID чанка: `{chunk_id}`")
                        with c2:
                            st.markdown(
                                f"<p style='text-align: right; color: #FF5A00; font-weight: 800; font-size: 1.2rem; margin:0;'>Score: {score * 100:.1f}%</p>",
                                unsafe_allow_html=True,
                            )

                        st.code(source_code, language="python")


# Страница 2: живой расчет метрик
elif st.session_state["current_page"] == "⏳ Живой расчет метрик":
    if st.button("← Вернуться на главную", key="back_metrics"):
        st.session_state["current_page"] = "🏠 Главная"
        st.rerun()

    st.markdown(
        """
    <div class="rt-subpage-header">
        <h2 style="color: #1D2053 !important; margin:0; font-size: 2.3rem; font-weight: 900;">⏳ Валидация системы и расчет метрик</h2>
        <p style="color: #51567A; margin: 0.4rem 0 0 0; font-weight: 600; font-size: 1.1rem;">Запуск автоматического тестирования RAG-решения на валидационном датасете.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown('<div class="rt-card-marker"></div>', unsafe_allow_html=True)
        if not backend_ready:
            st.warning(
                "⚠️ **Расчет метрик невозможен.** Векторная база данных пуста. Сначала проиндексируйте проект."
            )
        else:
            dataset_path = Path("data/eval_questions.json")
            if not dataset_path.exists():
                dataset_path = Path("eval_questions.json")

            if not dataset_path.exists():
                st.error("Файл датасета `eval_questions.json` не найден в проекте.")
            else:
                with open(dataset_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)

                st.info(f"Найдено тестовых вопросов в датасете: **{len(raw_data)}**")

                if st.button("Запустить валидацию и рассчитать Precision@5"):
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
                        status_text.text(
                            f"Тестирование [{i + 1}/{len(questions)}]: {q.query[:60]}..."
                        )
                        results, latency = search(q.query, top_k=5)
                        total_latency += latency
                        top_5_ids = [res["id"] for res in results]
                        predictions.append(
                            Prediction(
                                question_id=q.question_id, top_5_chunks=top_5_ids
                            )
                        )
                        progress_bar.progress((i + 1) / len(questions))

                    status_text.success("Тестирование завершено!")

                    metrics_dict = evaluate(predictions, questions)
                    precision_5 = metrics_dict.get("precision_at_5", 0.0)
                    mrr_score = metrics_dict.get("mrr", 0.0)
                    avg_latency = total_latency / len(questions)

                    with st.container(border=True):
                        st.markdown(
                            '<div class="rt-card-marker"></div>', unsafe_allow_html=True
                        )
                        st.markdown("### Итоговые результаты валидации")
                        m1, m2, m3 = st.columns(3)
                        with m1:
                            st.metric(
                                label="Итоговый Precision@5",
                                value=f"{precision_5 * 100:.2f}%",
                            )
                        with m2:
                            st.metric(
                                label="Метрика MRR",
                                value=f"{mrr_score:.4f}",
                            )
                        with m3:
                            st.metric(
                                label="Средний Latency",
                                value=f"{avg_latency:.4f} сек.",
                            )


# Страница 3: индексация базы
elif st.session_state["current_page"] == "⚙️ Индексация базы":
    if st.button("← Вернуться на главную", key="back_indexing"):
        st.session_state["current_page"] = "🏠 Главная"
        st.rerun()

    st.markdown(
        """
    <div class="rt-subpage-header">
        <h2 style="color: #1D2053 !important; margin:0; font-size: 2.3rem; font-weight: 900;">⚙️ Индексация и сборка векторной базы</h2>
        <p style="color: #51567A; margin: 0.4rem 0 0 0; font-weight: 600; font-size: 1.1rem;">Разбиение исходного кода Python-проектов на логические чанки (AST) и векторизация в ChromaDB.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown('<div class="rt-card-marker"></div>', unsafe_allow_html=True)
        if "indexing_result" in st.session_state:
            res = st.session_state["indexing_result"]
            st.balloons()
            st.success(
                f"✔ Система успешно проиндексировала {res['chunks']} чанков за {res['time']:.2f} сек.!"
            )
            del st.session_state["indexing_result"]

        repo_path = st.text_input(
            "Путь к директории проекта:",
            value="data/codebase_python",
        )
        reset_db = st.checkbox(
            "Полностью очистить старую базу перед индексацией (Reset)", value=True
        )

        if st.button("Начать индексацию"):
            if not Path(repo_path).exists() or not Path(repo_path).is_dir():
                st.error(f"Ошибка: Директория '{repo_path}' не найдена.")
            else:
                indexing_successful = False
                num_chunks = 0
                total_time = 0.0
                start_time = time.perf_counter()

                with st.status(
                    "Выполнение процессов на бэкенде...", expanded=True
                ) as status:
                    # Шаг 1
                    status.update(label="Шаг 1/3: Парсинг структуры проекта и AST...")
                    chunks = parse_directory(str(repo_path))

                    if not chunks:
                        status.update(
                            label="Индексация прервана: чанки не найдены", state="error"
                        )
                    else:
                        # Шаг 2
                        status.update(
                            label=f"Шаг 2/3: Загрузка модели (Выделено чанков: {len(chunks)})..."
                        )
                        model = load_model()

                        # Шаг 3
                        status.update(
                            label="Шаг 3/3: Генерация эмбеддингов и запись в ChromaDB..."
                        )
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

                            # Финал: Используем чистый state="complete", Streamlit сам нарисует красивую галочку
                            status.update(
                                label=f"Успешно завершено! Проиндексировано чанков: {num_chunks}",
                                state="complete",
                            )
                            indexing_successful = True

                        except Exception as e:
                            status.update(
                                label="💥 Ошибка при записи в ChromaDB", state="error"
                            )
                            st.error(f"Произошла ошибка: {e}")

                if indexing_successful:
                    st.session_state["indexing_result"] = {
                        "chunks": num_chunks,
                        "time": total_time,
                    }
                    importlib.reload(src.searcher)
                    st.cache_resource.clear()
                    st.rerun()
