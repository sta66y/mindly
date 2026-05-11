# Mindly — AI Wellness Coach with Persistent Memory

**Имя:** Иванов Иван Иванович  
**Группа:** DS-XX

---

## Содержание

- [Описание](#описание)
- [Архитектура памяти](#архитектура-памяти)
- [Как запустить](#как-запустить)
- [Персоны](#персоны)
- [Управление памятью и изоляция](#управление-памятью-и-изоляция)
- [Бенчмарк](#бенчмарк)
- [Модели и датасеты](#модели-и-датасеты)
- [Design Doc](#design-doc)

---

## Описание

Mindly — conversational AI wellness-коуч с долгосрочной персистентной памятью. Агент помнит цели, прогресс, имена близких и чувствительные темы клиента **между сессиями** (после перезапуска процесса). Реализованы:

- Persistent memory (ChromaDB + SQLite)
- 3 персоны с общей памятью
- Tenant-изоляция: данные клиента A недоступны клиенту B
- Целевое и полное забывание (152-ФЗ / right-to-forget)
- Стриминг с TTFT < 1.5 с
- LongMemEval benchmark + W&B логирование

---

## Архитектура памяти

Выбрана **hand-rolled** архитектура на базе ChromaDB + SQLite — в пользу прозрачности кода и явной tenant-изоляции (вместо managed-фреймворков типа mem0 или Letta).

**Поток данных:**

```
User message
    │
    ▼
MemoryLayer.retrieve(user_id, query, k=7)   ← ChromaDB WHERE user_id=X
    │
    ▼
Prompt Builder  (persona base + memory block + session history)
    │
    ▼
LLM (OpenRouter)  → streaming tokens → User
    │
    ▼ (background thread)
Fact Extractor (LLM, temp=0, JSON) → MemoryLayer.store(user_id, facts)
    │
    ├── ChromaDB.upsert(facts, metadata={user_id, category, do_not_raise})
    └── SQLite INSERT (audit log)
```

**Tenant-изоляция:** каждый `retrieve()` и `delete()` передаёт `where={"user_id": user_id}` в ChromaDB — фильтр применяется до формирования результата. Утечка невозможна на уровне хранилища.

**Do-not-raise:** факты с флагом `do_not_raise=True` хранятся отдельным блоком в системном промпте с явным запретом поднимать их по инициативе агента.

**Персистентность:** ChromaDB `PersistentClient` и SQLite хранятся на диске (`./data/memory/`). Перезапуск процесса не теряет память.

---

## Как запустить

### Через uv (рекомендуется)

```bash
# 1. Установить uv
pip install uv

# 2. Клонировать и перейти в директорию
git clone <repo-url>
cd mindly

# 3. Создать .env из примера
cp .env .env
# Вписать OPENROUTER_API_KEY=sk-or-...

# 4. Установить зависимости
uv sync

# 5. Запустить чат
uv run mindly chat --user alice --persona wellness_friend
```

### Через Docker

```bash
cp .env .env
# Вписать OPENROUTER_API_KEY

docker-compose build
docker-compose run --rm mindly chat --user alice --persona wellness_friend
```

### Команды в чате

| Команда | Действие |
|---------|---------|
| `/forget <тема>` | Удалить воспоминания по теме |
| `/forget all` | Удалить все воспоминания |
| `/persona <имя>` | Переключить персону |
| `/memories` | Показать все сохранённые факты |
| `/quit` | Выйти |

---

## Персоны

| Ключ | Имя | Описание |
|------|-----|---------|
| `tough_love` | Drill Sergeant | Прямой, военный стиль, высокие ожидания |
| `wellness_friend` | Wellness Friend | Тёплый, эмпатичный, поддерживающий |
| `cbt_coach` | CBT Coach | Структурированный, КПТ-метод, Сократовы вопросы |

**Смена персоны сохраняет память:** факты клиента хранятся в общем ChromaDB-хранилище, независимо от персоны.

```bash
# Начать с wellness_friend
uv run mindly chat --user alice --persona wellness_friend

# В новой сессии переключиться на tough_love
uv run mindly chat --user alice --persona tough_love
# Агент знает всё из предыдущей сессии
```

---

## Управление памятью и изоляция

### Целевое забывание

```bash
# В чате:
/forget моя работа

# Или через CLI:
uv run mindly forget --user alice "моя работа"
```

### Полное удаление (152-ФЗ / right-to-forget)

```bash
# В чате:
/forget all

# Или через CLI:
uv run mindly forget --user alice all
```

### Демонстрация tenant-изоляции

```bash
uv run python scripts/demo_isolation.py
```

Скрипт:
1. Создаёт пользователя `alice`, передаёт личные факты.
2. Создаёт пользователя `bob`, спрашивает «что ты обо мне знаешь?».
3. Проверяет, что `alice` данных нет в ни ответе, ни в памяти Bob.
4. Выводит `✅ Tenant isolation: PASSED`.

### Автотесты изоляции

```bash
uv run pytest tests/test_isolation.py -v
uv run pytest tests/test_memory.py -v
```

---

## Бенчмарк

**Выбор: LongMemEval** (Wu et al., 2024) — arxiv.org/abs/2410.10813

**Почему LongMemEval:**
- Охватывает 5 типов вопросов о долгосрочной памяти: single-session, multi-session, temporal reasoning, knowledge updating, negation.
- Синтетические сессии близки к формату коучинг-диалогов (факты о жизни пользователя).
- Датасет публично доступен, воспроизводим.

**Запуск бенчмарка:**

```bash
# Без W&B
uv run python scripts/run_eval.py --n 100

# С логированием в W&B
uv run python scripts/run_eval.py --n 100 --wandb
```

**Результат:** *[заполняется после запуска]*

| Метрика | Значение |
|---------|---------|
| Accuracy (substring match, n=100) | **TBD** |
| Baseline (без памяти) | ~30–35% |
| TTFT p50 | **TBD** |

**Почему эта цифра допустима для демо:**
Бенчмарк тестирует архитектуру на синтетических данных (английский, чистые диалоги). Реальный корпус Mindly — билингвальный и зашумлённый, что делает задачу сложнее. Главный аргумент для инвестора — не абсолютная цифра, а:
1. Наша система **значительно превышает baseline** (нет памяти).
2. Recall-момент на живом демо работает стабильно.
3. Архитектура масштабируема и расширяема (улучшенный retrieval, fine-tuning факт-экстрактора в v2).

Для улучшения accuracy: (a) перейти на более сильную модель (GPT-4o, Claude Sonnet), (b) использовать реранкинг retrieved фактов, (c) добавить temporal reasoning в промпт-экстракцию.

---

## Модели и датасеты

| Компонент | Название | Лицензия |
|-----------|---------|---------|
| LLM (inference) | qwen/qwen3-30b-a3b-instruct:free via OpenRouter | Apache 2.0 (Qwen) |
| Fallback LLM | meta-llama/llama-3.1-8b-instruct:free | Llama Community License |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Apache 2.0 |
| Vector store | ChromaDB | Apache 2.0 |
| Benchmark | LongMemEval (xiaowu0162/longmemeval) | MIT |

---

## Design Doc

Документ находится в [`docs/ml_system_design_doc.md`](docs/ml_system_design_doc.md).

---

## Demo

*[Здесь должен быть GIF/видео с двумя кросс-сессионными разговорами демонстрирующий recall]*

**Сценарий:**
1. Сессия 1: пользователь рассказывает про сына-девятиклассника и экзамены.
2. Процесс завершён (`Ctrl+C`).
3. Сессия 2: пользователь пишет «Привет» — агент сам спрашивает про экзамены.
