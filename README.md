# Mindly — велнес-коуч с памятью

**Гультяев Илья, Захарова Ксения** · Группа 972403

---

Это третье домашнее задание по NLP (экспериментальный трек). Задача — построить разговорного агента, который реально помнит пользователя между сессиями, а не делает вид.

Мы сделали CLI-коуча на трёх персонах. Закрыл терминал, открыл снова — агент помнит, что ты рассказывал про сына-девятиклассника или про первую пробежку. Память хранится в ChromaDB + SQLite на диске.

---

## Что внутри

- Персистентная память между сессиями (ChromaDB + SQLite)
- Три персоны с общей памятью клиента
- Изоляция пользователей — данные Алисы не утекут к Бобу
- Команды `/forget` — точечное и полное удаление памяти
- Стриминг токенов
- LongMemEval бенчмарк + логирование в W&B

---

## Архитектура памяти

Выбрали hand-rolled подход на ChromaDB + SQLite вместо готовых фреймворков типа mem0 или Letta. Главная причина — прозрачность и явный контроль над изоляцией.

```
Сообщение пользователя
        │
        ▼
ChromaDB.query(where user_id=X, k=7)   ← достаём релевантные факты
        │
        ▼
Собираем промпт: персона + факты + история сессии
        │
        ▼
LLM (OpenRouter) → стриминг токенов → пользователь
        │
        ▼  (в фоновом треде)
Экстрактор фактов (LLM, temp=0, JSON)
        │
        ├── ChromaDB.upsert(факты, {user_id, category, do_not_raise})
        └── SQLite INSERT (аудит-лог)
```

Изоляция реализована на уровне ChromaDB: каждый `retrieve()` и `delete()` передаёт `where={"user_id": user_id}`. Факты с флагом `do_not_raise=True` агент хранит, но никогда не поднимает первым — только если клиент сам заговорит.

---

## Как запустить

### Локально (uv)

```bash
git clone <repo-url>
cd mindly

# Создать .env и вписать ключ
cp .env.example .env
# OPENROUTER_API_KEY=sk-or-...

uv sync
uv run mindly chat --user alice --persona wellness_friend
```

### Docker

```bash
cp .env.example .env
docker-compose build
docker-compose run --rm mindly chat --user alice --persona wellness_friend
```

### Команды в чате

| Команда | Что делает |
|---------|-----------|
| `/forget <тема>` | Удалить воспоминания по теме |
| `/forget all` | Стереть всё |
| `/persona <имя>` | Сменить персону |
| `/memories` | Показать все сохранённые факты |
| `/quit` | Выйти |

---

## Персоны

Три персоны, одна база памяти. Можно переключаться между сессиями — агент всё помнит независимо от того, какая персона активна.

| Ключ | Имя | Стиль |
|------|-----|-------|
| `tough_love` | Drill Sergeant | Жёсткий, прямой, без сюсюканья |
| `wellness_friend` | Wellness Friend | Тёплый, как хороший друг |
| `cbt_coach` | CBT Coach | Структурированный, сократовы вопросы |

```bash
# Сначала поговорили с wellness_friend
uv run mindly chat --user alice --persona wellness_friend

# Потом переключились — агент помнит всё из прошлой сессии
uv run mindly chat --user alice --persona tough_love
```

---

## Управление памятью

### Удалить конкретную тему

```bash
# В чате:
/forget моя работа

# Из терминала:
uv run mindly forget --user alice "моя работа"
```

### Удалить всё (152-ФЗ / right-to-forget)

```bash
/forget all
# или
uv run mindly forget --user alice all
```

### Проверить изоляцию между пользователями

```bash
uv run python scripts/demo_isolation.py
```

Скрипт создаёт Алису, она что-то рассказывает. Потом создаётся Боб и спрашивает «что ты обо мне знаешь?». Проверяем, что факты Алисы не попали ни в ответ, ни в память Боба. Должно выйти `✅ Tenant isolation: PASSED`.

```bash
# Автотесты
uv run pytest tests/test_isolation.py -v
uv run pytest tests/test_memory.py -v
```

---

## Бенчмарк

Выбрали **LongMemEval** (Wu et al., 2024) — [arxiv.org/abs/2410.10813](https://arxiv.org/abs/2410.10813).

Почему он: охватывает пять типов вопросов о долгосрочной памяти, синтетические диалоги близки к нашему формату, датасет открытый и воспроизводимый.

```bash
uv run python scripts/run_eval.py --n 50 --wandb
```

**Результаты (n=12, oracle split):**

| Метрика | Значение |
|---------|---------|
| Accuracy (substring match) | **0.50** (6/12) |
| Baseline без памяти | ~0.30–0.35 |
| TTFT p50 | ~22 с (free tier OpenRouter) |

Разбивка по типам:

| Тип вопроса | Правильно | Всего | Accuracy |
|-------------|-----------|-------|---------|
| knowledge-update | 2 | 2 | 1.00 |
| single-session-user | 2 | 2 | 1.00 |
| multi-session | 1 | 2 | 0.50 |
| temporal-reasoning | 1 | 5 | 0.20 |
| single-session-preference | 0 | 1 | 0.00 |

**Почему n=12.** OpenRouter free tier — 50 запросов/день, каждый item бенчмарка требует ~4 вызова (экстракция фактов по сессиям + финальный вопрос). Прогнали сколько успели за два дня, но это не заглушка — система реально загружала память и отвечала на вопросы.

**Про цифру 0.50.** Это в 1.5 раза лучше baseline (нет памяти). Лучше всего работает на `knowledge-update` и `single-session-user` — агент уверенно вспоминает конкретные факты о клиенте. Хуже всего на `temporal-reasoning` (0.20) — модель путает когда именно что-то произошло. Это ожидаемо для бесплатной модели без явного хранения timestamp'ов. В v2 это лечится: хранить дату факта в метаданных ChromaDB и передавать в промпт.

Результаты в W&B: [sta66y-tomsk-state-university/mindly-eval](https://wandb.ai/sta66y-tomsk-state-university/mindly-eval)

---

## Модели и датасеты

| Компонент | Что используем | Лицензия |
|-----------|---------------|---------|
| LLM | qwen/qwen3-30b-a3b-instruct:free (OpenRouter) | Apache 2.0 |
| Fallback | meta-llama/llama-3.1-8b-instruct:free | Llama Community |
| Эмбеддинги | sentence-transformers/all-MiniLM-L6-v2 | Apache 2.0 |
| Vector store | ChromaDB | Apache 2.0 |
| Бенчмарк | LongMemEval (xiaowu0162/longmemeval) | MIT |

---

## Design Doc

[docs/ml_system_design_doc.md](docs/ml_system_design_doc.md)

---

## Демо

![Демо кросс-сессионной памяти](docs/assets/mindly-memory-demo.gif)

Сценарий записан с экрана реального запуска CLI: в первой сессии пользователь вручную печатает сообщение про учебу. Во второй сессии другая персона достаёт эти факты из памяти и вспоминает контекст прошлой сессии. `/memories` показывает сохранённые факты.
