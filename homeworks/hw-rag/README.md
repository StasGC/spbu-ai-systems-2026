# HW2. Ассистент-аналитик новостного архива: агент с search tool (RAG)

## Задача

Дан корпус текстов — [MultiHop-RAG](https://github.com/yixuantt/MultiHop-RAG): 609 англоязычных новостей (сентябрь–декабрь 2023) и
2 556 вопросов, ответ на каждый собирается из 2–4 статей и 301 `null` — ответа в архиве нет, правильный ответ `Insufficient information.`

Для каждого вопроса известны gold-предложения (`evidence_list[*].fact`), поэтому retrieval и ответы считаются
без судьи, по официальным правилам датасета, и ваши числа сравнимы со статьёй и с открытыми реализациями.

Вы пишете два способа отвечать и сравниваете их:

- **агент** (часть A): tool loop с инструментами `search` и `read_document`; модель сама решает, сколько раз искать
  и какую статью прочитать целиком;
- **single-shot RAG** (часть B): top-k чанков в один промпт, ответ в JSON с цитатами, которые проверяет код.

Под ними — ваш retrieval: chunking, BM25, dense-индекс с кэшем эмбеддингов, гибрид через RRF(Reciprocal Rank Fusion).

В репозитории есть макет:

- протоколы (`ChatModel`, `Embedder`, `Retriever`)
- OpenAI-compatible адаптеры
- моки (`ScriptedChat`, `HashEmbedder`, `RuleChat`)
- официальные правила подсчёта, CLI, данные и тесты. 

Всё остальное — `raise NotImplementedError("TODO …")`, надо реализовать.

> Скелет внутри репозитория не является строгим макетом работы. Если есть другое представления для реализации агента с RAG, можно реализовать по другому.

Контракт каждой функции — в её docstring; тесты в `tests/` — исполняемая спецификация. 

> Агентные фреймворки
(LangChain, LlamaIndex и подобные) в `newsrag/` запрещены; `rank_bm25` только в `baselines.py`.

## Установка

Окружение настраивается через `pyproject.toml`: 

```bash
python -m venv .venv && . .venv/bin/activate     
pip install -e ".[test]"
ruff check . && pytest                           
```

## Модель

Можно использовать любую локальную модель (желательно, выше 0.6B), или модели через API.

В репозитории пример с локальной моделью, сервинг через OpenAI-compatible API. По умолчанию Ollama:

```bash
ollama pull qwen3:8b && ollama pull bge-m3       # ⚠ имена тегов сверьте с ollama.com/library
OLLAMA_CONTEXT_LENGTH=16384 ollama serve         # ⚠ агенту нужно ≥ 16k контекста: две статьи + выдача search
```

| Переменная | По умолчанию |
|---|---|
| `NEWSRAG_BASE_URL` | `http://localhost:11434/v1` |
| `NEWSRAG_MODEL` | `qwen3:8b` |
| `NEWSRAG_EMBED_MODEL` | `bge-m3` |
| `NEWSRAG_API_KEY` | `none` |
| `NEWSRAG_EXTRA_BODY` | `{"reasoning_effort": "none"}` |
| `NEWSRAG_BACKEND` | — |

`response_format` с JSON-схемой на Ollama ⚠ сверьте на одном запросе `python -m newsrag ask "…"` до больших прогонов.

## Что написать

| TODO | Где | Что |
|---|---|---|
| A1 | `model.py` | `RetryingChat`: повтор только retryable-ошибок, `retry_after` или backoff с jitter |
| A2 | `agent.py` | схемы `TOOLS`, тела `search` и `read_document` |
| A3 | `agent.py` | `run_agent`: история с `tool_call_id`, пачка вызовов целиком, ошибки как tool-результат, лимит шагов |
| B1 | `corpus.py` | `chunk`: заголовок с источником и датой, по предложениям, ≤ `max_chars` |
| B2 | `retrieval.py` | `BM25Index` по формуле из docstring |
| B3 | `retrieval.py` | `DenseIndex`: cosine + JSON-кэш векторов |
| B4 | `retrieval.py` | `HybridIndex`: RRF |
| B5 | `agent.py` | `answer_grounded`: контекст с id, JSON через `response_format`, проверка цитат кодом |
| B6 | `evaluate.py` | `retrieval_metrics`, `answer_metrics`, `run_eval` по официальным правилам |
| — | `ADR.md` | эксперименты и записи решений (ниже) |

Порядок: A1 $\Rightarrow$ B1 $\Rightarrow$ B2 $\Rightarrow$ A2 $\Rightarrow$ A3 $\Rightarrow$ B5 $\Rightarrow$ B6 $\Rightarrow$ B3 $\Rightarrow$ B4. 

После A3 и B2 агент уже работает на BM25.

## Команды

```bash
python -m newsrag ask "Who is the CEO of OpenAI?" --index bm25      # single-shot RAG
python -m newsrag agent "…" --index hybrid                          # агент; история — в traces.jsonl
python -m newsrag eval retrieval --index bm25                       # Hits@4 / Hits@10 / MRR@10 на всех 2 255 вопросах с evidence
python -m newsrag eval answers --mode rag --index hybrid --sample 200 --seed 0
python -m newsrag eval answers --mode agent --index hybrid --sample 200 --seed 0
pip install -e ".[baselines]" && python baselines.py                # ваш BM25 рядом с rank_bm25 и строкой статьи
```

## Эксперименты и `ADR.md`

Дополнительно к коду надо составить архитектурный отчет ADR(Architecture Decision Records) о реализации основных компонентов системы.

В отчете можно представить схему архитектуры агента, ключевые решения в компонентах, хитрые решения и использование техник из лекций.

Минимум шесть записей: chunking; токенизация и BM25; dense-индекс (модель, кэш); гибрид и его параметры;
стратегия агента (когда искать повторно, когда читать статью, бюджет шагов) и правило отказа; выбор метрики
(official accuracy по общему слову против `exact_match`). 

В конце — раздел «Итоги» с двумя таблицами:

1. Retrieval: `bm25 | dense | hybrid` × Hits@4 / Hits@10 / MRR@10 на всех вопросах, плюс строки
   `rank_bm25` и результаты из статьи (bge-large-en-v1.5).
2. Answers: `rag | agent` на одной выборке `--sample 200 --seed 0` с лучшим индексом

Использование AI-ассистентов разрешено; что именно сгенерировано и как проверено — отдельной записью в `ADR.md`.

## Оценка

| Часть | Баллы | Из чего                                                                        |
|---|-------|--------------------------------------------------------------------------------|
| A. Агент | 20    | A1 — 5, A2 — 5, A3 — 10                                                        |
| B. RAG | 60    | B1 — 5, B2 — 5, B3 — 5, B4 — 5, B5 — 10, B6 и эксперименты — 15, `ADR.md` — 15 |
| Защита | 20    | Вопросы по вашему коду и решениям                                              |

Публичные тесты (`pytest -q` локально или в GitHub Actions вашего репозитория: workflow `.github/workflows/tests.yml`
уже в репозитории) и скрытые тесты (прогоняет препоаватель).

Финальную оценку ставит проверяющий по коду, `ADR.md`, артефактам и защите. 

На оценку не влияют: абсолютная точность модели, мощность компьютера, объём кода.

Штрафы за: 

- hardcode под известные вопросы
- тесты, которые «подогнаны» под ответы, 
- числа без артефактов

> **Бонус**
> 
> +5 баллов: LLM-rerank поверх гибрида с измеренным эффектом

## Сдача

1. Создайте **приватный** репозиторий
2. Скопируйте homeworks/hw-rag/ к себе в репозиторий
3. Добавьте пользователя http://github.com/StasGC к себе в репозиторий
4. Заполните данные в таблице

## Дедлайн

- финальный коммит с решением до **08.11.2026 23:59**