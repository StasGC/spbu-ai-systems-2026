"""
seminar1_mock.py — offline-mock OpenAI-совместимого клиента для семинара 1.

Зачем: ноутбук семинара должен исполняться без сервера (в CI и у студентов без GPU),
а тесты tool loop должны быть детерминированными. Mock воспроизводит *наблюдаемое*
поведение модели на примерах ноутбука (tool calls, structured output, типичные поломки),
но не её распределение: любые числа о качестве — только с живой модели.

Интерфейс совпадает с openai.OpenAI ровно в той части, которую использует ноутбук:
    client.chat.completions.create(model=..., messages=..., tools=..., tool_choice=...,
                                   response_format=..., stream=..., stream_options=..., ...)
Ответы собираются из тех же типов SDK (ChatCompletion / ChatCompletionChunk), поэтому
tool loop, парсер и трейсинг исполняются на mock тем же кодом, что и на живой модели.

Два режима:
  * эвристический (по умолчанию): ответ выбирается по ключевым словам последнего сообщения;
  * сценарный: FakeOpenAI(script=[...]) — очередь готовых ответов (dict в формате API)
    или исключений; подходит для unit-тестов (см. последний блок ноутбука).
"""
from __future__ import annotations

import json
from copy import deepcopy
import re
import time
from typing import Any

import httpx
import openai
from openai.types.chat import ChatCompletion, ChatCompletionChunk

ORDER_RE = re.compile(r"A-\d{4}")
PRICE_RE = re.compile(r"(?:до|дешевле|не дороже)\s*(\d[\d\s]*)")
PRODUCT_WORDS = ["наушники", "чайник", "клавиатур", "мыш", "монитор", "кабель", "колонк", "телефон", "ноутбук", "экран"]


def _now() -> int:
    return int(time.time())


def _tokens(s: str) -> int:
    return max(1, len(s) // 4)


class FakeOpenAI:
    """Минимальный двойник openai.OpenAI."""

    def __init__(self, script: list | None = None, *, latency_s: float = 0.015, fail_first: Exception | None = None, **_ignored):
        self.script = list(script or [])
        self.scripted = script is not None
        self.latency_s = latency_s
        self.fail_first = fail_first
        self.calls: list[dict[str, Any]] = []
        self._counter = 0
        self.chat = _Chat(self)


class _Chat:
    def __init__(self, client: FakeOpenAI):
        self.completions = _Completions(client)


class _Completions:
    def __init__(self, client: FakeOpenAI):
        self.c = client

    # ------------------------------------------------------------------ API
    def create(self, *, model: str, messages: list, tools: list | None = None, tool_choice=None,
               response_format=None, stream: bool = False, stream_options=None, **params):
        self.c.calls.append(deepcopy(dict(model=model, messages=messages, tools=tools, tool_choice=tool_choice,
                                         response_format=response_format, stream=stream, **params)))
        if self.c.latency_s:
            time.sleep(self.c.latency_s)
        if self.c.fail_first is not None:              # одна ошибка «по команде» (429 / таймаут)
            exc, self.c.fail_first = self.c.fail_first, None
            raise exc
        if self.c.scripted:                            # сценарий не переходит молча в эвристику
            if not self.c.script:
                raise AssertionError("FakeOpenAI: сценарий исчерпан — лишний вызов модели")
            item = self.c.script.pop(0)
            if isinstance(item, Exception):
                raise item
            resp = ChatCompletion.model_validate(_fill(item, model))
        else:
            resp = self._heuristic(model, messages, tools, tool_choice, response_format, params)
        if stream:
            return _stream(resp, include_usage=bool(stream_options and stream_options.get("include_usage")))
        return resp

    # -------------------------------------------------------------- эвристика
    def _heuristic(self, model, messages, tools, tool_choice, response_format, params):
        self.c._counter += 1
        last = messages[-1]
        role = last["role"] if isinstance(last, dict) else last.role
        user_text = _last_user_text(messages)
        prompt_tokens = _tokens(json.dumps(messages, ensure_ascii=False, default=_dump)) + (_tokens(json.dumps(tools)) if tools else 0)

        # 1) после результатов инструментов — финальный текстовый ответ
        if role == "tool":
            text = _answer_from_tool_results(messages)
            return _completion(model, text=text, prompt_tokens=prompt_tokens)

        # 2) structured output по схеме (response_format) — «грамматика гарантирует форму»
        if response_format and response_format.get("type") == "json_schema":
            text = json.dumps(_ticket(user_text), ensure_ascii=False)
            return _completion(model, text=text, prompt_tokens=prompt_tokens)

        # 3) prompt-only JSON: первая попытка ломается, после текста ошибки — исправляется
        if not tools and _asks_json(messages):
            retry = any(_is_validation_feedback(m) for m in messages)
            if not retry:
                broken = _ticket(user_text)
                broken["urgency"] = "очень срочно" if broken["urgency"] == "high" else "обычная"   # не из enum
                text = "Конечно! Вот JSON:\n```json\n" + json.dumps(broken, ensure_ascii=False, indent=2) + "\n```"
            else:
                text = json.dumps(_ticket(user_text), ensure_ascii=False)
            return _completion(model, text=text, prompt_tokens=prompt_tokens)

        # 4) function calling
        if tools and tool_choice != "none":
            names = [t["function"]["name"] for t in tools]
            calls = _plan_calls(user_text, names, params.get("parallel_tool_calls", True))
            forced = _forced_tool(tool_choice)
            if forced and not any(c[0] == forced for c in calls):
                # принудили вызов там, где модель хотела ответить текстом -> аргументы выдуманы
                calls = [(forced, _invented_args(forced, user_text))]
            elif tool_choice == "required" and not calls:
                calls = [(names[0], _invented_args(names[0], user_text))]
            if calls:
                return _completion(model, tool_calls=calls, prompt_tokens=prompt_tokens, counter=self.c._counter)

        # 5) обычный текст
        return _completion(model, text=_plain_answer(user_text), prompt_tokens=prompt_tokens)


# ====================================================================== helpers
def _dump(o):
    return o.model_dump(exclude_none=True) if hasattr(o, "model_dump") else str(o)


def _msg_role(m):
    return m["role"] if isinstance(m, dict) else m.role


def _msg_content(m) -> str:
    c = m["content"] if isinstance(m, dict) else m.content
    return c if isinstance(c, str) else ""


def _last_user_text(messages) -> str:
    """Последнее сообщение пользователя, не считая наших же сообщений с текстом ошибки валидации."""
    for m in reversed(messages):
        if _msg_role(m) == "user" and not _is_validation_feedback(m):
            return _msg_content(m)
    return ""


def _asks_json(messages) -> bool:
    for m in messages:
        if _msg_role(m) == "system" and "json" in _msg_content(m).lower():
            return True
    return False


def _is_validation_feedback(m) -> bool:
    return _msg_role(m) == "user" and "валидац" in _msg_content(m).lower()


def _forced_tool(tool_choice):
    if isinstance(tool_choice, dict):
        return tool_choice.get("function", {}).get("name")
    return None


def _guess_product(text: str) -> str:
    low = text.lower()
    for w in PRODUCT_WORDS:
        if w in low:
            return w if w in ("наушники", "чайник", "монитор", "кабель", "телефон", "ноутбук", "экран") else {"клавиатур": "клавиатура", "мыш": "мышь", "колонк": "колонки"}[w]
    return "товар"


def _guess_reason(text: str) -> str:
    low = text.lower()
    if any(k in low for k in ("разбит", "сломан", "поврежд", "трещин", "не работает")):
        return "damaged"
    if any(k in low for k in ("не подош", "не подходит", "мал", "велик")):
        return "not_suitable"
    if any(k in low for k in ("не тот", "другой товар", "перепутал")):
        return "wrong_item"
    return "other"


def _ticket(text: str) -> dict:
    low = text.lower()
    ids = ORDER_RE.findall(text)
    if any(k in low for k in ("верн", "возврат", "деньги назад")):
        intent = "return_request"
    elif ids or any(k in low for k in ("заказ", "доставк", "где мой")):
        intent = "order_status"
    elif any(k in low for k in ("есть ли", "есть у вас", "цена", "стоит", "сколько", "до ")) or _guess_product(text) != "товар":
        intent = "product_question"
    else:
        intent = "other"
    urgency = "high" if any(k in low for k in ("срочно", "немедленно", "сегодня")) else ("normal" if intent != "other" else "low")
    product = None if _guess_product(text) == "товар" else _guess_product(text)
    summary = {
        "return_request": f"Клиент просит возврат{' по заказу ' + ids[0] if ids else ''}: {_guess_reason(text)}",
        "order_status": f"Клиент спрашивает о статусе заказа{' ' + ids[0] if ids else ''}",
        "product_question": f"Вопрос о товаре: {product or 'не указан'}",
        "other": "Обращение без конкретного запроса",
    }[intent]
    return {
        "reasoning": f"В тексте {'есть' if ids else 'нет'} номер заказа; ключевые слова указывают на {intent}.",
        "intent": intent,
        "order_id": ids[0] if ids else None,
        "product": product,
        "urgency": urgency,
        "summary": summary,
    }


def _plan_calls(text: str, names: list[str], parallel: bool) -> list[tuple[str, dict]]:
    low = text.lower()
    ids = ORDER_RE.findall(text)
    calls: list[tuple[str, dict]] = []
    if ids and any(k in low for k in ("верн", "возврат")) and "create_return" in names:
        calls.append(("create_return", {"order_id": ids[0], "reason": _guess_reason(text)}))
    elif ids and "get_order" in names:
        for oid in (ids if parallel else ids[:1]):
            calls.append(("get_order", {"order_id": oid}))
    elif "search_products" in names and (any(k in low for k in ("есть ли", "есть у вас", "цена", "стоит", "сколько", "подобрать", "посоветуй", "хочу купить")) or _guess_product(text) != "товар") and not any(k in low for k in ("верн", "возврат")):
        m = PRICE_RE.search(low)
        args = {"query": _guess_product(text), "max_price": None}
        if m:
            args["max_price"] = int(m.group(1).replace(" ", ""))
        calls.append(("search_products", args))
    return calls


def _invented_args(name: str, text: str) -> dict:
    return {"get_order": {"order_id": "A-0000"}, "search_products": {"query": text[:20] or "товар", "max_price": None},
            "create_return": {"order_id": "A-0000", "reason": "other"}}.get(name, {})


def _plain_answer(text: str) -> str:
    low = text.lower()
    if any(k in low for k in ("верн", "возврат", "деньги")):
        return "Подскажите, пожалуйста, номер заказа (вида A-1234) — без него я не смогу оформить возврат."
    if any(k in low for k in ("привет", "здравств", "добрый")):
        return "Здравствуйте! Я ассистент поддержки магазина. Чем могу помочь: статус заказа, подбор товара или возврат?"
    return "Уточните, пожалуйста, что именно вас интересует: статус заказа, товар или возврат."


def _answer_from_tool_results(messages) -> str:
    results = []
    for m in reversed(messages):
        if _msg_role(m) != "tool":
            break
        try:
            results.append(json.loads(_msg_content(m)))
        except json.JSONDecodeError:
            results.append({"raw": _msg_content(m)})
    results.reverse()
    parts = []
    for r in results:
        if "error" in r:
            hint = r.get("hint") or r.get("detail") or ""
            if r["error"] == "not_found":
                parts.append("Заказ с таким номером не найден — проверьте, пожалуйста, номер (вид A-1234) и напишите его ещё раз.")
            elif r["error"] == "unknown_tool":
                parts.append("Я попробовал вызвать несуществующий инструмент; повторю запрос правильно.")
            else:
                parts.append(f"Не удалось выполнить запрос ({r['error']}). {hint}".strip())
        elif "status" in r and "order_id" in r:
            parts.append(f"Заказ {r['order_id']}: статус — {r['status']}, ожидаемая доставка — {r.get('eta', 'уточняется')}, позиции: {', '.join(r.get('items', []))}.")
        elif "items" in r and isinstance(r["items"], list) and r["items"] and isinstance(r["items"][0], dict):
            lines = [f"{i['name']} — {i['price']} ₽ ({'в наличии' if i.get('in_stock') else 'нет в наличии'})" for i in r["items"]]
            parts.append("Вот что нашлось: " + "; ".join(lines) + ".")
        elif "return_id" in r:
            parts.append(f"Оформил возврат {r['return_id']} по заказу {r['order_id']}: {r.get('next_step', '')}".strip())
        else:
            parts.append(json.dumps(r, ensure_ascii=False))
    return " ".join(parts) or "Готово."


def _completion(model: str, *, text: str | None = None, tool_calls: list[tuple[str, dict]] | None = None,
                prompt_tokens: int = 50, counter: int = 0) -> ChatCompletion:
    message: dict[str, Any] = {"role": "assistant", "content": text}
    finish = "stop"
    completion_text = text or ""
    if tool_calls:
        message["tool_calls"] = [
            {"id": f"call_{counter:03d}{i}", "type": "function",
             "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}
            for i, (name, args) in enumerate(tool_calls)
        ]
        finish = "tool_calls"
        completion_text = json.dumps(message["tool_calls"])
    data = {
        "id": f"chatcmpl-mock-{_now()}", "object": "chat.completion", "created": _now(), "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": _tokens(completion_text),
                  "total_tokens": prompt_tokens + _tokens(completion_text)},
    }
    return ChatCompletion.model_validate(data)


def _fill(item: dict, model: str) -> dict:
    """Дополняет сценарный dict обязательными полями формата API."""
    d = deepcopy(item)
    d.setdefault("id", f"chatcmpl-script-{_now()}")
    d.setdefault("object", "chat.completion")
    d.setdefault("created", _now())
    d.setdefault("model", model)
    d.setdefault("usage", {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60})
    for ch in d["choices"]:
        ch.setdefault("index", 0)
        ch["message"].setdefault("role", "assistant")
        ch["message"].setdefault("content", None)
        ch.setdefault("finish_reason", "tool_calls" if ch["message"].get("tool_calls") else "stop")
    return d


def _stream(resp: ChatCompletion, *, include_usage: bool):
    """Режет готовый ответ на чанки так, как это делает сервер: дельты текста и фрагменты аргументов."""
    base = {"id": resp.id, "object": "chat.completion.chunk", "created": resp.created, "model": resp.model}
    msg = resp.choices[0].message

    def chunk(delta: dict, finish=None):
        return ChatCompletionChunk.model_validate({**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})

    yield chunk({"role": "assistant", "content": ""})
    if msg.content:
        for i in range(0, len(msg.content), 7):
            yield chunk({"content": msg.content[i:i + 7]})
    for idx, tc in enumerate(msg.tool_calls or []):
        yield chunk({"tool_calls": [{"index": idx, "id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": ""}}]})
        args = tc.function.arguments
        for i in range(0, len(args), 6):
            yield chunk({"tool_calls": [{"index": idx, "function": {"arguments": args[i:i + 6]}}]})
    yield chunk({}, finish=resp.choices[0].finish_reason)
    if include_usage:
        yield ChatCompletionChunk.model_validate({**base, "choices": [], "usage": resp.usage.model_dump()})


# ------------------------------------------------------------- ошибки по команде
def rate_limit_error(retry_after: str = "0.2") -> openai.RateLimitError:
    """Готовый 429 для демонстрации ретраев: FakeOpenAI(fail_first=rate_limit_error())."""
    req = httpx.Request("POST", "http://mock/v1/chat/completions")
    resp = httpx.Response(429, request=req, headers={"retry-after": retry_after}, json={"error": {"message": "rate limited"}})
    return openai.RateLimitError("rate limited", response=resp, body=None)


def timeout_error() -> openai.APITimeoutError:
    return openai.APITimeoutError(request=httpx.Request("POST", "http://mock/v1/chat/completions"))
