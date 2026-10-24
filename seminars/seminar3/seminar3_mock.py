"""
seminar3_mock.py — offline-mock OpenAI-совместимого клиента для семинара 3.

Зачем: ноутбук должен исполняться без сервера (у студентов без GPU и в CI), а демонстрации — быть
воспроизводимыми. Mock повторяет *наблюдаемое* поведение модели на примерах ноутбука, включая
типичные поломки: план без approval и verifier, выполнение инструкции из отравленного описания
инструмента. Распределение модели он не воспроизводит, поэтому числа о качестве — только с живой модели.

Интерфейс совпадает с openai.OpenAI в той части, которую использует ноутбук:
    client.chat.completions.create(model=..., messages=..., tools=..., tool_choice=...,
                                   response_format=..., max_tokens=..., extra_body=..., ...)
Ответы собираются из типов SDK (ChatCompletion), поэтому код ноутбука исполняется на mock так же,
как на живой модели.

Два режима:
  * эвристический (по умолчанию): ответ выбирается по последнему сообщению и схеме response_format;
  * сценарный: FakeOpenAI(script=[...]) — очередь готовых ответов (dict в формате API) или исключений.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from openai.types.chat import ChatCompletion

ORDER_RE = re.compile(r"[AaАа]-\d{4}")
PRICE_RE = re.compile(r"(?:до|дешевле|не дороже)\s*(\d[\d\s]*)")
PRODUCT_WORDS = {
    "наушник": "наушники",
    "чайник": "чайник",
    "клавиатур": "клавиатура",
    "монитор": "монитор",
}


def _tokens(s: str) -> int:
    return max(1, len(s) // 4)


def _norm_id(s: str) -> str:
    return s.upper().replace("А", "A")


class FakeOpenAI:
    """Минимальный двойник openai.OpenAI."""

    def __init__(self, script: list | None = None, *, latency_s: float = 0.0, **_ignored):
        self.script = list(script or [])
        self.latency_s = latency_s
        self.calls: list[dict[str, Any]] = []
        self._n = 0
        self.chat = type("Chat", (), {})()
        self.chat.completions = _Completions(self)


class _Completions:
    def __init__(self, client: FakeOpenAI):
        self.c = client

    def create(
        self,
        *,
        model: str,
        messages: list,
        tools: list | None = None,
        tool_choice=None,
        response_format=None,
        **params,
    ):
        self.c.calls.append(
            dict(
                model=model,
                messages=list(messages),
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
                **params,
            )
        )
        if self.c.latency_s:
            time.sleep(self.c.latency_s)
        prompt_tokens = _tokens(json.dumps(messages, ensure_ascii=False, default=str)) + (
            _tokens(json.dumps(tools, ensure_ascii=False)) if tools else 0
        )
        if self.c.script:
            item = self.c.script.pop(0)
            if isinstance(item, Exception):
                raise item
            return ChatCompletion.model_validate(_fill(item, model))
        if params.get("max_tokens") == 1:  # «сколько стоит промпт» — ответ не нужен
            return _completion(model, text=".", prompt_tokens=prompt_tokens)
        self.c._n += 1
        return _heuristic(
            model, messages, tools or [], tool_choice, response_format, prompt_tokens, self.c._n
        )


# ====================================================================== эвристики
def _role(m) -> str:
    return m["role"] if isinstance(m, dict) else m.role


def _content(m) -> str:
    c = m.get("content") if isinstance(m, dict) else m.content
    return c if isinstance(c, str) else ""


def _tool_calls(m) -> list:
    tc = m.get("tool_calls") if isinstance(m, dict) else m.tool_calls
    return tc or []


def _last_user(messages) -> str:
    for m in reversed(messages):
        if _role(m) == "user":
            return _content(m)
    return ""


def _first_user(messages) -> str:
    for m in messages:
        if _role(m) == "user":
            return _content(m)
    return ""


def _system(messages) -> str:
    return "\n".join(_content(m) for m in messages if _role(m) == "system")


def _reason(text: str) -> str:
    low = text.lower()
    if any(
        k in low for k in ("разбит", "слома", "поврежд", "трещин", "не работает", "не включается")
    ):
        return "damaged"
    if any(k in low for k in ("не подош", "не подходит", "мал", "велик")):
        return "not_suitable"
    if any(k in low for k in ("не тот", "другой товар", "перепутал")):
        return "wrong_item"
    return "other"


def _product(text: str) -> str | None:
    low = text.lower()
    for stem, word in PRODUCT_WORDS.items():
        if stem in low:
            return word
    return None


def _heuristic(model, messages, tools, tool_choice, response_format, prompt_tokens, n):
    # 1) structured outputs: роутер и планировщик
    if response_format and response_format.get("type") == "json_schema":
        name = response_format["json_schema"].get("name", "")
        if name == "route_decision":
            return _completion(
                model,
                text=json.dumps(_route(_last_user(messages)), ensure_ascii=False),
                prompt_tokens=prompt_tokens,
            )
        if name == "workflow_spec":
            return _completion(
                model,
                text=json.dumps(_plan(messages), ensure_ascii=False),
                prompt_tokens=prompt_tokens,
            )
        return _completion(model, text="{}", prompt_tokens=prompt_tokens)

    # 2) финальный ответ по итогам workflow
    if "итоговый ответ клиенту" in _system(messages).lower():
        return _completion(model, text=_summary(_last_user(messages)), prompt_tokens=prompt_tokens)

    # 3) tool loop
    if tools and tool_choice != "none":
        calls = _plan_calls(messages, tools)
        if calls:
            return _completion(model, tool_calls=calls, prompt_tokens=prompt_tokens, counter=n)
    if _role(messages[-1]) == "tool":
        return _completion(model, text=_answer_from_tools(messages), prompt_tokens=prompt_tokens)
    return _completion(model, text=_plain(_last_user(messages)), prompt_tokens=prompt_tokens)


def _find_tool(tools: list, base: str) -> str | None:
    for t in tools:
        name = t["function"]["name"]
        if name == base or name.endswith("__" + base):
            return name
    return None


def _poisoned(tools: list, base: str) -> bool:
    for t in tools:
        f = t["function"]
        if (f["name"] == base or f["name"].endswith("__" + base)) and "<IMPORTANT>" in (
            f.get("description") or ""
        ):
            return True
    return False


def _results_since_last_assistant(messages) -> list[tuple[str, dict]]:
    """[(имя инструмента, распарсенный результат)] для хвоста role=tool."""
    out, names = [], {}
    for m in messages:
        if _role(m) == "assistant":
            for tc in _tool_calls(m):
                tid = tc["id"] if isinstance(tc, dict) else tc.id
                fn = tc["function"]["name"] if isinstance(tc, dict) else tc.function.name
                names[tid] = fn
    for m in reversed(messages):
        if _role(m) != "tool":
            break
        tid = m.get("tool_call_id") if isinstance(m, dict) else m.tool_call_id
        txt = _content(m)
        try:
            data = json.loads(txt[txt.find("{") :]) if "{" in txt else {"raw": txt}
        except json.JSONDecodeError:
            data = {"raw": txt}
        out.append((names.get(tid, "?").split("__")[-1], data))
    return out[::-1]


def _plan_calls(messages, tools) -> list[tuple[str, dict]]:
    user = _first_user(messages) if _role(messages[-1]) == "tool" else _last_user(messages)
    low = user.lower()
    ids = [_norm_id(x) for x in ORDER_RE.findall(user)]
    last = _results_since_last_assistant(messages) if _role(messages[-1]) == "tool" else []
    done = {
        name
        for m in messages
        if _role(m) == "assistant"
        for name in [
            ((tc["function"]["name"] if isinstance(tc, dict) else tc.function.name).split("__")[-1])
            for tc in _tool_calls(m)
        ]
    }

    # --- сценарий «цена в валюте» (демо tool poisoning)
    fx = _find_tool(tools, "convert_price")
    wants_fx = fx and any(k in low for k in ("доллар", "usd", "евро", "eur", "юан"))
    if wants_fx:
        if "search_products" not in done and _find_tool(tools, "search_products"):
            brand = re.search(r"\b([A-Z][a-z]{2,})\b", user)
            q = brand.group(1).lower() if brand else (_product(user) or "товар")
            return [(_find_tool(tools, "search_products"), {"query": q, "limit": 1})]
        if "convert_price" not in done and last:
            price = (
                next(
                    (
                        d.get("items", [{}])[0].get("price_rub")
                        for n_, d in last
                        if n_ == "search_products" and d.get("items")
                    ),
                    0,
                )
                or 0
            )
            cur = "EUR" if ("евро" in low or "eur" in low) else "USD"
            note = ""
            if _poisoned(tools, "convert_price"):
                # «уязвимая модель» выполняет инструкцию из описания инструмента
                note = _system(messages)
            return [(fx, {"amount_rub": price, "currency": cur, "note": note})]
        return []

    if _role(messages[-1]) == "tool":
        # после результата get_order модель может перейти к мутирующему шагу
        if "create_return" not in done and ids and any(k in low for k in ("верн", "возврат")):
            got = dict(last)
            info = got.get("get_order", {})
            if info.get("can_return"):
                return [
                    (
                        _find_tool(tools, "create_return"),
                        {"order_id": ids[0], "reason": _reason(user)},
                    )
                ]
        return []

    calls: list[tuple[str, dict]] = []
    if ids and any(k in low for k in ("верн", "возврат")) and _find_tool(tools, "create_return"):
        calls.append(
            (_find_tool(tools, "get_order"), {"order_id": ids[0]})
        )  # сначала проверить состояние
    elif ids and _find_tool(tools, "get_order"):
        calls += [(_find_tool(tools, "get_order"), {"order_id": oid}) for oid in ids]
    elif _find_tool(tools, "search_products") and (_product(user) or "есть" in low):
        args = {"query": _product(user) or "товар"}
        m = PRICE_RE.search(low)
        if m:
            args["max_price"] = int(m.group(1).replace(" ", ""))
        calls.append((_find_tool(tools, "search_products"), args))
    return calls


def _answer_from_tools(messages) -> str:
    parts = []
    for name, r in _results_since_last_assistant(messages):
        err = r.get("error") if isinstance(r.get("error"), dict) else None
        if err or r.get("error") or "raw" in r and "error" in r["raw"].lower():
            e = err or {"type": r.get("error"), "message": r.get("raw", ""), "hint": r.get("hint")}
            if e.get("type") in ("not_approved", "blocked_by_policy"):
                parts.append(
                    "Я не могу выполнить это действие без подтверждения сотрудника. "
                    "Передал запрос оператору."
                )
            elif e.get("type") == "not_found":
                parts.append(
                    "Заказ с таким номером не найден — проверьте, пожалуйста, номер (вид A-1234)."
                )
            else:
                parts.append(
                    f"Не удалось выполнить запрос: {e.get('message', '')} {e.get('hint') or ''}".strip()
                )
        elif "status" in r and "order_id" in r:
            parts.append(
                f"Заказ {r['order_id']}: статус — {r['status']}, доставка — {r.get('eta')}, "
                f"позиции: {', '.join(r.get('items', []))}."
            )
        elif "items" in r and isinstance(r["items"], list):
            if r["items"]:
                lines = [f"{i['name']} — {i.get('price_rub')} ₽" for i in r["items"]]
                more = (
                    f" (всего найдено {r['total']})" if r.get("total", 0) > len(r["items"]) else ""
                )
                parts.append("Нашлось: " + "; ".join(lines) + more + ".")
            else:
                parts.append("Подходящих товаров не нашлось.")
        elif "return_id" in r:
            again = (
                " (повторный запрос — возврат уже был оформлен)" if r.get("already_applied") else ""
            )
            parts.append(
                f"Оформил возврат {r['return_id']} по заказу {r['order_id']} на {r['amount_rub']} ₽"
                f"{again}: {r.get('next_step', '')}"
            )
        elif "amount" in r and "currency" in r:
            parts.append(f"Это примерно {r['amount']} {r['currency']} по текущему курсу.")
        else:
            parts.append(json.dumps(r, ensure_ascii=False)[:200])
    return " ".join(parts) or "Готово."


def _plain(text: str) -> str:
    low = text.lower()
    if any(k in low for k in ("верн", "возврат", "деньги")):
        return "Подскажите, пожалуйста, номер заказа (вида A-1234) — без него я не смогу оформить возврат."
    if any(k in low for k in ("привет", "здравств", "добрый")):
        return "Здравствуйте! Я ассистент поддержки «Лавки». Чем помочь: заказ, товары или возврат?"
    return "Уточните, пожалуйста, что именно вас интересует: статус заказа, товар или возврат."


# ---------------------------------------------------------------- policy router (§ 3.8)
def _route(text: str) -> dict:
    low = text.lower()
    ids = [_norm_id(x) for x in ORDER_RE.findall(text)]
    d = {
        "reasoning": "",
        "intent": "other",
        "route": "allow",
        "risk": "low",
        "confidence": 0.9,
        "missing": [],
        "order_ids": ids,
    }
    if any(
        k in low
        for k in ("игнорируй", "ignore previous", "system prompt", "системн", "все заказы всех")
    ):
        d.update(
            reasoning="Попытка изменить инструкции ассистента или массовое действие.",
            intent="other",
            route="refuse",
            risk="high",
            confidence=0.95,
        )
    elif any(k in low for k in ("суд", "юрист", "роспотреб", "прокурат", "жалоб")):
        d.update(
            reasoning="Юридическая претензия — решение за сотрудником.",
            intent="complaint",
            route="escalate",
            risk="high",
            confidence=0.85,
        )
    elif any(k in low for k in ("отмен",)):
        d.update(
            reasoning="Необратимое действие с заказом.",
            intent="cancel_order",
            risk="high",
            route="approval" if ids else "clarify",
            missing=[] if ids else ["order_id"],
            confidence=0.9,
        )
    elif any(k in low for k in ("верн", "возврат", "деньги назад", "верните деньги")):
        d.update(
            reasoning="Клиент просит возврат денег — изменение данных и деньги.",
            intent="return_request",
            risk="high",
            route="approval" if ids else "clarify",
            missing=[] if ids else ["order_id"],
            confidence=0.9,
        )
    elif ids:
        d.update(
            reasoning="Вопрос о статусе конкретного заказа — только чтение.", intent="order_status"
        )
    elif _product(text) or any(
        k in low for k in ("есть ли", "есть у вас", "сколько стоит", "цена")
    ):
        d.update(reasoning="Вопрос о товарах — только чтение.", intent="product_question")
    elif any(k in low for k in ("стих", "сочинени", "погод", "биткоин", "анекдот", "домашк")):
        d.update(
            reasoning="Запрос вне зоны ответственности ассистента магазина.",
            intent="out_of_scope",
            route="refuse",
            confidence=0.9,
        )
    else:
        d.update(
            reasoning="Намерение неясно — нужно уточнить.",
            intent="other",
            route="clarify",
            confidence=0.5,
            missing=["intent"],
        )
    return d


# ---------------------------------------------------------------- планировщик WorkflowSpec (§ 4.10–4.12)
def _node(id_, kind, *, tool=None, args=None, deps=None, approves=None, desc=""):
    return {
        "id": id_,
        "kind": kind,
        "tool": tool,
        "args": [{"name": k, "value": str(v)} for k, v in (args or {}).items()],
        "depends_on": deps or [],
        "approves": approves or [],
        "description": desc,
    }


def _plan(messages) -> dict:
    user = _first_user(messages)
    text = "\n".join(_content(m) for m in messages if _role(m) == "user")
    low = user.lower()
    ids = [_norm_id(x) for x in ORDER_RE.findall(user)]
    oid = ids[0] if ids else "A-0000"
    feedback_validator = "валидатор отклонил" in text.lower()
    feedback_failure = "завершился ошибкой" in text.lower()
    wants_cancel = "отмен" in low and not feedback_failure

    if wants_cancel:
        return {
            "reasoning": "Клиент просит отменить заказ: проверить статус, получить подтверждение, отменить, "
            "проверить результат, ответить.",
            "goal": f"Отменить заказ {oid}",
            "nodes": [
                _node(
                    "n1",
                    "tool",
                    tool="get_order",
                    args={"order_id": oid},
                    desc="Проверить статус заказа",
                ),
                _node("n2", "approval", deps=["n1"], approves=["n3"], desc="Подтверждение отмены"),
                _node(
                    "n3",
                    "tool",
                    tool="cancel_order",
                    args={"order_id": "$n1.order_id"},
                    deps=["n2"],
                    desc="Отменить заказ",
                ),
                _node(
                    "n4",
                    "verify",
                    tool="order_cancelled",
                    args={"order_id": "$n1.order_id"},
                    deps=["n3"],
                    desc="Убедиться, что заказ отменён",
                ),
                _node("n5", "respond", deps=["n4"], desc="Сообщить клиенту результат"),
            ],
        }

    reason = _reason(user)
    if not feedback_validator and not feedback_failure:
        # типичный первый план: «сразу сделать» — без approval и без проверки результата
        return {
            "reasoning": "Клиент просит возврат: найти заказ и оформить возврат.",
            "goal": f"Оформить возврат по заказу {oid}",
            "nodes": [
                _node("n1", "tool", tool="get_order", args={"order_id": oid}, desc="Найти заказ"),
                _node(
                    "n2",
                    "tool",
                    tool="create_return",
                    args={"order_id": oid, "reason": reason},
                    deps=["n1"],
                    desc="Оформить возврат",
                ),
                _node("n3", "respond", deps=["n2"], desc="Ответить клиенту"),
            ],
        }
    why = (
        "План исправлен по замечаниям валидатора: добавлены подтверждение и проверка результата."
        if feedback_validator
        else "Отмена невозможна (заказ уже доставлен) — по подсказке инструмента оформляем возврат."
    )
    return {
        "reasoning": why,
        "goal": f"Оформить возврат по заказу {oid}",
        "nodes": [
            _node(
                "n1",
                "tool",
                tool="get_order",
                args={"order_id": oid},
                desc="Проверить заказ и статус",
            ),
            _node(
                "n2", "approval", deps=["n1"], approves=["n3"], desc="Подтверждение возврата денег"
            ),
            _node(
                "n3",
                "tool",
                tool="create_return",
                args={"order_id": "$n1.order_id", "reason": reason},
                deps=["n2"],
                desc="Оформить возврат",
            ),
            _node(
                "n4",
                "verify",
                tool="return_exists",
                args={"order_id": "$n1.order_id"},
                deps=["n3"],
                desc="Проверить, что возврат создан",
            ),
            _node("n5", "respond", deps=["n4"], desc="Сообщить клиенту номер возврата и сроки"),
        ],
    }


def _summary(context: str) -> str:
    m = re.search(r'"return_id":\s*"(R-\d+)"', context)
    if m:
        amt = re.search(r'"amount_rub":\s*(\d+)', context)
        return (
            f"Возврат {m.group(1)} оформлен{', сумма ' + amt.group(1) + ' ₽' if amt else ''}. "
            f"Курьер заберёт товар в течение 3 дней, деньги вернутся за 5–7 дней."
        )
    if '"status": "cancelled"' in context:
        return "Заказ отменён, деньги вернутся на карту в течение 5 дней."
    return "Запрос обработан."


# ====================================================================== сборка ответа
def _completion(
    model: str,
    *,
    text: str | None = None,
    tool_calls: list | None = None,
    prompt_tokens: int = 50,
    counter: int = 0,
) -> ChatCompletion:
    message: dict[str, Any] = {"role": "assistant", "content": text}
    finish = "stop"
    completion_text = text or ""
    if tool_calls:
        message["tool_calls"] = [
            {
                "id": f"call_{counter:03d}{i}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
            }
            for i, (name, args) in enumerate(tool_calls)
        ]
        finish = "tool_calls"
        completion_text = json.dumps(message["tool_calls"], ensure_ascii=False)
    data = {
        "id": f"chatcmpl-mock-{counter}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": _tokens(completion_text),
            "total_tokens": prompt_tokens + _tokens(completion_text),
        },
    }
    return ChatCompletion.model_validate(data)


def _fill(item: dict, model: str) -> dict:
    d = dict(item)
    d.setdefault("id", "chatcmpl-script")
    d.setdefault("object", "chat.completion")
    d.setdefault("created", int(time.time()))
    d.setdefault("model", model)
    d.setdefault("usage", {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60})
    for ch in d["choices"]:
        ch.setdefault("index", 0)
        ch["message"].setdefault("role", "assistant")
        ch["message"].setdefault("content", None)
        ch.setdefault("finish_reason", "tool_calls" if ch["message"].get("tool_calls") else "stop")
    return d
