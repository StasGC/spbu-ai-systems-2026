"""
seminar3_store.py — магазин «Лавка» для семинара 3: состояние в SQLite, инструменты, реестр ToolSpec.

Один модуль используют и ноутбук, и MCP-сервер (seminar3_mcp_server.py), поэтому состояние
лежит в файле store.db, а не в памяти: сервер-подпроцесс и ноутбук видят одни и те же заказы,
возвраты, ключи идемпотентности и audit log.

Что здесь показано (лекция 3, часть 1):
  * инструменты «под задачу», а не обёртки над таблицами (§ 1.2);
  * ответы: concise / detailed, лимит и подсказка при усечении (§ 1.4);
  * typed errors: код, сообщение, retryable, hint (§ 1.5);
  * классы побочных эффектов и риск (§ 1.6), идемпотентность по ключу (§ 1.7), dry-run и audit (§ 1.8);
  * ToolSpec — «контракт инструмента» в одном месте (§ 1.9).
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field

DB_PATH = os.environ.get(
    "S3_DB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "store.db")
)

# ================================================================== данные магазина
SEED_ORDERS = [
    # order_id, status, eta, items, total_rub, customer
    ("A-1042", "shipped", "2026-09-20", ["Наушники Sennheiser HD 560S"], 18990, "c-17"),
    ("A-1043", "packed", "2026-09-22", ["Чайник Bork K515", "Кабель USB-C 2 м"], 13980, "c-17"),
    ("A-1044", "delivered", "2026-09-15", ["Клавиатура Keychron K2"], 9990, "c-23"),
    ("A-1045", "delivered", "2026-09-10", ["Монитор Dell U2724D"], 32990, "c-23"),
]
SEED_PRODUCTS = [
    # sku, name, price, stock, tags, description
    (
        "SKU-1001",
        "Наушники Sony WH-CH520",
        4990,
        12,
        "наушники",
        "Беспроводные накладные, 50 ч работы, USB-C",
    ),
    (
        "SKU-1002",
        "Наушники Sennheiser HD 560S",
        18990,
        3,
        "наушники",
        "Открытые студийные, 120 Ом, кабель 3 м",
    ),
    (
        "SKU-1003",
        "Наушники JBL Tune 510BT",
        3490,
        0,
        "наушники",
        "Беспроводные, складные, 40 ч работы",
    ),
    (
        "SKU-1004",
        "Наушники Marshall Major IV",
        11990,
        5,
        "наушники",
        "Беспроводные, 80 ч, беспроводная зарядка",
    ),
    ("SKU-1005", "Наушники Xiaomi Buds 5", 5490, 20, "наушники", "TWS, шумоподавление, IP54"),
    (
        "SKU-1006",
        "Наушники Apple AirPods 4",
        14990,
        7,
        "наушники",
        "TWS, USB-C, пространственный звук",
    ),
    ("SKU-1007", "Наушники Audio-Technica M40x", 12490, 2, "наушники", "Закрытые студийные, 35 Ом"),
    ("SKU-2001", "Чайник Bork K515", 12990, 4, "чайник", "1.7 л, 7 температурных режимов"),
    ("SKU-3001", "Клавиатура Keychron K2", 9990, 9, "клавиатура", "Механическая, 75%, Bluetooth"),
    ("SKU-4001", "Монитор Dell U2724D", 32990, 2, "монитор", '27", 2560x1440, IPS, USB-C 90 Вт'),
]


def connect(db: str | None = None) -> sqlite3.Connection:
    con = sqlite3.connect(
        db or DB_PATH, timeout=10, isolation_level=None
    )  # autocommit; транзакции явно
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    return con


def reset(db: str | None = None) -> None:
    """Пересоздать базу магазина (идемпотентно: можно звать перед каждым демо)."""
    con = connect(db)
    con.executescript("""
        DROP TABLE IF EXISTS orders; DROP TABLE IF EXISTS products; DROP TABLE IF EXISTS returns;
        DROP TABLE IF EXISTS idempotency; DROP TABLE IF EXISTS audit;
        CREATE TABLE orders (order_id TEXT PRIMARY KEY, status TEXT, eta TEXT, items TEXT, total_rub INT, customer TEXT);
        CREATE TABLE products (sku TEXT PRIMARY KEY, name TEXT, price INT, stock INT, tags TEXT, description TEXT);
        CREATE TABLE returns (return_id TEXT PRIMARY KEY, order_id TEXT, reason TEXT, amount_rub INT, created REAL);
        CREATE TABLE idempotency (key TEXT PRIMARY KEY, fingerprint TEXT, result TEXT, created REAL);
        CREATE TABLE audit (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, actor TEXT, tool TEXT,
                            args TEXT, dry_run INT, outcome TEXT, detail TEXT);
    """)
    con.executemany(
        "INSERT INTO orders VALUES (?,?,?,?,?,?)",
        [(o, s, e, json.dumps(i, ensure_ascii=False), t, c) for o, s, e, i, t, c in SEED_ORDERS],
    )
    con.executemany("INSERT INTO products VALUES (?,?,?,?,?,?)", SEED_PRODUCTS)
    con.close()


def ensure_db(db: str | None = None) -> None:
    con = connect(db)
    ok = con.execute("SELECT name FROM sqlite_master WHERE name='orders'").fetchone()
    con.close()
    if not ok:
        reset(db)


# ================================================================== typed errors
class ToolError(Exception):
    """Ожидаемая ошибка инструмента: уходит модели как РЕЗУЛЬТАТ, а не как исключение (лекция 1, § 3.6).

    code      — машинный класс: validation | not_found | conflict | not_allowed | rate_limited | unavailable
    retryable — есть ли смысл повторить тот же вызов (аналог Stripe-Should-Retry)
    hint      — что делать дальше: модели или человеку
    """

    def __init__(
        self, code: str, message: str, *, retryable: bool = False, hint: str | None = None
    ):
        super().__init__(f"{code}: {message}")
        self.code, self.message, self.retryable, self.hint = code, message, retryable, hint

    def to_dict(self) -> dict:
        d = {"error": {"type": self.code, "message": self.message, "retryable": self.retryable}}
        if self.hint:
            d["error"]["hint"] = self.hint
        return d


# ================================================================== audit
def audit(
    actor: str,
    tool: str,
    args: dict,
    *,
    dry_run: bool,
    outcome: str,
    detail: str = "",
    db: str | None = None,
):
    con = connect(db)
    con.execute(
        "INSERT INTO audit (ts, actor, tool, args, dry_run, outcome, detail) VALUES (?,?,?,?,?,?,?)",
        (
            time.time(),
            actor,
            tool,
            json.dumps(args, ensure_ascii=False),
            int(dry_run),
            outcome,
            detail[:500],
        ),
    )
    con.close()


def audit_log(limit: int = 20, db: str | None = None) -> list[dict]:
    con = connect(db)
    rows = [dict(r) for r in con.execute("SELECT * FROM audit ORDER BY id DESC LIMIT ?", (limit,))]
    con.close()
    return rows[::-1]


# ================================================================== идемпотентность
def fingerprint(tool: str, args: dict) -> str:
    """Канонический отпечаток вызова: имя + аргументы с отсортированными ключами."""
    canon = json.dumps(
        {"tool": tool, "args": args}, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


@contextmanager
def _tx(con: sqlite3.Connection):
    con.execute("BEGIN IMMEDIATE")
    try:
        yield
        con.execute("COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        raise


# ================================================================== инструменты (функции)
def _order(con, order_id: str) -> sqlite3.Row:
    row = con.execute(
        "SELECT * FROM orders WHERE order_id = ?", (order_id.strip().upper(),)
    ).fetchone()
    if row is None:
        raise ToolError(
            "not_found",
            f"Заказа {order_id} нет.",
            hint="Попроси клиента проверить номер (вид A-1234). Не угадывай номер.",
        )
    return row


def get_order(order_id: str, *, db: str | None = None) -> dict:
    """READ. Статус, состав, дата доставки и сумма заказа — одним вызовом (консолидация, § 1.2)."""
    con = connect(db)
    try:
        o = _order(con, order_id)
        n_ret = con.execute(
            "SELECT COUNT(*) FROM returns WHERE order_id = ?", (o["order_id"],)
        ).fetchone()[0]
        return {
            "order_id": o["order_id"],
            "status": o["status"],
            "eta": o["eta"],
            "items": json.loads(o["items"]),
            "total_rub": o["total_rub"],
            "returns": n_ret,
            "can_cancel": o["status"] == "packed",
            "can_return": o["status"] in ("shipped", "delivered"),
        }
    finally:
        con.close()


def search_products(
    query: str,
    max_price: int | None = None,
    limit: int = 3,
    response_format: Literal["concise", "detailed"] = "concise",
    *,
    db: str | None = None,
) -> dict:
    """READ. Поиск по каталогу; ответ компактный по умолчанию, detailed — по запросу (§ 1.4)."""
    con = connect(db)
    try:
        q = query.strip().lower()
        rows = [dict(r) for r in con.execute("SELECT * FROM products ORDER BY price")]
    finally:
        con.close()
    hits = [r for r in rows if q in r["tags"] or q in r["name"].lower() or r["tags"] in q]
    if max_price is not None:
        hits = [r for r in hits if r["price"] <= max_price]
    limit = max(1, min(limit, 10))
    shown = hits[:limit]
    if response_format == "concise":
        items = [
            {"name": r["name"], "price_rub": r["price"], "in_stock": r["stock"] > 0} for r in shown
        ]
    else:
        items = [
            {
                "sku": r["sku"],
                "name": r["name"],
                "price_rub": r["price"],
                "stock": r["stock"],
                "description": r["description"],
                "tags": r["tags"].split(","),
            }
            for r in shown
        ]
    out: dict[str, Any] = {
        "query": query,
        "max_price": max_price,
        "total": len(hits),
        "items": items,
    }
    if len(hits) > len(shown):
        out["truncated"] = True
        out["hint"] = (
            f"Показаны {len(shown)} из {len(hits)}. Уточни запрос или max_price, "
            f"либо попроси limit побольше (до 10)."
        )
    return out


def create_return(
    order_id: str,
    reason: str,
    *,
    idempotency_key: str,
    dry_run: bool = False,
    actor: str = "agent",
    db: str | None = None,
) -> dict:
    """WRITE (additive, деньги!). Идемпотентен по ключу; dry_run показывает, что произойдёт (§ 1.7–1.8)."""
    args = {"order_id": order_id, "reason": reason}
    con = connect(db)
    try:
        if dry_run:
            o = _order(con, order_id)
            if o["status"] not in ("shipped", "delivered"):
                raise ToolError(
                    "not_allowed",
                    "Возврат возможен только для отправленного или доставленного заказа.",
                )
            amount = o["total_rub"]
            preview = {
                "dry_run": True,
                "would_create": {"order_id": o["order_id"], "reason": reason, "amount_rub": amount},
                "side_effects": [f"возврат {amount} ₽ на карту клиента", "курьер заберёт товар"],
            }
            audit(actor, "create_return", args, dry_run=True, outcome="preview", db=db)
            return preview
        fp = fingerprint("create_return", args)
        with _tx(con):
            prev = con.execute(
                "SELECT fingerprint, result FROM idempotency WHERE key = ?", (idempotency_key,)
            ).fetchone()
            if prev is not None:
                if prev["fingerprint"] != fp:
                    raise ToolError(
                        "conflict",
                        "Ключ идемпотентности уже использован с другими аргументами.",
                        hint="Это ошибка вызывающего кода, а не модели: новый вызов — новый ключ.",
                    )
                res = json.loads(prev["result"])
                res["already_applied"] = True  # аналог заголовка Idempotent-Replayed
                return res
            o = _order(con, order_id)
            if o["status"] not in ("shipped", "delivered"):
                raise ToolError(
                    "not_allowed",
                    f"Заказ {o['order_id']} в статусе {o['status']} — возврат невозможен.",
                    hint="Заказ ещё не отправлен: его можно отменить через cancel_order.",
                )
            amount = o["total_rub"]
            n = con.execute("SELECT COUNT(*) FROM returns").fetchone()[0]
            rid = f"R-{n + 1:03d}"
            con.execute(
                "INSERT INTO returns VALUES (?,?,?,?,?)",
                (rid, o["order_id"], reason, amount, time.time()),
            )
            res = {
                "return_id": rid,
                "order_id": o["order_id"],
                "reason": reason,
                "amount_rub": amount,
                "next_step": "курьер заберёт товар в течение 3 дней, деньги вернутся за 5–7 дней",
                "already_applied": False,
            }
            con.execute(
                "INSERT INTO idempotency VALUES (?,?,?,?)",
                (idempotency_key, fp, json.dumps(res, ensure_ascii=False), time.time()),
            )
        audit(actor, "create_return", args, dry_run=False, outcome="ok", detail=rid, db=db)
        return res
    finally:
        con.close()


def cancel_order(
    order_id: str,
    *,
    idempotency_key: str,
    dry_run: bool = False,
    actor: str = "agent",
    db: str | None = None,
) -> dict:
    """DESTRUCTIVE (необратимо). Отмена возможна только для статуса packed."""
    args = {"order_id": order_id}
    con = connect(db)
    try:
        if dry_run:
            o = _order(con, order_id)
            if o["status"] != "packed":
                raise ToolError(
                    "not_allowed",
                    "Отменить можно только собранный заказ.",
                    hint="Для доставленного заказа оформи возврат (create_return).",
                )
            audit(actor, "cancel_order", args, dry_run=True, outcome="preview", db=db)
            return {
                "dry_run": True,
                "would_cancel": o["order_id"],
                "refund_rub": o["total_rub"],
                "irreversible": True,
            }
        fp = fingerprint("cancel_order", args)
        with _tx(con):
            prev = con.execute(
                "SELECT fingerprint, result FROM idempotency WHERE key = ?", (idempotency_key,)
            ).fetchone()
            if prev is not None:
                if prev["fingerprint"] != fp:
                    raise ToolError(
                        "conflict",
                        "Ключ идемпотентности уже использован с другими аргументами.",
                        hint="Это ошибка вызывающего кода, а не модели: новый вызов — новый ключ.",
                    )
                res = json.loads(prev["result"])
                res["already_applied"] = True
                return res
            o = _order(con, order_id)
            if o["status"] != "packed":
                raise ToolError(
                    "not_allowed",
                    f"Заказ {o['order_id']} в статусе {o['status']} — отмена невозможна.",
                    hint="Отменить можно только собранный, но не отправленный заказ. "
                    "Для доставленного заказа оформи возврат (create_return).",
                )
            con.execute(
                "UPDATE orders SET status = 'cancelled' WHERE order_id = ?", (o["order_id"],)
            )
            res = {
                "order_id": o["order_id"],
                "status": "cancelled",
                "refund_rub": o["total_rub"],
                "already_applied": False,
            }
            con.execute(
                "INSERT INTO idempotency VALUES (?,?,?,?)",
                (idempotency_key, fp, json.dumps(res, ensure_ascii=False), time.time()),
            )
        audit(actor, "cancel_order", args, dry_run=False, outcome="ok", db=db)
        return res
    finally:
        con.close()


def returns_for(order_id: str, db: str | None = None) -> list[dict]:
    con = connect(db)
    rows = [
        dict(r)
        for r in con.execute("SELECT * FROM returns WHERE order_id = ?", (order_id.upper(),))
    ]
    con.close()
    return rows


def count_returns(db: str | None = None) -> int:
    con = connect(db)
    n = con.execute("SELECT COUNT(*) FROM returns").fetchone()[0]
    con.close()
    return n


# ================================================================== схемы аргументов (то, что видит МОДЕЛЬ)
class GetOrderArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: str = Field(description="Номер заказа вида A-1042")


class SearchProductsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(description="Что ищет клиент, 1–3 слова, например 'наушники'")
    max_price: int | None = Field(
        default=None, description="Верхняя граница цены в рублях; null, если не задана"
    )
    limit: int = Field(default=3, description="Сколько товаров вернуть, 1–10")
    response_format: Literal["concise", "detailed"] = Field(
        default="concise",
        description="concise — имя, цена, наличие; detailed — плюс sku, остаток, описание",
    )


class CreateReturnArgs(BaseModel):
    # idempotency_key и dry_run модель НЕ заполняет: их подставляет runtime (§ 1.3 «не заставляйте модель
    # заполнять то, что знает код»)
    model_config = ConfigDict(extra="forbid")
    order_id: str = Field(description="Номер заказа вида A-1042")
    reason: Literal["damaged", "not_suitable", "wrong_item", "other"] = Field(
        description="Причина возврата"
    )


class CancelOrderArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: str = Field(description="Номер заказа вида A-1042")


# ================================================================== ToolSpec: контракт инструмента
Effect = Literal["read", "write", "destructive"]
Risk = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str  # это промпт (лекция 1, § 3.11)
    args: type[BaseModel]  # схема для модели + валидатор входа
    fn: Callable[..., dict]
    effect: Effect  # read | write (additive) | destructive
    risk: Risk  # low | medium | high: деньги, необратимость, права
    idempotent: bool  # безопасно ли повторить вызов с теми же аргументами
    open_world: bool = False  # ходит ли во внешний мир (почта, веб, чужие API)
    timeout_s: float = 5.0
    max_output_chars: int = 2000
    runtime_args: tuple[str, ...] = ()  # что подставляет runtime, а не модель

    @property
    def needs_approval(self) -> bool:  # политика по умолчанию; её можно ужесточить, но не ослабить
        return self.effect != "read" or self.risk == "high"

    def openai(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args.model_json_schema(),
            },
        }

    def mcp_annotations(self) -> dict:  # тот же словарь, что ToolAnnotations в MCP (§ 1.6, § 2.7)
        return {
            "readOnlyHint": self.effect == "read",
            "destructiveHint": self.effect == "destructive",
            "idempotentHint": self.idempotent,
            "openWorldHint": self.open_world,
        }


REGISTRY: dict[str, ToolSpec] = {
    s.name: s
    for s in [
        ToolSpec(
            "get_order",
            "Возвращает статус, состав, дату доставки, сумму заказа и можно ли его отменить или вернуть. "
            "Только чтение. Используй, когда клиент спрашивает о конкретном заказе и назвал номер вида A-1234. "
            "Если номера нет — не угадывай, спроси.",
            GetOrderArgs,
            get_order,
            effect="read",
            risk="low",
            idempotent=True,
        ),
        ToolSpec(
            "search_products",
            "Ищет товары в каталоге по запросу и необязательной верхней границе цены в рублях. По умолчанию "
            "возвращает 3 товара в кратком виде и общее число найденных. Используй для вопросов о товарах и ценах. "
            "Не используй для вопросов о заказах — для этого есть get_order.",
            SearchProductsArgs,
            search_products,
            effect="read",
            risk="low",
            idempotent=True,
        ),
        ToolSpec(
            "create_return",
            "Оформляет возврат по доставленному или отправленному заказу и возвращает деньги клиенту (изменяет "
            "данные!). Используй, только когда клиент явно просит возврат и назвал номер заказа. Перед этим "
            "проверь заказ через get_order. Для заказа, который ещё не отправлен, используй cancel_order.",
            CreateReturnArgs,
            create_return,
            effect="write",
            risk="high",
            idempotent=True,
            runtime_args=("idempotency_key", "dry_run"),
        ),
        ToolSpec(
            "cancel_order",
            "Отменяет заказ в статусе packed (необратимо!). Используй, только когда клиент явно просит отменить "
            "ещё не отправленный заказ. Для доставленного заказа используй create_return.",
            CancelOrderArgs,
            cancel_order,
            effect="destructive",
            risk="high",
            idempotent=True,
            runtime_args=("idempotency_key", "dry_run"),
        ),
    ]
}
