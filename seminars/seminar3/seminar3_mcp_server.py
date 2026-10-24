"""
seminar3_mcp_server.py — MCP-сервер магазина «Лавка» (Python SDK `mcp` v2, спецификация 2026-07-28).

Запуск:
    python seminar3_mcp_server.py                       # stdio: так его запускает хост (ноутбук, Claude, IDE)
    mcp dev seminar3_mcp_server.py                      # MCP Inspector в браузере
    mcp run seminar3_mcp_server.py --transport streamable-http   # HTTP на http://localhost:8000/mcp

Правила stdio: stdout занят протоколом — никаких print(); логи только в stderr.
v1 SDK (mcp<2): вместо `from mcp.server import MCPServer` было `from mcp.server.fastmcp import FastMCP`.
"""

import json
import sys
from typing import Literal, TypedDict

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError as MCPToolError
from mcp.types import ToolAnnotations

import seminar3_store as store

mcp = MCPServer("lavka-store", instructions="Инструменты поддержки интернет-магазина «Лавка».")


def log(*a) -> None:  # stderr, а не stdout!
    print("[lavka-store]", *a, file=sys.stderr, flush=True)


class OrderInfo(TypedDict):  # -> outputSchema + structuredContent
    order_id: str
    status: str
    eta: str
    items: list[str]
    total_rub: int
    returns: int
    can_cancel: bool
    can_return: bool


def _call(fn, *args, **kwargs):
    """Ожидаемые ошибки -> isError: true с JSON внутри (модель увидит и исправится).
    Неожиданные исключения SDK сам превращает в isError без деталей — наружу ничего не утечёт."""
    try:
        return fn(*args, **kwargs)
    except store.ToolError as e:
        raise MCPToolError(json.dumps(e.to_dict(), ensure_ascii=False)) from e


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False))
def get_order(order_id: str) -> OrderInfo:
    """Возвращает статус, состав, дату доставки, сумму заказа и можно ли его отменить или вернуть.
    Только чтение. Используй, когда клиент спрашивает о конкретном заказе и назвал номер вида A-1234.
    Если номера нет — не угадывай, спроси."""
    log("get_order", order_id)
    return _call(store.get_order, order_id)


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False))
def search_products(
    query: str,
    max_price: int | None = None,
    limit: int = 3,
    response_format: Literal["concise", "detailed"] = "concise",
) -> dict:
    """Ищет товары в каталоге по запросу и необязательной верхней границе цены в рублях.
    По умолчанию возвращает 3 товара кратко и общее число найденных. Для вопросов о заказах не используй."""
    return _call(store.search_products, query, max_price, limit, response_format)


@mcp.tool(
    annotations=ToolAnnotations(
        read_only_hint=False, destructive_hint=False, idempotent_hint=True, open_world_hint=False
    )
)
def create_return(
    order_id: str,
    reason: Literal["damaged", "not_suitable", "wrong_item", "other"],
    idempotency_key: str,
    dry_run: bool = False,
) -> dict:
    """Оформляет возврат по доставленному или отправленному заказу и возвращает деньги (изменяет данные!).
    Повтор с тем же idempotency_key не создаёт второй возврат. dry_run=true — только показать, что произойдёт."""
    log("create_return", order_id, reason, idempotency_key, "dry" if dry_run else "")
    return _call(
        store.create_return,
        order_id,
        reason,
        idempotency_key=idempotency_key,
        dry_run=dry_run,
        actor="mcp",
    )


@mcp.tool(
    annotations=ToolAnnotations(
        read_only_hint=False, destructive_hint=True, idempotent_hint=True, open_world_hint=False
    )
)
def cancel_order(order_id: str, idempotency_key: str, dry_run: bool = False) -> dict:
    """Отменяет заказ в статусе packed (необратимо!). Для доставленного заказа используй create_return."""
    return _call(
        store.cancel_order, order_id, idempotency_key=idempotency_key, dry_run=dry_run, actor="mcp"
    )


if __name__ == "__main__":
    store.ensure_db()
    mcp.run()  # stdio по умолчанию
