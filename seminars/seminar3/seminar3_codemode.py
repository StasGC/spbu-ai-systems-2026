"""Code mode over MCP for an open-weight model (sketch). Python SDK v2 (`mcp>=2.2`)."""

import asyncio
import hashlib
import json
import sys
from asyncio.subprocess import PIPE

PY = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "array": "list",
    "object": "dict",
}
HIDDEN = {"idempotency_key", "dry_run"}  # host-owned args, as in the direct-call host


def stub(t) -> str:
    """MCP tool -> typed Python function. The same text goes into the prompt and into the sandbox."""
    req, props = set(t.input_schema.get("required", [])), t.input_schema.get("properties", {})
    props = {k: v for k, v in props.items() if k not in HIDDEN}

    def python_type(schema):
        if "enum" in schema:
            return f"Literal{schema['enum']!r}"
        return PY.get(schema.get("type"), "object")

    args = ", ".join(
        f"{k}: {python_type(v)}" + ("" if k in req else " = None") for k, v in props.items()
    )
    doc = (t.description or "") + "".join(
        f"\n{k}: {v['description']}" for k, v in props.items() if "description" in v
    )
    return (
        f"def {t.name}({'*, ' + args if args else ''}) -> dict:\n"
        f"    {json.dumps(doc, ensure_ascii=False)}\n"
        f"    return _call({t.name!r}, {{k: v for k, v in locals().items() if v is not None}})\n"
    )


PRELUDE = r"""import json, sys, io
from typing import Literal
_proto, sys.stdout = sys.stdout, io.StringIO()  # print() -> buffer for the model; real stdout = protocol
def _call(name, args):
    _proto.write(json.dumps({"call": name, "args": args}) + "\n"); _proto.flush()
    r = json.loads(sys.stdin.readline())
    if "error" in r: raise RuntimeError(r["error"])  # MCP isError -> exception, model code can try/except
    return r["result"]
"""
EPILOGUE = '\n_proto.write(json.dumps({"done": sys.stdout.getvalue()}) + "\\n")\n'


async def execute_code(client, tools: dict, code: str, approve, timeout: float = 30.0) -> str:
    """The ONE tool the model sees. Every call from the sandbox comes back to this MCP client."""
    run = hashlib.sha256(code.encode()).hexdigest()[
        :12
    ]  # same script re-run -> same keys -> no double writes
    src = PRELUDE + "".join(map(stub, tools.values())) + code + EPILOGUE
    # shortcut: `python -I` is NOT isolation; in prod run it in `docker run --network none --read-only -m 256m` / nsjail
    p = await asyncio.create_subprocess_exec(
        sys.executable, "-I", "-c", src, stdin=PIPE, stdout=PIPE, stderr=PIPE, limit=2**24
    )
    try:
        async with asyncio.timeout(timeout):
            for seq in range(10_000):  # cap on calls per script
                line = await p.stdout.readline()
                if not line:
                    break
                msg = json.loads(line)  # the sandbox is untrusted: validate everything it sends
                if "done" in msg:
                    return msg["done"][-4000:]  # only this reaches the model context
                t = tools.get(msg.get("call"))
                args = dict(msg.get("args") or {})
                if t is None:
                    reply = {"error": f"unknown tool {msg.get('call')!r}"}
                elif not (t.annotations and t.annotations.read_only_hint) and not approve(
                    t.name, args
                ):
                    reply = {"error": "not_approved"}  # same per-call HITL policy as direct calls
                else:
                    if "idempotency_key" in t.input_schema.get("properties", {}):
                        args["idempotency_key"] = f"{run}:{seq}"
                    r = await client.call_tool(t.name, args)
                    text = "\n".join(c.text for c in r.content if getattr(c, "text", None))
                    res = r.structured_content  # typed only if the server declared outputSchema
                    if res is None:
                        try:
                            res = json.loads(text)
                        except ValueError:
                            res = text
                    reply = {"error": text} if r.is_error else {"result": res}
                p.stdin.write((json.dumps(reply, ensure_ascii=False) + "\n").encode())
                await p.stdin.drain()
    except TimeoutError:
        return f"ERROR: timeout after {timeout}s"
    finally:
        if p.returncode is None:
            p.kill()
        await p.wait()
    return "ERROR:\n" + (await p.stderr.read()).decode()[-2000:]  # traceback -> model self-corrects
