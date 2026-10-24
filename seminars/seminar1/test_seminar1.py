"""Регрессии реальных ошибок семинара. Запуск: python seminars/seminar1/test_seminar1.py.

Исполняет код ноутбука на mock в отдельном временном каталоге, затем проверяет
пограничные случаи gateway, dispatcher и scorer. Живая модель не используется.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from seminar1_mock import FakeOpenAI, rate_limit_error


def response(calls=None, text=None, finish=None):
    message = {"role": "assistant", "content": text}
    if calls is not None:
        message["tool_calls"] = [
            {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": args}}
            for i, (name, args) in enumerate(calls)
        ]
    choice = {"message": message}
    if finish is not None:
        choice["finish_reason"] = finish
    return {"choices": [choice]}


class SeminarRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="s1-regression-")
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.ns = {"__name__": "seminar1_under_test"}
        notebook = json.loads((HERE / "seminar1-function-calling.ipynb").read_text())
        cwd, previous = Path.cwd(), os.environ.get("S1_BACKEND")
        try:
            os.chdir(cls.tmp.name)
            os.environ["S1_BACKEND"] = "mock"
            with contextlib.redirect_stdout(io.StringIO()):
                for i, cell in enumerate(notebook["cells"]):
                    if cell["cell_type"] == "code":
                        exec(compile("".join(cell["source"]), f"notebook:cell-{i}", "exec"), cls.ns)
        finally:
            os.chdir(cwd)
            if previous is None:
                os.environ.pop("S1_BACKEND", None)
            else:
                os.environ["S1_BACKEND"] = previous

    def setUp(self):
        self.ns["TRACES"] = str(Path(self.tmp.name) / "case.jsonl")
        Path(self.ns["TRACES"]).write_text("")
        self.ns["RETURNS"].clear()
        self.ns["client"] = FakeOpenAI(script=[], latency_s=0)

    def calls(self, items):
        fake = FakeOpenAI(script=[response(calls=items)], latency_s=0)
        return fake.chat.completions.create(model="mock", messages=[]).choices[0].message.tool_calls

    def score(self, case, items):
        return self.ns["score_calls"](case, self.calls(items))

    def test_strict_schema_requires_nullable_price(self):
        schema = self.ns["SearchProductsArgs"]
        self.assertEqual(set(schema.model_json_schema()["required"]), {"query", "max_price"})
        schema.model_validate_json('{"query":"чайник","max_price":null}')
        with self.assertRaises(self.ns["ValidationError"]):
            schema.model_validate_json('{"query":"чайник"}')

    def test_schema_rejects_coercion_and_negative_price(self):
        for value in ('"5000"', 'true', '-1'):
            with self.subTest(value=value), self.assertRaises(self.ns["ValidationError"]):
                self.ns["SearchProductsArgs"].model_validate_json('{"query":"чайник","max_price":' + value + '}')

    def test_ticket_rejects_order_not_in_source(self):
        ticket = self.ns["ticket"].model_dump(mode="json")
        ticket["order_id"] = "A-9999"
        with self.assertRaises(self.ns["ValidationError"]):
            self.ns["SupportTicket"].model_validate_json(json.dumps(ticket), context={"text": "Заказ A-1042"})

    def test_length_never_executes_mutating_call(self):
        self.ns["client"] = FakeOpenAI(script=[response(
            calls=[("create_return", '{"order_id":"A-1044","reason":"damaged"}')], finish="length"
        )], latency_s=0)
        with self.assertRaises(self.ns["IncompleteResponse"]):
            self.ns["run"]("Вернуть A-1044", verbose=False)
        self.assertEqual(self.ns["RETURNS"], [])
        span = json.loads(Path(self.ns["TRACES"]).read_text())
        self.assertEqual(span["finish_reason"], "length")
        self.assertEqual(span["error"], "IncompleteResponse")

    def test_empty_response_is_not_success(self):
        self.ns["client"] = FakeOpenAI(script=[response(text=None, finish="stop")], latency_s=0)
        with self.assertRaises(self.ns["IncompleteResponse"]):
            self.ns["complete"]([])

    def test_exhausted_retry_is_logged(self):
        self.ns["client"] = FakeOpenAI(script=[rate_limit_error()], latency_s=0)
        with self.assertRaises(Exception) as raised:
            self.ns["complete"]([], attempts=1)
        self.assertEqual(type(raised.exception).__name__, "RateLimitError")
        span = json.loads(Path(self.ns["TRACES"]).read_text())
        self.assertEqual(span["attempts"], 1)
        self.assertEqual(span["error"], "RateLimitError")
        self.assertIsNone(span["cost_usd"])

    def test_retry_after_cannot_exceed_budget(self):
        fake = FakeOpenAI(script=[rate_limit_error(retry_after="120"), response(text="ok")], latency_s=0)
        self.ns["client"] = fake
        with self.assertRaises(TimeoutError):
            self.ns["complete"]([], budget_s=1)
        self.assertEqual(len(fake.calls), 1)

    def test_retry_after_http_date_and_invalid_header(self):
        self.assertEqual(self.ns["retry_delay"]("Wed, 21 Oct 2015 07:28:00 GMT", 0.3), 0)
        self.assertEqual(self.ns["retry_delay"]("not a date", 0.3), 0.3)

    def test_return_repeat_does_not_duplicate_action(self):
        first = self.ns["create_return"]("A-1044", "damaged")
        self.assertEqual(first, self.ns["create_return"]("A-1044", "damaged"))
        self.assertEqual(len(self.ns["RETURNS"]), 1)
        with self.assertRaises(self.ns["ToolError"]):
            self.ns["create_return"]("A-1044", "other")

    def test_duplicate_order_does_not_pass_parallel_case(self):
        case = self.ns["CASES"][8]
        result = self.score(case, [("get_order", '{"order_id":"A-1042"}')] * 2)
        self.assertTrue(result["tool_ok"])
        self.assertFalse(result["args_ok"])
        reverse = self.score(case, [("get_order", '{"order_id":"A-1043"}'),
                                    ("get_order", '{"order_id":"A-1042"}')])
        self.assertTrue(reverse["args_ok"])

    def test_unwanted_call_is_counted_and_validated(self):
        result = self.score(self.ns["CASES"][9], [("get_order", '{broken')])
        self.assertFalse(result["tool_ok"])
        self.assertEqual(result["extra_calls"], 1)
        self.assertEqual(result["json_count"], 0)

    def test_extra_call_and_null_field_fail_exact_match(self):
        result = self.score(self.ns["CASES"][0], [
            ("get_order", '{"order_id":"A-1042"}'), ("get_order", '{"order_id":"A-1043"}')])
        self.assertFalse(result["tool_ok"])
        self.assertFalse(result["args_ok"])
        result = self.score(self.ns["CASES"][0], [("get_order", '{"order_id":"A-1042","invented":null}')])
        self.assertFalse(result["args_ok"])
        self.assertEqual(result["schema_count"], 0)

    def test_scalar_json_does_not_crash_scorer(self):
        result = self.score(self.ns["CASES"][0], [("get_order", '42')])
        self.assertEqual(result["json_count"], 1)
        self.assertEqual(result["schema_count"], 0)
        self.assertFalse(result["args_ok"])

    def test_tool_bug_is_not_misreported_as_bad_model_arguments(self):
        def buggy_tool(order_id):
            self.ns["GetOrderArgs"].model_validate_json('{"wrong":"field"}')
        registry = {"get_order": (self.ns["GetOrderArgs"], buggy_tool, "read")}
        tc = self.calls([("get_order", '{"order_id":"A-1042"}')])[0]
        with self.assertRaises(self.ns["ValidationError"]):
            self.ns["execute_tool"](registry, tc)

    def test_mock_script_exhaustion_is_visible(self):
        with self.assertRaises(AssertionError):
            FakeOpenAI(script=[], latency_s=0).chat.completions.create(model="mock", messages=[])

    def test_eval_cases_have_distinct_trace_ids(self):
        # Полный прогон в setUpClass записан отдельно от per-test trace.
        spans = [json.loads(line) for line in (Path(self.tmp.name) / "traces.jsonl").read_text().splitlines()]
        ids = [s["trace_id"] for s in spans if s["trace_id"].startswith("t-eval-")]
        self.assertEqual(len(ids), 12)
        self.assertEqual(len(set(ids)), 12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
