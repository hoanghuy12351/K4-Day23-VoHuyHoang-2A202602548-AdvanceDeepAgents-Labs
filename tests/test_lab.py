"""Offline contract tests for the Deep Research lab."""

import json
import io
import os
import tempfile
import unittest
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import check_citations
import agents
import finalize_citations
import research
import tools


SOURCES = [
    {"n": 1, "url": "https://arxiv.org/abs/2501.00001", "source": "arxiv"},
    {"n": 2, "url": "https://huggingface.co/papers/2501.00002", "source": "hf-search"},
    {"n": 3, "url": "https://example.org/research", "source": "web"},
]
REPORT = """# Survey

## TL;DR
Evidence [1, 2] and more evidence [3].
```python
example = "[99]"
```
A Markdown link [77](https://example.org) is not a citation.

## References
[1] Paper A. arxiv. https://arxiv.org/abs/2501.00001 (2025-01-01)
[2] Paper B. hf-search. https://huggingface.co/papers/2501.00002 (2025-01-02)
[3] Web. web. https://example.org/research (2025-01-03)
"""


class CitationTests(unittest.TestCase):
    def test_valid_grouped_citations_ignore_code_and_links(self):
        self.assertEqual(check_citations.check(REPORT, SOURCES), [])

    def test_wrong_reference_and_duplicate_url_are_rejected(self):
        broken = [dict(source) for source in SOURCES]
        broken[1]["url"] = broken[0]["url"]
        problems = check_citations.check(REPORT, broken)
        self.assertTrue(any("repeats url" in problem for problem in problems))
        self.assertTrue(any("does not match" in problem for problem in problems))

    def test_missing_reference_is_rejected(self):
        problems = check_citations.check(REPORT.split("[3] Web. web.")[0], SOURCES)
        self.assertTrue(any("source [3] needs exactly one reference" in problem
                            for problem in problems))

    def test_source_family_must_match_its_url(self):
        broken = [dict(source) for source in SOURCES]
        broken[1]["url"] = "https://proceedings.mlr.press/v162/example.html"
        problems = check_citations.check(REPORT, broken)
        self.assertTrue(any("labeled hf-search" in problem for problem in problems))

    def test_finalizer_generates_missing_references(self):
        body = REPORT.split("## References", 1)[0]
        report, sources, problems = finalize_citations.finalize(body, SOURCES)
        self.assertEqual(problems, [])
        self.assertIn("## References", report)
        self.assertEqual(check_citations.check(report, sources), [])


class ToolTests(unittest.TestCase):
    def test_retry_after_and_nonretryable_errors(self):
        calls = []

        def transient():
            calls.append(1)
            if len(calls) == 1:
                raise tools.RetryableError("limited", retry_after=5)
            return "ok"

        with patch.object(tools.time, "sleep") as sleep:
            self.assertEqual(tools.with_retry(transient, cap=10), "ok")
        sleep.assert_called_once_with(5.0)
        self.assertEqual(len(calls), 2)

        with patch.object(tools.time, "sleep") as sleep:
            with self.assertRaises(ValueError):
                tools.with_retry(lambda: (_ for _ in ()).throw(ValueError("bug")))
        sleep.assert_not_called()

    def test_no_sleep_after_final_attempt(self):
        with patch.object(tools.time, "sleep") as sleep:
            with self.assertRaises(tools.RetryableError):
                tools.with_retry(lambda: (_ for _ in ()).throw(tools.RetryableError("429")),
                                 attempts=1)
        sleep.assert_not_called()

    def test_arxiv_normalizes_id_and_url(self):
        xml = b"""<feed xmlns="http://www.w3.org/2005/Atom"><entry>
        <id>http://arxiv.org/abs/2501.00001v2</id>
        <published>2025-01-01T12:00:00Z</published>
        <title> A   Paper </title><summary> A\n summary </summary>
        </entry></feed>"""
        with patch.object(tools, "_request", return_value=SimpleNamespace(content=xml)), \
             patch.object(tools.time, "sleep"):
            raw = tools.arxiv_search.invoke({"query": "world model", "max_results": 1})
        paper = json.loads(raw)[0]
        self.assertEqual(paper["id"], "2501.00001")
        self.assertEqual(paper["url"], "https://arxiv.org/abs/2501.00001")

    def test_exa_rate_limit_in_metadata_is_retried(self):
        rate = {"result": {"_meta": {"isRateLimited": True}, "content": []}}
        ok = {"result": {"content": [{"type": "text", "text": "A useful page"}]}}
        responses = [SimpleNamespace(text="data: " + json.dumps(rate)),
                     SimpleNamespace(text="data: " + json.dumps(ok))]
        with patch.object(tools, "_request", side_effect=responses) as request, \
             patch.object(tools.time, "sleep"):
            text = tools._exa_call("web_search_exa", {"query": "topic", "objective": "find"})
        self.assertEqual(text, "A useful page")
        self.assertEqual(request.call_count, 2)
        self.assertFalse(tools._is_rate_limited({"isRateLimited": False}, "A useful page"))

    def test_hugging_face_records_are_filtered_and_summarized(self):
        items = [
            {"paper": {"id": "2501.00001", "title": "World Models",
                       "summary": "Long abstract", "ai_summary": "Short summary",
                       "upvotes": 8, "publishedAt": "2025-01-01T10:00:00"}},
            {"paper": {"id": "2501.00002", "title": "Unrelated",
                       "summary": "Other topic", "upvotes": 20}},
        ]
        response = SimpleNamespace(json=lambda: items)
        with patch.object(tools, "_request", return_value=response):
            daily = json.loads(tools.hf_daily_papers.invoke({"keyword": "world"}))
            search = json.loads(tools.hf_search_papers.invoke({"query": "world"}))
        self.assertEqual([item["id"] for item in daily], ["2501.00001"])
        self.assertEqual(search[0]["summary"], "Short summary")

    def test_exa_error_never_returns_secret(self):
        secret = "test-key-for-local-redaction"
        with patch.dict(os.environ, {"EXA_API_KEY": secret}), \
             patch.object(tools, "_request", side_effect=ValueError(f"URL contains {secret}")):
            result = tools._exa_call("web_search_exa", {"query": "topic"})
        self.assertTrue(result.startswith("ERROR:"))
        self.assertNotIn(secret, result)


class RunnerTests(unittest.TestCase):
    def test_limits_raise_instead_of_ending_without_report(self):
        limits = agents._limits(3, 5)
        self.assertEqual([limit.exit_behavior for limit in limits], ["error", "error"])

    def test_slug_cannot_escape_reports(self):
        slug = research.slugify("../../x")
        self.assertEqual(slug, "x")
        self.assertNotIn("/", slug)

    def test_save_outputs_preserves_sandbox_bytes(self):
        report = REPORT.encode("utf-8")
        sources = json.dumps(SOURCES).encode("utf-8")
        files = {research.REPORT_PATH: report, research.SOURCES_PATH: sources}
        messages = [{"tool_calls": [{"name": "task"}] * 3,
                     "usage_metadata": {"input_tokens": 10, "output_tokens": 5}}]
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(research, "download", return_value=files):
            destination = Path(temporary) / "reports"
            path = research.save_outputs(object(), "survey", messages, 2.0, "model",
                                         reports_dir=destination)
            self.assertEqual(path.read_bytes(), report)
            self.assertEqual((destination / "survey.sources.json").read_bytes(), sources)
            meta = json.loads((destination / "survey.meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["subagent_calls"], 3)
            self.assertEqual(len(meta["source_families"]), 3)

    def test_invalid_download_writes_nothing(self):
        files = {research.REPORT_PATH: b"invalid", research.SOURCES_PATH: b"[]"}
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(research, "download", return_value=files):
            destination = Path(temporary) / "reports"
            with self.assertRaises(RuntimeError):
                research.save_outputs(object(), "topic", [], 0, "model",
                                      reports_dir=destination)
            self.assertFalse(destination.exists())

    def test_main_finalizes_then_validates_before_saving(self):
        commands = []

        def execute(command):
            commands.append(command)
            if research.FINALIZER_PATH in command:
                return SimpleNamespace(exit_code=0, output="FINALIZED: 3 sources")
            if research.VALIDATOR_PATH in command:
                return SimpleNamespace(exit_code=0, output="OK: 3 sources")
            return SimpleNamespace(exit_code=0, output="")

        backend = SimpleNamespace(execute=execute)
        files = {research.REPORT_PATH: b"Body [1]",
                 research.SOURCES_PATH: json.dumps(SOURCES).encode()}
        agent = SimpleNamespace(invoke=lambda *_args, **_kwargs: {"messages": []})
        with patch.object(research, "make_model", return_value=object()), \
             patch.object(research, "open_sandbox", return_value=nullcontext(backend)), \
             patch.object(research, "upload"), \
             patch.object(research, "build_lead_agent", return_value=agent), \
             patch.object(research, "download", return_value=files), \
             patch.object(research, "save_outputs", return_value=Path("reports/example.md")) as save, \
             redirect_stdout(io.StringIO()):
            self.assertEqual(research.main("example"), 0)
        self.assertEqual(commands[1:], [f"python3 {research.FINALIZER_PATH}",
                                         f"python3 {research.VALIDATOR_PATH}"])
        save.assert_called_once()

    def test_main_missing_report_fails_before_finalization(self):
        commands = []

        def execute(command):
            commands.append(command)
            return SimpleNamespace(exit_code=1 if "test -s" in command else 0,
                                   output="")

        backend = SimpleNamespace(execute=execute)
        files = {research.REPORT_PATH: None, research.SOURCES_PATH: b"[{}]"}
        agent = SimpleNamespace(invoke=lambda *_args, **_kwargs: {"messages": []})
        error = io.StringIO()
        with patch.object(research, "make_model", return_value=object()), \
             patch.object(research, "open_sandbox", return_value=nullcontext(backend)), \
             patch.object(research, "upload"), \
             patch.object(research, "build_lead_agent", return_value=agent), \
             patch.object(research, "download", return_value=files), \
             redirect_stderr(error):
            self.assertEqual(research.main("example"), 1)
        self.assertEqual(len(commands), 2)
        self.assertIn(research.MISPLACED_REPORT_PATH, commands[1])
        self.assertIn("agent ended without required sandbox file", error.getvalue())

    def test_main_recovers_observed_report_path_inside_sandbox(self):
        commands = []

        def execute(command):
            commands.append(command)
            if research.FINALIZER_PATH in command:
                return SimpleNamespace(exit_code=0, output="FINALIZED: 3 sources")
            if research.VALIDATOR_PATH in command:
                return SimpleNamespace(exit_code=0, output="OK: 3 sources")
            return SimpleNamespace(exit_code=0, output="")

        backend = SimpleNamespace(execute=execute)
        source_bytes = json.dumps(SOURCES).encode()
        missing = {research.REPORT_PATH: None, research.SOURCES_PATH: source_bytes}
        recovered = {research.REPORT_PATH: b"Body [1]", research.SOURCES_PATH: source_bytes}
        agent = SimpleNamespace(invoke=lambda *_args, **_kwargs: {"messages": []})
        output = io.StringIO()
        with patch.object(research, "make_model", return_value=object()), \
             patch.object(research, "open_sandbox", return_value=nullcontext(backend)), \
             patch.object(research, "upload"), \
             patch.object(research, "build_lead_agent", return_value=agent), \
             patch.object(research, "download", side_effect=[missing, recovered, recovered]), \
             patch.object(research, "save_outputs", return_value=Path("reports/example.md")), \
             redirect_stdout(output):
            self.assertEqual(research.main("example"), 0)
        self.assertIn(research.MISPLACED_REPORT_PATH, commands[1])
        self.assertEqual(commands[2:], [f"python3 {research.FINALIZER_PATH}",
                                         f"python3 {research.VALIDATOR_PATH}"])
        self.assertIn("Recovered report", output.getvalue())

    def test_main_repairs_missing_source_family_once_in_same_sandbox(self):
        commands, prompts = [], []

        def execute(command):
            commands.append(command)
            if research.FINALIZER_PATH in command:
                return SimpleNamespace(exit_code=0, output="FINALIZED: sources")
            if research.VALIDATOR_PATH in command:
                return SimpleNamespace(exit_code=0, output="OK: sources")
            return SimpleNamespace(exit_code=0, output="")

        def invoke(state, **_kwargs):
            prompts.append(state["messages"][0]["content"])
            return {"messages": [{"tool_calls": [{"name": "task"}]}]}

        two = json.dumps(SOURCES[:2]).encode()
        three = json.dumps(SOURCES).encode()
        first = {research.REPORT_PATH: b"Body", research.SOURCES_PATH: two}
        second = {research.REPORT_PATH: b"Body", research.SOURCES_PATH: three}
        backend = SimpleNamespace(execute=execute)
        agent = SimpleNamespace(invoke=invoke)
        output = io.StringIO()
        with patch.object(research, "make_model", return_value=object()), \
             patch.object(research, "open_sandbox", return_value=nullcontext(backend)), \
             patch.object(research, "upload"), \
             patch.object(research, "build_lead_agent", return_value=agent), \
             patch.object(research, "download", side_effect=[first, first, second, second]), \
             patch.object(research, "save_outputs", return_value=Path("reports/example.md")) as save, \
             redirect_stdout(output):
            self.assertEqual(research.main("example"), 0)
        self.assertEqual(len(prompts), 2)
        self.assertIn("currently cite only", prompts[1])
        self.assertIn("Only 2 source families", output.getvalue())
        self.assertEqual(sum(research.FINALIZER_PATH in command for command in commands), 2)
        self.assertEqual(len(save.call_args.args[2]), 2)

    def test_main_stops_after_one_failed_family_repair(self):
        calls = []

        def execute(command):
            if research.FINALIZER_PATH in command:
                return SimpleNamespace(exit_code=0, output="FINALIZED: sources")
            if research.VALIDATOR_PATH in command:
                return SimpleNamespace(exit_code=0, output="OK: sources")
            return SimpleNamespace(exit_code=0, output="")

        def invoke(*_args, **_kwargs):
            calls.append(1)
            return {"messages": []}

        two = json.dumps(SOURCES[:2]).encode()
        files = {research.REPORT_PATH: b"Body", research.SOURCES_PATH: two}
        error = io.StringIO()
        with patch.object(research, "make_model", return_value=object()), \
             patch.object(research, "open_sandbox", return_value=nullcontext(
                 SimpleNamespace(execute=execute))), \
             patch.object(research, "upload"), \
             patch.object(research, "build_lead_agent", return_value=SimpleNamespace(invoke=invoke)), \
             patch.object(research, "download", return_value=files), \
             patch.object(research, "save_outputs") as save, \
             redirect_stderr(error), redirect_stdout(io.StringIO()):
            self.assertEqual(research.main("example"), 1)
        self.assertEqual(len(calls), 2)
        save.assert_not_called()
        self.assertIn("after one repair pass", error.getvalue())


if __name__ == "__main__":
    unittest.main()
