"""Repair an existing generated report inside a fresh sandbox.

Usage: python repair_report.py <report stem>
The report and sources are replaced only after sandbox finalization and validation.
"""

import json
import os
import re
import sys
import time
from pathlib import Path

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, build_lead_agent
from check_citations import check
from model import make_model
from research import FINALIZER_SOURCE, REPORTS, VALIDATOR_SOURCE, summarize
from sandbox import download, open_sandbox, upload


def named_source_problems(report, sources):
    """Catch named methods whose own source is absent from the cited paragraph."""
    body = report.split("## References", 1)[0]
    markers = {}
    for source in sources:
        title = source.get("title", "")
        for marker in re.findall(r"\b(?:[A-Z][a-z]+[A-Z][A-Za-z0-9]*|[A-Z]{3,}[a-zA-Z0-9]*)\b", title):
            if marker not in {"LLM", "LLMs", "SLM", "SLMs", "GPT", "GPU", "CPU", "BERT"}:
                markers.setdefault(marker.casefold(), set()).add(source["n"])
    problems = []
    for paragraph in re.split(r"\n\s*\n", body):
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", paragraph)}
        for marker, numbers in markers.items():
            if len(numbers) == 1 and re.search(rf"\b{re.escape(marker)}\b", paragraph, re.I):
                expected = next(iter(numbers))
                if expected not in cited:
                    problems.append(f"{marker} appears without its source [{expected}]")
    return list(dict.fromkeys(problems))


def main(stem, issue):
    report_path = REPORTS / f"{stem}.md"
    sources_path = REPORTS / f"{stem}.sources.json"
    meta_path = REPORTS / f"{stem}.meta.json"
    for path in (report_path, sources_path, meta_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    original_report = report_path.read_bytes()
    original_sources = sources_path.read_bytes()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model = make_model()
    started = time.monotonic()
    with open_sandbox() as backend:
        prepared = backend.execute("mkdir -p /tmp/work/report /tmp/work/research")
        if prepared.exit_code != 0:
            raise RuntimeError(prepared.output)
        upload(backend, {
            REPORT_PATH: original_report,
            SOURCES_PATH: original_sources,
            VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
            FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
        })
        agent = build_lead_agent(backend, model)
        prompt = f"""Repair citations in the EXISTING generated report at {REPORT_PATH}.
Read {SOURCES_PATH} and compare every named method or paper in each paragraph
with the titles and evidence of its cited source numbers. Do not repeat the
original research plan. Preserve the report structure and supported claims.
Known defect to fix: {issue}
Remove any claim whose supporting source is absent, or retrieve a real source
and add it accurately. Do not merely renumber an unsupported citation.
Check ALL paragraphs for similar errors, including TL;DR and trends. The
source number in a structurally valid citation can still point to the wrong
paper. Run python3 {FINALIZER_PATH}, then python3 {VALIDATOR_PATH} inside the
sandbox. Keep at least three source families. End only after both pass."""
        state = agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config={"recursion_limit": 1000},
        )
        finalization = backend.execute(f"python3 {FINALIZER_PATH}")
        if finalization.exit_code != 0 or not finalization.output.startswith("FINALIZED:"):
            raise RuntimeError(f"sandbox finalization failed: {finalization.output[:500]}")
        verdict = backend.execute(f"python3 {VALIDATOR_PATH}")
        if verdict.exit_code != 0 or not verdict.output.startswith("OK:"):
            raise RuntimeError(f"sandbox citation check failed: {verdict.output[:500]}")
        files = download(backend, [REPORT_PATH, SOURCES_PATH])
        new_report, new_sources = files.get(REPORT_PATH), files.get(SOURCES_PATH)
        if not new_report or not new_sources:
            raise RuntimeError("repaired sandbox artifacts are missing")
        report_text = new_report.decode("utf-8")
        sources = json.loads(new_sources.decode("utf-8"))
        problems = check(report_text, sources) + named_source_problems(report_text, sources)
        if "self-speculative decoding" in report_text.casefold() and not any(
            "self-speculative decoding" in str(source.get("title", "")).casefold()
            for source in sources
        ):
            problems.append("self-speculative decoding claim lacks its own supporting paper")
        if problems:
            raise RuntimeError("repaired report still has problems: " + "; ".join(problems[:8]))
        families = {entry.get("source") for entry in sources}
        if len(families & {"arxiv", "hf-daily", "hf-search", "web"}) < 3:
            raise RuntimeError("repaired report lost the required source-family coverage")
        messages = state.get("messages", []) if isinstance(state, dict) else []
        repair = summarize(messages, time.monotonic() - started, type(model).__name__)
    report_path.write_bytes(new_report)
    sources_path.write_bytes(new_sources)
    meta["n_sources"] = len(sources)
    meta["source_families"] = sorted(families)
    repairs = meta.pop("citation_repairs", [])
    if "citation_repair" in meta:
        repairs.append(meta.pop("citation_repair"))
    repairs.append(repair)
    meta["citation_repairs"] = repairs
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Repaired: {report_path}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python repair_report.py <report stem> <known defect>")
    os.environ.setdefault("SANDBOX", "docker")
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
