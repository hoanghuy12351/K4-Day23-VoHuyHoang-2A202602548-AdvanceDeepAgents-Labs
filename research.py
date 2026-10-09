"""Run one Deep Research topic and save its validated sandbox artifacts."""

import json
import os
import re
import sys
import threading
import time
from collections import Counter
from pathlib import Path

from agents import (
    FINALIZER_PATH,
    REPORT_PATH,
    SOURCES_PATH,
    VALIDATOR_PATH,
    WORKDIR,
    build_lead_agent,
)
from check_citations import check
from model import make_model
from sandbox import download, open_sandbox, upload

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"
FAMILIES = {"arxiv", "hf-daily", "hf-search", "web"}
MISPLACED_REPORT_PATH = f"{WORKDIR}/research/report/report.md"


def slugify(topic):
    """Turn arbitrary user input into a short, safe filename component."""
    slug = re.sub(r"[^\w]+", "-", str(topic).casefold(), flags=re.UNICODE).strip("-_")
    slug = slug[:60].rstrip("-_") or "topic"
    if slug.upper() in {"CON", "PRN", "AUX", "NUL", *{f"COM{i}" for i in range(1, 10)},
                        *{f"LPT{i}" for i in range(1, 10)}}:
        slug = f"topic-{slug}"
    return slug


def build_prompt(topic):
    """Give the lead a concrete topic and remind it of its deliverables."""
    return (
        f"Research the topic: {topic.strip()}\n"
        f"Write the report body to exactly {REPORT_PATH} and sources to "
        f"exactly {SOURCES_PATH}. "
        "Produce an English thematic survey with verified inline citations. "
        "Delegate at least three independent subquestions, use at least three "
        "source families, and complete finalization and validation inside "
        "the sandbox before finishing."
    )


def build_family_repair_prompt(topic, families):
    """Ask the lead to repair a finalized report that lost a source family."""
    missing = sorted(FAMILIES - families)
    return (
        f"Continue the existing research for {topic!r} in this sandbox. "
        f"The finalized report at {REPORT_PATH} and sources at {SOURCES_PATH} "
        f"currently cite only these source families: {sorted(families)}. "
        f"The Lab requires at least three; try a relevant source from {missing}. "
        "Read the existing report and notes. Ask a researcher for more evidence "
        "if needed. Add a specific supported claim with an inline citation and "
        "the matching source entry; never add an uncited source or relabel one. "
        f"Run {FINALIZER_PATH} and {VALIDATOR_PATH} again and check that at "
        "least three families remain. Preserve the supported existing report."
    )


def summarize(messages, elapsed, model_name):
    """Count visible lead tool calls and token usage; subagent tokens are unavailable."""
    calls = Counter()
    input_tokens = output_tokens = 0
    for message in messages:
        tool_calls = getattr(message, "tool_calls", None)
        usage = getattr(message, "usage_metadata", None)
        if isinstance(message, dict):
            tool_calls = tool_calls or message.get("tool_calls", [])
            usage = usage or message.get("usage_metadata", {})
        for call in tool_calls or []:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            if name:
                calls[name] += 1
        if isinstance(usage, dict):
            input_tokens += int(usage.get("input_tokens") or 0)
            output_tokens += int(usage.get("output_tokens") or 0)
    return {
        "model": model_name,
        "elapsed_s": round(elapsed, 1),
        "subagent_calls": calls["task"],
        "tool_calls": dict(sorted(calls.items())),
        "tokens": {"input": input_tokens, "output": output_tokens},
    }


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Validate downloaded bytes before writing the report, sources and metadata."""
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report_bytes, sources_bytes = files.get(REPORT_PATH), files.get(SOURCES_PATH)
    if not report_bytes or not sources_bytes:
        raise RuntimeError("sandbox report.md or sources.json is missing or empty")
    try:
        report_text = report_bytes.decode("utf-8")
        sources = json.loads(sources_bytes.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError(f"sandbox report or sources.json is invalid: {exc}") from exc
    if not report_text.strip() or not isinstance(sources, list) or not sources:
        raise RuntimeError("sandbox report or sources.json is empty")
    problems = check(report_text, sources)
    if problems:
        raise RuntimeError("citation validation failed: " + "; ".join(problems[:5]))

    family_set = set()
    for source in sources:
        family, url = source.get("source"), source.get("url", "")
        if family not in FAMILIES:
            raise RuntimeError(f"invalid source family: {family!r}")
        if family == "arxiv" and not url.startswith("https://arxiv.org/abs/"):
            raise RuntimeError(f"arxiv source has a mismatched URL: {url}")
        if family in {"hf-daily", "hf-search"} and not url.startswith(
            "https://huggingface.co/papers/"
        ):
            raise RuntimeError(f"Hugging Face source has a mismatched URL: {url}")
        family_set.add(family)
    if len(family_set) < 3:
        raise RuntimeError(f"only {len(family_set)} source families remain after finalization: {sorted(family_set)}")
    metadata = {"topic": topic, **summarize(messages, elapsed, model_name),
                "n_sources": len(sources), "source_families": sorted(family_set)}
    if metadata["subagent_calls"] < 3:
        raise RuntimeError(f"only {metadata['subagent_calls']} researcher delegations were recorded")

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    stem = slugify(topic)
    report_path = reports_dir / f"{stem}.md"
    (reports_dir / f"{stem}.sources.json").write_bytes(sources_bytes)
    (reports_dir / f"{stem}.meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report_path.write_bytes(report_bytes)
    return report_path


def invoke_with_heartbeat(agent, prompt, started):
    """Show elapsed time while a synchronous agent invocation is in progress."""
    finished = threading.Event()

    def report_progress():
        while not finished.wait(60):
            minutes = (time.monotonic() - started) / 60
            print(f"Research still running after {minutes:.0f} min", file=sys.stderr, flush=True)

    threading.Thread(target=report_progress, daemon=True).start()
    try:
        return agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config={"recursion_limit": 1000},
        )
    finally:
        finished.set()


def main(topic):
    """Execute one bounded research run; return 0 on success, 1 on failure."""
    if not topic or not topic.strip():
        print('Usage: python research.py "your research topic"', file=sys.stderr)
        return 2
    try:
        model = make_model()
        model_name = os.getenv("LAB_MODEL", getattr(model, "model_name", type(model).__name__))
        started = time.monotonic()
        with open_sandbox() as backend:
            result = backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
            if result.exit_code != 0:
                raise RuntimeError(f"cannot prepare sandbox: {result.output[:300]}")
            upload(backend, {
                VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
                FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
            })
            agent = build_lead_agent(backend, model)
            messages = []
            families = set()
            for attempt in range(2):  # initial research plus one bounded repair
                prompt = build_prompt(topic) if attempt == 0 else build_family_repair_prompt(topic, families)
                state = invoke_with_heartbeat(agent, prompt, started)
                messages.extend(state.get("messages", []) if isinstance(state, dict) else [])
                artifacts = download(backend, [REPORT_PATH, SOURCES_PATH])
                if not artifacts.get(REPORT_PATH):
                    # Recover the observed path mix-up without changing the agent's
                    # report text; finalization and validation still run in sandbox.
                    recovery = backend.execute(
                        f"test -s {MISPLACED_REPORT_PATH} && cp {MISPLACED_REPORT_PATH} {REPORT_PATH}"
                    )
                    if recovery.exit_code == 0:
                        artifacts = download(backend, [REPORT_PATH, SOURCES_PATH])
                        if artifacts.get(REPORT_PATH):
                            print(f"Recovered report from {MISPLACED_REPORT_PATH} inside sandbox")
                missing = [path for path in (REPORT_PATH, SOURCES_PATH) if not artifacts.get(path)]
                if missing:
                    calls = summarize(messages, time.monotonic() - started, model_name)["tool_calls"]
                    last_reply = next(
                        (message for message in reversed(messages)
                         if getattr(message, "type", None) == "ai"), None
                    )
                    reply = " ".join(str(getattr(last_reply, "content", "")).split())[:240]
                    raise RuntimeError(
                        f"agent ended without required sandbox file(s) {missing}; "
                        f"lead tool calls: {calls}; last reply: {reply!r}"
                    )
                # Finalize deterministically in the sandbox, then validate.
                finalization = backend.execute(f"python3 {FINALIZER_PATH}")
                if finalization.exit_code != 0 or not finalization.output.startswith("FINALIZED:"):
                    raise RuntimeError(f"sandbox citation finalization failed: {finalization.output[:500]}")
                verdict = backend.execute(f"python3 {VALIDATOR_PATH}")
                if verdict.exit_code != 0 or not verdict.output.startswith("OK:"):
                    raise RuntimeError(f"sandbox citation check failed: {verdict.output[:500]}")
                finalized = download(backend, [SOURCES_PATH]).get(SOURCES_PATH)
                try:
                    sources = json.loads(finalized.decode("utf-8")) if finalized else []
                except (UnicodeDecodeError, ValueError) as exc:
                    raise RuntimeError(f"finalized sources.json is invalid: {exc}") from exc
                if not isinstance(sources, list) or not all(isinstance(item, dict) for item in sources):
                    raise RuntimeError("finalized sources.json is not a list of objects")
                families = FAMILIES & {item.get("source") for item in sources}
                if len(families) >= 3:
                    break
                if attempt == 0:
                    print(f"Only {len(families)} source families remain ({sorted(families)}); "
                          "asking lead for one repair pass")
            if len(families) < 3:
                raise RuntimeError(f"only {len(families)} source families remain after one repair pass: {sorted(families)}")
            path = save_outputs(backend, topic, messages, time.monotonic() - started,
                                model_name)
        print(f"Saved: {path}")
        return 0
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        for name in ("OPENAI_API_KEY", "LAB_API_KEY", "EXA_API_KEY"):
            secret = os.getenv(name, "")
            if secret:
                message = message.replace(secret, "[REDACTED]")
        print(f"FAILED: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
