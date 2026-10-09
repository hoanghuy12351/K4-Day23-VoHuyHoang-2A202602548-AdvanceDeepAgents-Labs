"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"


def check(report_text, sources):
    """Return structural citation problems; an empty list means valid."""
    problems = []
    if not isinstance(sources, list) or not sources:
        return ["no sources in sources.json"]

    by_number = {}
    seen_urls = set()
    for index, source in enumerate(sources, 1):
        if not isinstance(source, dict):
            problems.append(f"source {index} is not an object")
            continue
        number, url = source.get("n"), source.get("url")
        if type(number) is not int:
            problems.append(f"source {index} has a non-integer n")
        elif number < 1:
            problems.append(f"source {index} has a non-positive n")
        elif number in by_number:
            problems.append(f"source [{number}] is duplicated in sources.json")
        else:
            by_number[number] = source
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            problems.append(f"source {index} has an invalid url")
        elif url in seen_urls:
            problems.append(f"source {index} repeats url {url}")
        else:
            seen_urls.add(url)
        family = source.get("source")
        expected_prefix = {
            "arxiv": "https://arxiv.org/abs/",
            "hf-daily": "https://huggingface.co/papers/",
            "hf-search": "https://huggingface.co/papers/",
        }.get(family)
        if family not in {"arxiv", "hf-daily", "hf-search", "web"}:
            problems.append(f"source [{number}] has invalid family {family!r}")
        elif expected_prefix and isinstance(url, str) and not url.startswith(expected_prefix):
            problems.append(
                f"source [{number}] labeled {family} has URL {url!r}; "
                f"use the tool's canonical URL starting with {expected_prefix} "
                "or restore the actual source tool label"
            )
    if sorted(by_number) != list(range(1, len(sources) + 1)):
        problems.append("source numbers must be consecutive starting at 1")

    heading = re.search(r"(?m)^##[ \t]+References[ \t]*$", report_text)
    if heading is None:
        problems.append("missing ## References heading")
        body, references = report_text, ""
    else:
        body, references = report_text[:heading.start()], report_text[heading.end():]

    # Fenced and inline code are examples, not claims. Markdown links are not citations.
    body = re.sub(r"(?ms)^ {0,3}(```|~~~).*?^ {0,3}\1[^\n]*$", "", body)
    body = re.sub(r"`[^`\n]*`", "", body)
    citation = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\s*\()")
    cited = set()
    for match in citation.finditer(body):
        for part in re.split(r"\s*,\s*", match.group(1)):
            span = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
            if span:
                start, end = map(int, span.groups())
                if end < start or end - start > 200:
                    problems.append(f"invalid citation range [{part}]")
                    continue
                cited.update(range(start, end + 1))
            else:
                cited.add(int(part))

    for number in sorted(cited - by_number.keys()):
        problems.append(f"[{number}] cited but missing from sources.json")
    for number in sorted(by_number.keys() - cited):
        problems.append(f"source [{number}] never cited")

    reference_counts = {}
    for line in references.splitlines():
        match = re.match(r"^\s*\[(\d+)\](.*)$", line)
        if not match:
            continue
        number, detail = int(match.group(1)), match.group(2)
        reference_counts[number] = reference_counts.get(number, 0) + 1
        if number not in by_number:
            problems.append(f"reference [{number}] is missing from sources.json")
        urls = re.findall(r"https?://[^\s<>]+", detail)
        if len(urls) != 1:
            problems.append(f"reference [{number}] must contain exactly one URL")
        elif number in by_number and urls[0] != by_number[number].get("url"):
            problems.append(f"reference [{number}] URL does not match sources.json")
    for number in sorted(by_number):
        count = reference_counts.get(number, 0)
        if count != 1:
            problems.append(f"source [{number}] needs exactly one reference line (found {count})")
    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
