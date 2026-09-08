#!/usr/bin/env python3
"""Add (or refresh) the Structured RLM sources in index.html, without running
the generator.

`generate_dashboard.py` reads run trees under `../kvpress/...` that exist only
on the evaluation host; running it from a laptop blanks every source it cannot
rebuild and deletes their published CSVs. `refresh_dashboard.py` splices back
the 9 legacy sources it names, but NOT the RLM ones -- so on a laptop even a
refresh loses rlm-fixedgrid and rlm-new.

This script never runs the generator. It takes the committed index.html's DATA
as the baseline, replaces any structured-rlm-* entries with freshly collected
ones, and rewrites index.html from template.html. Every other source is passed
through byte-identical.

Usage:
    python3 splice_structured_rlm.py
"""

import json
import re
import subprocess
import sys
from pathlib import Path

from generate_dashboard import structured_rlm_sources

HERE = Path(__file__).resolve().parent
PREFIX = "structured-rlm-"


def extract_data(html_text: str) -> list:
    match = re.search(r"const DATA=(\[.*?\]);", html_text, re.S)
    if not match:
        raise ValueError("could not find `const DATA=[...]` in the given HTML")
    return json.loads(match.group(1))


def main() -> None:
    committed = subprocess.run(
        ["git", "show", "HEAD:index.html"],
        cwd=str(HERE),
        stdout=subprocess.PIPE,
        universal_newlines=True,
        check=True,
    ).stdout
    baseline = extract_data(committed)

    fresh = structured_rlm_sources()
    if not fresh:
        sys.exit("structured_rlm_sources() returned nothing -- is data/structured_rlm_loft128k.csv present?")

    kept = [d for d in baseline if not d["id"].startswith(PREFIX)]
    dropped = len(baseline) - len(kept)
    merged = kept + fresh
    ids = [d["id"] for d in merged]
    assert len(ids) == len(set(ids)), "duplicate ids after merge"

    # The metric dropdown is the INTERSECTION of `metrics` across the sources
    # selected in a group, so a source that omits a metric silently removes it
    # for every other source it shares a group with. This campaign declares a
    # short list (no f1, no coverage), which is safe only because it sits alone
    # in its own group. Check that rather than trusting it to stay true.
    groups = {}
    for dataset in merged:
        groups.setdefault(dataset["group"], []).append(dataset)
    for group, members in groups.items():
        metric_sets = [set(m["metrics"]) for m in members if m["metrics"]]
        if not metric_sets:
            continue
        shared = set.intersection(*metric_sets)
        for member in members:
            if member["metrics"] and set(member["metrics"]) != shared:
                print(f"  note: {group}/{member['id']} declares metrics beyond the group intersection {sorted(shared)}")

    template = (HERE / "template.html").read_text()
    (HERE / "index.html").write_text(template.replace("__DASHBOARD_DATA__", json.dumps(merged, separators=(",", ":"))))

    print(f"Wrote index.html: {len(kept)} passed through, {dropped} structured-rlm replaced, {len(fresh)} added.")
    for dataset in fresh:
        cells = sum(len(runs) for runs in dataset["tasks"].values())
        print(f"  [{dataset['id']:32}] {len(dataset['tasks'])} tasks, {len(dataset['budgets'])} arms, {cells} cells")


if __name__ == "__main__":
    main()
