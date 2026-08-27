#!/usr/bin/env python3
"""Regenerate index.html, preserving the legacy sources generate_dashboard.py
can no longer rebuild on its own.

Why this exists: `evaluation/results` (the symlink generate_dashboard.py's
EVAL constant points at) currently only reaches the LOFT-32K matrices, not
RULER-32K or the Qwen3-8B Synthetic-KV/LOFT-128K runs. Running
`generate_dashboard.py` directly regenerates every *other* source correctly
(including anything new dropped into evaluation/results/rlm/, like the
LOFT-128K auto-chunk-sizing sweeps), but blanks those 9 legacy sources and
deletes their published prediction CSVs as "stale."

This script runs the generator, then splices the 9 legacy sources back in
from the last git-committed index.html (which still has them, since nothing
has re-generated them since the symlink broke) before writing the final
file. It is self-contained -- it does not depend on any cached files outside
this repo, so it is safe to run standalone, any time, including from a
machine/session that never touched this dashboard before.

Usage:
    python3 refresh_dashboard.py
    git add -A -- generate_dashboard.py index.html downloads/
    git commit -m "..."
    git push origin main
"""

import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# These are the only sources generate_dashboard.py cannot currently rebuild
# (see EVAL/BENCH comment at the top of generate_dashboard.py). Every other
# source -- old or new -- comes from the fresh regeneration.
LEGACY_IDS_TO_PRESERVE = {
    "loft32k-fp",
    "loft32k-awq",
    "loft128k",
    "loft128k-awq",
    "ruler32k",
    "synthetic32k-nonquantized",
    "synthetic32k-awq",
    "synthetic64k-awq",
    "synthetic64k-nonquantized",
}


def extract_data(html_text: str) -> list:
    match = re.search(r"const DATA=(\[.*?\]);", html_text, re.S)
    if not match:
        raise ValueError("could not find `const DATA=[...]` in the given HTML")
    return json.loads(match.group(1))


def last_committed_index_html() -> str:
    result = subprocess.run(
        ["git", "show", "HEAD:index.html"],
        cwd=str(HERE),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=True,
    )
    return result.stdout


def main() -> None:
    # 1. Capture the legacy sources from the last commit BEFORE regenerating
    #    (generate_dashboard.py's own run will blank them and delete their
    #    downloads/*.csv files as "stale" -- git restores those after).
    baseline_data = extract_data(last_committed_index_html())
    preserved = [d for d in baseline_data if d["id"] in LEGACY_IDS_TO_PRESERVE]
    found_ids = {d["id"] for d in preserved}
    missing = LEGACY_IDS_TO_PRESERVE - found_ids
    if missing:
        print(f"WARNING: expected legacy sources not found in the last commit: {sorted(missing)}", file=sys.stderr)

    # 2. Run the real generator. This is what picks up everything new (e.g.
    #    any additional evaluation/results/rlm/loft128k_autosub_*/ runs that
    #    have completed more configs since the last refresh).
    subprocess.run([sys.executable, "generate_dashboard.py"], cwd=HERE, check=True)

    # 3. generate_dashboard.py's own cleanup step deletes prediction CSVs
    #    for the now-blanked legacy sources, thinking they're stale. Restore
    #    them from git -- this is safe because step 4 below puts an
    #    identical dataset entry back, so the CSVs are genuinely still
    #    referenced by the final index.html.
    subprocess.run(["git", "checkout", "--", "downloads/"], cwd=HERE, check=True)

    # 4. Splice the preserved legacy sources back into the freshly generated
    #    DATA, replacing whatever blanked versions generate_dashboard.py
    #    just produced for those same ids.
    fresh_html = (HERE / "index.html").read_text()
    fresh_data = extract_data(fresh_html)
    fresh_ids = {d["id"] for d in fresh_data}
    preserved_by_id = {d["id"]: d for d in preserved}
    merged = [preserved_by_id[d["id"]] if d["id"] in found_ids else d for d in fresh_data]
    # generate_dashboard.py always emits an (possibly blank) entry for every id
    # still in its SOURCES list, so this normally adds nothing -- defensive only,
    # in case a legacy id's SOURCES entry is ever removed there.
    for source in preserved:
        if source["id"] not in fresh_ids:
            merged.append(source)
    merged_ids = [d["id"] for d in merged]
    assert len(merged_ids) == len(set(merged_ids)), "duplicate ids after merge"

    template = (HERE / "template.html").read_text()
    output = template.replace("__DASHBOARD_DATA__", json.dumps(merged, separators=(",", ":")))
    (HERE / "index.html").write_text(output)

    print(f"\nWrote index.html with {len(merged)} datasets ({len(preserved)} preserved legacy + {len(merged) - len(preserved)} freshly generated).")
    for d in merged:
        tag = "preserved" if d["id"] in found_ids else "fresh"
        print(f"  [{tag:9}] {d['id']:35} {len(d['tasks'])} tasks, excluded={d['excluded']}")


if __name__ == "__main__":
    main()
