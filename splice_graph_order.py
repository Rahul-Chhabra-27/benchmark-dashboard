#!/usr/bin/env python3
"""Add (or refresh) the BM25-graph vs source-order campaign in the Structured
RLM explorer, together with same-question references: vanilla, KVzip only and
base RLM.

    python3 splice_graph_order.py path/to/export.json

`export.json` is written on the compute host by the benchmark repo's export
step: every row is scored with the canonical LOFT scorer on the SAME 55 test
questions (test rows 0-54). The 110-row reference runs are aligned by question
text and checked against gold answers before anything is exported.

This touches only the `graph-order` campaign: its runs in
data/structured_results.json, their rows in data/structured_results.csv, and
downloads/structured/graph-order-*.csv. Everything else passes through as-is,
and index.html is not rebuilt (the explorer fetches its JSON at load time).
"""

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAMPAIGN = "graph-order"
DOWNLOADS = HERE / "downloads/structured"
# Same flat export columns as update_structured_dashboard.py, plus the
# input/output split this campaign records.
FIELDS = ['id', 'campaign', 'task', 'label', 'selection', 'k', 'window', 'read_chars', 'finding_chars', 'hops',
          'metric', 'score', 'em', 'n', 'tokens_per_query', 'total_tokens', 'latency_s', 'root_peak_tokens',
          'document_coverage', 'map_calls', 'abstained', 'errors', 'unfinished', 'run_name']
EXTRA_FIELDS = ['prompt_tokens_per_query', 'completion_tokens_per_query']
SELECTION = {'source': 'order-source', 'bm25-graph': 'order-graph', 'vanilla': 'vanilla', 'rlm': 'rlm',
             'kvzip': 'kvzip'}
READER = {
    'source': 'Every 2,000-char chunk, in source order, packed into ≤8,192-token reader groups → reduce',
    'bm25-graph': 'Every 2,000-char chunk, in BM25-graph order, packed into ≤8,192-token reader groups → reduce',
    'vanilla': 'Whole document in one call (RLM harness, no truncation)',
    'rlm': 'Agentic REPL root with sub-calls (Aug 30 guards rerun)',
    'kvzip': 'Whole document prefilled, KV cache compressed by KVzip, one answer (kvpress harness)',
}
NAMES = {'nq_128k': 'NQ', 'hotpotqa_128k': 'HotpotQA', 'musique_128k': 'MuSiQue', 'qampari_128k': 'Qampari',
         'quest_128k': 'Quest'}
ORDER = {'bm25-graph': 0, 'source': 1, 'vanilla': 2, 'rlm': 3, 'kvzip': 4}
# How the graph order is built -- mirrors evaluation/rlm/organization.py (config
# revision 1). No step ever sees the question or its answers.
EXPLAINER = [
    '1 · Chunk. The document is cut into consecutive 2,000-character chunks with no overlap — about 240 '
    'chunks for a 128k-token LOFT document.',
    '2 · Score every pair with BM25. Each chunk is used as a BM25 query against every chunk (lower-cased '
    'alphanumeric words, stopwords removed, k1 = 1.5, b = 0.75). A shared word counts for more when it is rare '
    'in the document (inverse document frequency) and frequent in the matched chunk, with a penalty for long '
    'chunks. So two chunks score high when they share distinctive words — the same names, dates or entities.',
    '3 · Make it symmetric. A chunk\'s match with itself is dropped, each chunk\'s scores are divided by its '
    'own best match, and the two directions are averaged: w(i, j) = (s(i→j)/max_i + s(j→i)/max_j) / 2. It '
    'is 1 only when each chunk is the other\'s best match.',
    '4 · Keep strong links only. Each chunk keeps edges to its 5 highest-weight neighbours (the union over '
    'all chunks), giving a sparse similarity graph.',
    '5 · Spanning forest. Kruskal\'s algorithm keeps the strongest edges that do not close a cycle — a '
    'maximum-weight spanning forest, one tree per connected group of chunks (on nq: 238 chunks, 779 kept '
    'edges, one tree).',
    '6 · Walk it. Starting from the earliest chunk, each tree is walked depth-first, always following the '
    'strongest untaken edge and backtracking at dead ends; then the next unvisited earliest chunk starts a new '
    'walk. Ties go to the earlier chunk, so the order is deterministic. Chunks with no edges are kept, in '
    'place. Building the graph costs under 1 CPU-second per document, once, shared by every question.',
    'Both arms then pack chunks in their order into reader groups of at most 8,192 tokens (counted with the '
    'real chat template, instructions and question included); each chunk keeps its own [source chars s:e] '
    'label, so distant chunks placed side by side stay distinguishable. One reader call per group writes '
    'findings, and the reducer sees them in group order. Source order is simply chunk 1, 2, 3, … — so both '
    'arms read identical text under identical budgets, and only the order differs.',
]


def slug(row):
    if row['kind'] == 'kvzip':
        return 'kvzip-' + (str(row['budget']) if row.get('budget') else 'nocompress')
    return row['kind']


def task_note(task, rows, paired):
    arms = {r['kind']: r for r in rows if r['kind'] in ('source', 'bm25-graph')}
    parts = []
    running = [f"{'graph' if k == 'bm25-graph' else 'source'} {r['n']}/55" for k, r in arms.items() if r['n'] < 55]
    if len(arms) < 2:
        missing = [k for k in ('source', 'bm25-graph') if k not in arms]
        running += [f"{'graph' if k == 'bm25-graph' else 'source'} 0/55" for k in missing]
    if running:
        parts.append(f"STILL RUNNING — {', '.join(running)} questions finished; scores below are interim.")
    if paired:
        scale = 100
        parts.append(
            f"Paired on the same {paired['n']} questions: graph − source = {paired['delta'] * scale:+.1f} points "
            f"(95% bootstrap CI {paired['ci_lo'] * scale:+.1f} to {paired['ci_hi'] * scale:+.1f}); "
            f"graph wins / losses / ties {paired['wins']} / {paired['losses']} / {paired['ties']}; "
            f"sign test p = {paired['sign_p']:.2g}. One question is {scale / 55:.1f} points at n = 55."
        )
    return ' '.join(parts)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    export = json.loads(Path(sys.argv[1]).read_text())
    path = HERE / 'data/structured_results.json'
    data = json.loads(path.read_text())

    data['runs'] = [r for r in data['runs'] if r['campaign'] != CAMPAIGN]
    data['campaigns'] = [c for c in data['campaigns'] if c['id'] != CAMPAIGN]
    for old in DOWNLOADS.glob(f'{CAMPAIGN}-*.csv'):
        old.unlink()

    runs, task_notes, summary = [], {}, {'paired': export['paired'], 'tasks': {}}
    for task, rows in export['tasks'].items():
        rows = sorted(rows, key=lambda r: (ORDER[r['kind']], r.get('budget') or 0))
        task_notes[task] = task_note(task, rows, export['paired'].get(task))
        summary['tasks'][task] = [{k: v for k, v in r.items() if k != 'preds'} for r in rows]
        for row in rows:
            if not row['n']:
                continue
            run_id = f"{CAMPAIGN}-{task.replace('_128k', '')}-{slug(row)}"
            filename = run_id + '.csv'
            preds = row['preds']
            with (DOWNLOADS / filename).open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['id', 'answers', 'prediction', 'tokens', 'latency_s', 'score'],
                                        extrasaction='ignore', lineterminator='\n')
                writer.writeheader()
                writer.writerows(preds)
            assert sum(p['tokens'] for p in preds) == row['total_tokens'], run_id
            reading = row['kind'] in ('source', 'bm25-graph')
            runs.append(dict(
                id=run_id, campaign=CAMPAIGN, task=task, label=row['label'], arm=row['kind'],
                selection=SELECTION[row['kind']], reader=READER[row['kind']],
                k=None, window=2000 if reading else None, overlap=0 if reading else None, read_chars=0,
                finding_chars=2000 if reading else None, hops=1 if reading else None,
                granularity='reader groups' if reading else None, query='task',
                metric=row['metric'], score=row['score'], em=None, n=row['n'],
                tokens_per_query=row['tokens_per_query'], total_tokens=row['total_tokens'],
                prompt_tokens_per_query=row.get('prompt_tokens_per_query'),
                completion_tokens_per_query=row.get('completion_tokens_per_query'),
                latency_s=row.get('latency_s'),
                # Root context = the reducer's input on the reading arms; unknown elsewhere.
                root_peak_tokens=row.get('root_peak_tokens') if reading else None,
                document_coverage=1.0 if reading else None, map_calls=row.get('map_calls'),
                abstained=None, errors=row.get('errors'), unfinished=None,
                prediction_url='downloads/structured/' + filename, preview=preds[:3],
                run_name=run_id, complete=row.get('complete', True)))

    interim = any(r['n'] < 55 for r in runs if r['arm'] in ('source', 'bm25-graph')) or \
        len({r['task'] for r in runs if r['arm'] in ('source', 'bm25-graph')}) < 5
    campaign = dict(
        id=CAMPAIGN,
        title='Graph order vs source order · Sep 10' + (' (running)' if interim else ''),
        description=(
            'Does organizing a document WITHOUT the question help? Both arms read every 2,000-character chunk of the '
            'document, packed into reader groups of at most 8,192 tokens, then reduce; the only difference is chunk '
            'ORDER — source order, or a question-independent BM25 similarity graph (maximum spanning forest, '
            'depth-first). References on the same 55 test questions: vanilla (whole document in one call), base RLM '
            '(agentic REPL) and KVzip only (compressed KV cache, no RLM).'),
        note=(
            ('Interim snapshot: the campaign is still running and the per-example gate has not been run on the full '
             'set yet. ' if interim else '') +
            'Token usage is recorded input + output model tokens per query. The two reading arms read the whole '
            'document, so they cost about as much as vanilla (~126k tokens / query); base RLM reads a fraction of it. '
            'KVzip rows come from the kvpress harness (LOFT\'s own prompt, 256-token answer cap), so compare its '
            'budgets with its own no-compression row, not with vanilla; its tokens are the prefilled prompt plus the '
            'generated answer counted with the Qwen tokenizer, excluding KVzip\'s own scoring pass. Every row here, '
            'KVzip included, is rescored with the same LOFT scorer on these 55 questions, so KVzip numbers differ '
            'from the LOFT 128K tab (different scorer and row set). Vanilla and base '
            'RLM come from the Aug 30 110-row runs, restricted to these 55 test questions. qampari and quest are '
            'scored on answer coverage, the others on subspan EM — never compare across the two.'),
        task_notes=task_notes,
        explainer=EXPLAINER,
    )
    data['campaigns'].insert(0, campaign)
    data['runs'] = runs + data['runs']
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')) + '\n')
    (HERE / 'data/graph_order_loft128k.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1) + '\n')

    # The flat CSV keeps every other row's values unchanged (they only gain the two
    # new, empty input/output columns); this campaign's rows go to the end.
    csv_path = HERE / 'data/structured_results.csv'
    existing = [r for r in csv.DictReader(csv_path.open()) if r['campaign'] != CAMPAIGN]
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS + EXTRA_FIELDS, extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        writer.writerows(existing)
        writer.writerows(runs)
    for task, note in task_notes.items():
        print(f"{NAMES.get(task, task)}: {note}")
    print(f"{len(runs)} graph-order runs written ({'interim' if interim else 'complete'}).")


if __name__ == '__main__':
    main()
