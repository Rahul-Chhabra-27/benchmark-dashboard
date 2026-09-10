#!/usr/bin/env python3
"""Build the Structured RLM explorer without regenerating unrelated run trees.

With --snapshot, import the audited InfoLabs raw/ directory. Without it, render
the committed data. Token usage is the recorded root+sub-model total, including
input and output tokens, not peak context or KV retention.
"""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def import_snapshot(root):
    import yaml
    campaigns = [
        dict(id='query', title='Query selection study · Sep 10', path='qagnostic/results',
             description='84 completed configurations · 55 test queries per subset. Every arm uses the question while reading and answering. Lead, stride, random and exhaustive choose windows without it.',
             note='All 84 cells pass the audited completion and map-call checks. Exhaustive uses 8,000-character windows; the retrieval ladder uses 2,000. Holdout answering and notes-based answering have not run.'),
        dict(id='geometry', title='Chunk geometry · Sep 9', path='structured_grid',
             description='117 completed configurations · 55 test queries per subset. Window size, read size, query format and retrieval hops are separate axes.',
             note='11 corrected-query configurations remain incomplete, including two partial checkpoints (18 and 28 rows). Only completed cells appear. Oracle, exposure and two-hop cells use the old task-query setting.'),
    ]
    labels = {'bm25-bare':'BM25 · question only', 'bm25-task':'BM25 · task + question',
              'all':'Exhaustive · every window', 'lead':'Lead windows', 'stride':'Evenly spaced windows',
              'random-s0':'Random · seed 0','random-s1':'Random · seed 1','random-s2':'Random · seed 2',
              'oracle':'Oracle · gold-answer windows'}
    runs = []
    downloads = HERE / 'downloads/structured'
    downloads.mkdir(exist_ok=True)
    for campaign in campaigns:
        for mp in sorted((root / campaign['path']).glob('*/metrics.json')):
            directory = mp.parent
            config = yaml.safe_load((directory / 'config.yaml').read_text())
            m = json.loads(mp.read_text()); rt = m['runtime']
            records = {}
            for line in (directory / 'checkpoint.jsonl').read_text().splitlines():
                if line.strip():
                    row = json.loads(line); records[row['id']] = row
            rows = list(records.values())
            assert rt['scored'] == len(rows) == 55 and rt['errors'] == 0, directory
            task = config['data_dir']; metric = 'coverage' if task.startswith(('qampari','quest')) else 'subspan_em'
            arm = config.get('chunks_tag') or config['chunks_source']
            if arm == 'bm25': arm += '-' + config.get('chunks_query','task')
            if arm == 'file': arm = 'oracle'
            selection = 'oracle' if arm == 'oracle' else 'aware' if arm.startswith('bm25') else 'independent'
            k = config['chunks_k']; window = config['search_window_chars']
            label = labels[arm] + ('' if arm == 'all' else f' · k={k}')
            if campaign['id'] == 'geometry':
                label += f' · w={window:,}'
                if config.get('structured_map_chars'): label += f" · read={config['structured_map_chars']:,}"
                if config.get('structured_hops',1)>1: label += ' · 2 hops'
            uid = campaign['id'] + '-' + hashlib.sha256(directory.name.encode()).hexdigest()[:16]
            filename = uid + '.csv'
            fields = ['id','question','answers','prediction','tokens','latency_s','finished','end_reason']
            exported = [dict(id=r['id'],question=r.get('question',''),answers=json.dumps(r['answers'],ensure_ascii=False),
                             prediction=r.get('pred',''),tokens=r.get('tokens'),latency_s=r.get('latency_s'),
                             finished=r.get('finished'),end_reason=r.get('end_reason')) for r in rows]
            with (downloads / filename).open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(exported)
            total = sum(r['tokens'] for r in rows)
            assert abs(total / len(rows) - rt['average_tokens']) < 1e-6, directory
            runs.append(dict(id=uid,campaign=campaign['id'],task=task,label=label,arm=arm,selection=selection,
                             k=k,window=window,overlap=config['search_overlap_chars'],
                             read_chars=config.get('structured_map_chars',0),finding_chars=config.get('reduce_finding_chars',2000),
                             hops=config.get('structured_hops',1),granularity=config.get('structured_map_granularity'),
                             query=config.get('chunks_query','task'),metric=metric,score=m[metric]*100,
                             em=m['em']*100,n=len(rows),tokens_per_query=rt['average_tokens'],total_tokens=total,
                             latency_s=rt['average_latency_s'],root_peak_tokens=rt.get('average_peak_context_tokens'),
                             document_coverage=rt.get('document_coverage_fraction'),map_calls=rt.get('average_map_calls'),
                             abstained=rt.get('abstained'),errors=rt.get('errors'),unfinished=rt.get('unfinished'),
                             prediction_url='downloads/structured/'+filename,preview=exported[:3],run_name=directory.name))
        campaign.pop('path')
    # Preserve the distinct historical denominators; do not fabricate missing usage.
    titles={'baseline':'Vanilla reference · 110 rows','full55':'Original pipeline comparison · 55 rows',
            'hardslice':'Selected hard slice · 22 / 29 rows','rev2_2x2':'Prompt / clustering ablation · pooled 110',
            'transfer':'Chunk transfer · pooled 51'}
    for key,title in titles.items():
        campaigns.append(dict(id=key,title=title,description='Historical September 8 comparison. Reported aggregates retained from the original export.',
            note=('The vanilla reference uses 110 rows and cannot be paired with the 55-row experiments. Its context length is a document-size estimate, not recorded token usage.' if key=='baseline' else
                  'Selected on the agentic arm failing; these scores do not estimate unselected task accuracy.' if key in ('hardslice','transfer') else
                  'The structured retriever used the old instruction-heavy query. This comparison cannot isolate pipeline quality from retrieval-query quality.') + ' Usage is recovered from the matching checkpoints where available; a dash means unavailable, not zero.'))
    cache = {}
    def matching(folder, task, **wanted):
        key=(folder,task,tuple(wanted.items()))
        if key not in cache:
            found=[]
            for cp in (root/folder).glob('*/config.yaml'):
                c=yaml.safe_load(cp.read_text())
                if c['data_dir']==task and all(c.get(k)==v for k,v in wanted.items()):
                    values={}
                    for line in (cp.parent/'checkpoint.jsonl').read_text().splitlines():
                        if line.strip():
                            r=json.loads(line);values[r['id']]=r
                    found.append(list(values.values()))
            assert len(found)==1,(key,len(found))
            cache[key]=found[0]
        return cache[key]
    def historical_rows(source, arm, task):
        if source=='baseline': return []
        if task.startswith('pooled_'):
            return sum((historical_rows(source,arm,t) for t in ['nq_128k','hotpotqa_128k']),[])
        if arm.startswith('A ·'):
            values=matching('loft128k_main',task,search_k=0)
        elif arm.startswith('B ·'):
            values=matching('loft128k_main',task,search_k=5)
        elif arm.startswith('B′ rev1') or arm.startswith('cluster off · shape offer'):
            values=matching('structured_main',task,chunks_source='bm25')
        elif arm.startswith('B′ rev2') or arm.startswith('cluster on · shape require'):
            values=matching('structured_main_r2',task,chunks_source='bm25')
        elif arm.startswith('C rev1'):
            values=matching('structured_main',task,chunks_source='file')
        elif arm.startswith('C rev2') or arm.startswith('C ·'):
            values=matching('structured_main_r2',task,chunks_source='file')
        elif arm.startswith('abl 1'):
            values=matching('structured_ablation',task,structured_map_question='query')
        elif arm.startswith('abl 2'):
            values=matching('structured_ablation',task,structured_map_granularity='all')
        elif arm.startswith('cluster off'):
            values=matching('structured_ablation',task,structured_map_granularity='window')
        elif arm.startswith('cluster on'):
            values=matching('structured_ablation',task,structured_answer_shape='offer')
        elif arm.startswith(('4c','4 ')):
            folder='structured_ablation2' if 'capped' in arm else 'structured_ablation'
            values=matching(folder,task,mode='rlm',chunks_source='bm25' if arm.startswith('4c') else 'file')
        else: raise ValueError(arm)
        if source in ('hardslice','transfer'):
            ids={r['id'] for r in matching('structured_main_r2',task,chunks_source='file')}
            values=[r for r in values if r['id'] in ids]
            assert {r['id'] for r in values}==ids
        return values
    for index,row in enumerate(csv.DictReader((HERE/'data/structured_rlm_loft128k.csv').open())):
        runs.append(dict(id=f'archive-{index}',campaign=row['source'],task=row['task'],label=row['arm'],arm='archive',
                         selection='historical',k=None,window=None,metric='subspan_em',score=float(row['subspan_em'])*100,
                         em=float(row['em'])*100 if row['em'] else None,n=int(row['samples']),tokens_per_query=None,
                         total_tokens=None,latency_s=None,root_peak_tokens=float(row['peak_context_tokens']) if row['peak_context_tokens'] and row['source']!='baseline' else None,
                         document_coverage=None,map_calls=None,abstained=None,errors=None,unfinished=None,prediction_url=None,preview=[]))
        values=historical_rows(row['source'],row['arm'],row['task'])
        if values:
            assert len(values)==int(row['samples']),row
            total=sum(r['tokens'] for r in values)
            filename=f'archive-{index}.csv'
            exported=[dict(id=r['id'],question=r.get('question',''),answers=json.dumps(r['answers'],ensure_ascii=False),prediction=r.get('pred',''),tokens=r['tokens'],latency_s=r.get('latency_s'),finished=r.get('finished'),end_reason=r.get('end_reason')) for r in values]
            with (downloads/filename).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(exported)
            runs[-1].update(tokens_per_query=total/len(values),total_tokens=total,
                            latency_s=sum(r['latency_s'] for r in values)/len(values),
                            prediction_url='downloads/structured/'+filename,preview=exported[:3])
    assert len([r for r in runs if r['campaign']=='query'])==84
    assert len([r for r in runs if r['campaign']=='geometry'])==117
    return dict(updated='2026-09-10',model='Qwen3-4B-Instruct-2507',campaigns=campaigns,runs=runs,
                pending=['Notes indexes: all five built; answering not evaluated.','Held-out queries: 45 IDs per subset prepared; answering not evaluated.'],
                token_definition='Recorded input + output model tokens across root and sub-calls. Per-query is the mean over scored queries; run total is their sum. These are cumulative usage, not peak context or physical KV memory.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,help='Path to the downloaded raw/ directory')
    args=parser.parse_args()
    if args.snapshot:
        data=import_snapshot(args.snapshot)
        (HERE/'data/structured_results.json').write_text(json.dumps(data,ensure_ascii=False,separators=(',',':'))+'\n')
        fields=['id','campaign','task','label','selection','k','window','read_chars','finding_chars','hops','metric','score','em','n','tokens_per_query','total_tokens','latency_s','root_peak_tokens','document_coverage','map_calls','abstained','errors','unfinished','run_name']
        with (HERE/'data/structured_results.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(data['runs'])
    html=(HERE/'index.html').read_text()
    match=re.search(r'const DATA=(\[.*?\]);',html,re.S)
    if not match: raise ValueError('Missing legacy DATA; refusing to rebuild')
    # Keep all legacy sources and downloads. The new explorer reads its own data.
    payload=match.group(1)
    output=(HERE/'template.html').read_text().replace('__DASHBOARD_DATA__',payload)
    (HERE/'index.html').write_text(output)
    print('Updated Structured RLM; all legacy source data preserved.')


if __name__=='__main__': main()
