"""Integrity checks for the public result data and prediction downloads."""
import csv
import json
import math
import unittest
from pathlib import Path

HERE=Path(__file__).resolve().parent


class StructuredDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=json.loads((HERE/'data/structured_results.json').read_text())

    def test_complete_campaigns_and_unique_runs(self):
        rows=self.data['runs']
        self.assertEqual(sum(r['campaign']!='graph-order' for r in rows),235)
        self.assertEqual(len({r['id'] for r in rows}),len(rows))
        self.assertEqual(sum(r['campaign']=='query' for r in rows),84)
        self.assertEqual(sum(r['campaign']=='geometry' for r in rows),117)
        for r in rows:
            self.assertTrue(math.isfinite(r['score']) and 0<=r['score']<=100)
            if r['campaign'] in ('query','geometry'):
                self.assertEqual(r['n'],55)
                self.assertEqual(r['errors'],0)
                expected='coverage' if r['task'].startswith(('qampari','quest')) else 'subspan_em'
                self.assertEqual(r['metric'],expected)
                self.assertEqual(r['selection']=='independent',r['arm'] in ('lead','stride','all','random-s0','random-s1','random-s2'))

    def test_downloaded_usage_matches_aggregates(self):
        for run in self.data['runs']:
            with self.subTest(run=run['id']):
                if not run['prediction_url']:
                    self.assertEqual(run['campaign'],'baseline')
                    self.assertIsNone(run['tokens_per_query'])
                    continue
                path=HERE/run['prediction_url']
                self.assertTrue(path.is_relative_to(HERE/'downloads/structured'))
                with path.open() as f: rows=list(csv.DictReader(f))
                self.assertEqual(len(rows),run['n'])
                self.assertEqual(len({r['id'] for r in rows}),run['n'])
                total=sum(int(r['tokens']) for r in rows)
                self.assertEqual(total,run['total_tokens'])
                self.assertAlmostEqual(total/len(rows),run['tokens_per_query'])

    def test_exhaustive_is_full_read_and_usage_is_not_root_context(self):
        for r in self.data['runs']:
            if r.get('arm')!='all': continue
            self.assertEqual(r['document_coverage'],1)
            self.assertEqual(r['map_calls'],r['k'])
            self.assertGreater(r['tokens_per_query'],160000)
            self.assertLess(r['root_peak_tokens'],2000)


    def test_compare_view_rows_carry_usage(self):
        # The cross-campaign view plots every Sep 9 / Sep 10 row on a token axis.
        for r in self.data['runs']:
            if r['campaign'] in ('graph-order','query','geometry'):
                with self.subTest(run=r['id']):
                    self.assertIsNotNone(r['tokens_per_query'])
                    self.assertGreater(r['tokens_per_query'],0)
                    self.assertLessEqual(r['n'],55)


if __name__=='__main__': unittest.main()
