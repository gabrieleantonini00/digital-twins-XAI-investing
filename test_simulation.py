"""Offline checks for sample integrity and batch result reconciliation."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import simulation as s


class StudyTests(unittest.TestCase):
    def setUp(self):
        self.config = s.read_json(s.ROOT / 'study_config.json')
        self.row = dict(pid='17', persona_summary='A profile.', Market_Context=s.MARKETS[0], Transparency=s.TRANSPARENCIES[1])

    def test_protocol_sample(self):
        run = s.ROOT / 'runs' / self.config['study_id']
        rows = s.verify_manifest(run, self.config)
        self.assertEqual(len({r['pid'] for r in rows}), 500)
        from collections import Counter
        self.assertEqual(set(Counter((r['Market_Context'], r['Transparency']) for r in rows).values()), {125})
        self.assertEqual(len({r['persona_summary'] for r in rows}), 500)

    def test_baseline_cannot_see_treatment(self):
        req = s.make_request(self.row, self.config, 'baseline')
        text = json.dumps(req)
        for condition in s.MARKETS + s.TRANSPARENCIES:
            self.assertNotIn(condition, text)
        self.assertEqual(req['body']['temperature'], 1)
        self.assertNotIn('top_p', req['body'])

    def test_reject_invalid_or_extra_outcome(self):
        for obj in [{'Willingness_to_Invest': 8}, {'Willingness_to_Invest': True}, {'Willingness_to_Invest': 2.5}, {'Willingness_to_Invest': 4, 'pid': 17}]:
            with self.assertRaises(ValueError):
                s.validate_answer('decision', obj)

    def test_null_demographics_are_preserved(self):
        obj = dict(Financial_Literacy=4, Tech_Savviness=4, Age=None, Gender=None, Education=None)
        self.assertEqual(s.validate_answer('baseline', obj), obj)

    def test_incomplete_response_never_becomes_outcome(self):
        with self.assertRaises(ValueError):
            s.response_answer('decision', {'status': 'incomplete', 'output': []})

    def test_collect_matches_ids_not_order(self):
        rows = [dict(self.row, pid=str(n)) for n in range(1, 501)]
        responses = []
        for row in reversed(rows):
            body = {'id': 'r' + row['pid'], 'model': self.config['model'], 'status': 'completed',
                    'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': json.dumps({'Willingness_to_Invest': 1 + int(row['pid']) % 7})}]}],
                    'usage': {'input_tokens': 10, 'output_tokens': 5}}
            responses.append({'custom_id': 'decision-' + row['pid'], 'response': {'status_code': 200, 'body': body}})
        raw = '\n'.join(json.dumps(r) for r in responses)
        client = SimpleNamespace(files=SimpleNamespace(content=lambda _: SimpleNamespace(text=raw)))
        batch = SimpleNamespace(id='test', output_file_id='file', error_file_id=None, status='completed')
        with tempfile.TemporaryDirectory() as tmp, patch.object(s, 'verify_manifest', return_value=rows):
            run = Path(tmp)
            s.collect_batch(client, run, self.config, 'decision', batch)
            answers = s.read_json(run / 'decision.answers.json')
            self.assertEqual(answers['1']['Willingness_to_Invest'], 2)
            self.assertEqual(answers['500']['Willingness_to_Invest'], 4)
            responses[0]['custom_id'] = responses[1]['custom_id']
            raw = '\n'.join(json.dumps(r) for r in responses)
            with self.assertRaises(ValueError):
                s.collect_batch(client, run, self.config, 'decision', batch)

    def test_cost_includes_reasoning_and_cache_writes(self):
        usage = {'input_tokens': 1000, 'input_tokens_details': {'cached_tokens': 100, 'cache_write_tokens': 200},
                 'output_tokens': 500, 'output_tokens_details': {'reasoning_tokens': 400}}
        self.assertAlmostEqual(s.usage_cost(usage, self.config), (700*2 + 200*2.5 + 100*.2 + 500*10)/1e6)

    def test_partial_batch_uses_frozen_pid_subset(self):
        body = {'status': 'completed', 'model': self.config['model'], 'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': '{"Willingness_to_Invest": 3}'}]}]}
        result = {'custom_id': 'decision-17', 'response': {'status_code': 200, 'body': body}}
        client = SimpleNamespace(files=SimpleNamespace(content=lambda _: SimpleNamespace(text=json.dumps(result))))
        batch = SimpleNamespace(id='test-part', output_file_id='file', error_file_id=None, status='completed')
        with tempfile.TemporaryDirectory() as tmp, patch.object(s, 'verify_manifest', return_value=[self.row, dict(self.row, pid='18')]):
            s.collect_batch(client, Path(tmp), self.config, 'decision', batch, label='decision.part01', expected_pids={'17'})
            self.assertEqual(set(s.read_json(Path(tmp) / 'decision.part01.answers.json')), {'17'})


if __name__ == '__main__':
    unittest.main()
