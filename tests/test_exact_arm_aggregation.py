"""Regression checks for the BATTERY arm-name collision; no policy execution."""
import json
from pathlib import Path
import tempfile
import unittest

from training import stage1, stage2


class ExactArmAggregation(unittest.TestCase):
    def check_pair(self, stage, tag, longer_tag):
        with tempfile.TemporaryDirectory() as d:
            shards = Path(d) / 'shards'
            shards.mkdir()
            for label, successes in [(tag, 0), (longer_tag, 5)]:
                for start in (0, 5):
                    p = shards / f'BATTERY_sac_0_2000_{label}_{start}__0123456789.json'
                    p.write_text(json.dumps({'summary': {
                        'counts': {'episodes': 5, 'safe_completions': successes},
                        'constraint_hash': 'fixture'}}))
            actual = stage.gather(d, 'BATTERY', 'sac', 0, 2000, tag)
            self.assertEqual(actual['counts'], {'episodes': 10, 'safe_completions': 0})
            longer = stage.gather(d, 'BATTERY', 'sac', 0, 2000, longer_tag)
            self.assertEqual(longer['counts'], {'episodes': 10, 'safe_completions': 10})

    def test_nominal_trix_does_not_include_preventive(self):
        self.check_pair(stage1, 'test_trix', 'test_trix_preventive')

    def test_filter_aware_box_does_not_include_preventive(self):
        for phase in ('screen', 'confirm', 'test'):
            with self.subTest(phase=phase):
                self.check_pair(stage2, phase + '__box_clip', phase + '__box_clip_preventive')


if __name__ == '__main__':
    unittest.main()
