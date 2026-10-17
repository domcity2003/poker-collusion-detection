import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from metrics import average_precision, composite, evidence_map5, EVIDENCE_COLUMNS


def frame(risk, behavior, evidence):
    df = pd.DataFrame({'pair_id': list('AB'), 'risk_score': risk, 'predicted_behavior': behavior})
    for k, column in enumerate(EVIDENCE_COLUMNS):
        df[column] = [row[k] for row in evidence]
    return df


class AveragePrecision(unittest.TestCase):
    def test_hand_computed_case(self):
        # hits at ranks 1 and 3: (1/1 + 2/3) / 2
        self.assertAlmostEqual(average_precision([1, 0, 1, 0], [.9, .8, .7, .1]), 5 / 6)

    def test_perfect_and_empty(self):
        self.assertEqual(average_precision([1, 1, 0], [.9, .8, .1]), 1.0)
        self.assertEqual(average_precision([0, 0], [.9, .1]), 0.0)


class EvidenceAndComposite(unittest.TestCase):
    def truth(self):
        return frame([1, 0], ['soft_play', 'none'],
                     [[f'H{k}' for k in range(1, 6)], ['NO_EVIDENCE'] * 5])

    def test_evidence_credit_is_independent_of_risk(self):
        truth = self.truth()
        good = frame([0., 1.], ['soft_play', 'none'], [[f'H{k}' for k in range(1, 6)], ['NO_EVIDENCE'] * 5])
        blank = frame([0., 1.], ['soft_play', 'none'], [['NO_EVIDENCE'] * 5] * 2)
        gap = composite(truth, good)['score'] - composite(truth, blank)['score']
        self.assertAlmostEqual(gap, .2)

    def test_partial_evidence_is_rank_weighted(self):
        truth = self.truth()
        # relevant H1..H5; hits at ranks 1 and 3 give (1/1 + 2/3) / min(5, 5)
        pred = frame([1., 0.], ['soft_play', 'none'], [['H1', 'X', 'H2', 'Y', 'Z'], ['NO_EVIDENCE'] * 5])
        self.assertAlmostEqual(evidence_map5(truth, pred), (1 + 2 / 3) / 5)

    def test_missing_family_scores_zero_in_behaviour(self):
        truth = self.truth()
        pred = frame([1., 0.], ['soft_play', 'none'], [[f'H{k}' for k in range(1, 6)], ['NO_EVIDENCE'] * 5])
        # soft_play ranks perfectly; the other two families have no true pairs
        self.assertAlmostEqual(composite(truth, pred)['behavior_map'], 1 / 3)


if __name__ == '__main__':
    unittest.main()
