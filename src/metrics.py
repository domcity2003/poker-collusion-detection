"""Scoring for the three outputs the pipeline produces for every candidate pair.

    score = 0.70 * pair AP  +  0.20 * evidence MAP@5  +  0.10 * behaviour macro-AP

Pair AP ranks pairs by collusion risk. Evidence MAP@5 asks whether the five hands
named for a colluding pair are the ones that demonstrate it. Behaviour macro-AP
averages, over the three collusion families, how well the predicted family ranks
that family's pairs; a family with no true pairs scores zero.

`truth` and `pred` are DataFrames indexed by the same pair ids, with columns
`risk_score`, `predicted_behavior` and `evidence_hand_1` .. `evidence_hand_5`.
In `truth`, `risk_score` is the 0/1 label and `predicted_behavior` the true family.
"""
import numpy as np
import pandas as pd

BEHAVIORS = ('directed_transfer', 'soft_play', 'coordinated_isolation')
EVIDENCE_COLUMNS = tuple(f'evidence_hand_{k}' for k in range(1, 6))
NO_EVIDENCE = 'NO_EVIDENCE'
WEIGHTS = {'pair': .70, 'evidence': .20, 'behavior': .10}


def average_precision(y_true, scores):
    """Mean precision at each true positive, ties broken by input order."""
    y_true = np.asarray(y_true)
    positives = int(y_true.sum())
    if positives == 0:
        return 0.0
    ranked = y_true[np.argsort(-np.asarray(scores), kind='mergesort')]
    hits = np.cumsum(ranked)
    return float(np.sum(hits / np.arange(1, len(ranked) + 1) * ranked) / positives)


def hand_ids(row):
    """Evidence hand ids of one row, without blanks or the NO_EVIDENCE marker."""
    out = []
    for value in row:
        text = '' if pd.isna(value) else str(value).strip()
        if text and text != NO_EVIDENCE:
            out.append(text)
    return out


def evidence_map5(truth, pred):
    """MAP@5 over truly colluding pairs; each pair is capped at min(#relevant, 5)."""
    scores = []
    for t, p in zip(truth.loc[truth.risk_score == 1, list(EVIDENCE_COLUMNS)].itertuples(index=False, name=None),
                    pred.loc[truth.risk_score == 1, list(EVIDENCE_COLUMNS)].itertuples(index=False, name=None)):
        relevant, shown = set(hand_ids(t)), hand_ids(p)[:5]
        if not relevant:
            scores.append(0.0)
            continue
        hits, total = 0, 0.0
        for rank, hand in enumerate(shown, start=1):
            if hand in relevant:
                hits += 1
                total += hits / rank
        scores.append(total / min(len(relevant), 5))
    return float(np.mean(scores)) if scores else 0.0


def behavior_map(truth, pred):
    """Macro AP over the three families; a pair only scores for the family it is predicted as."""
    risk = pred.risk_score.to_numpy(dtype=float)
    out = []
    for family in BEHAVIORS:
        y = (truth.predicted_behavior.to_numpy() == family).astype(int)
        shown = np.where(pred.predicted_behavior.to_numpy() == family, risk, 0.0)
        out.append(average_precision(y, shown) if y.sum() else 0.0)
    return float(np.mean(out))


def composite(truth, pred):
    """All components and the weighted total. Frames must share the same pair order."""
    pair = average_precision(truth.risk_score.to_numpy(dtype=int), pred.risk_score.to_numpy(dtype=float))
    evidence, behavior = evidence_map5(truth, pred), behavior_map(truth, pred)
    total = WEIGHTS['pair'] * pair + WEIGHTS['evidence'] * evidence + WEIGHTS['behavior'] * behavior
    return {'score': total, 'pair_ap': pair, 'evidence_map5': evidence, 'behavior_map': behavior}
