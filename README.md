# Detecting collusion in online poker

Solution for the Kaggle community competition
[Detect Suspicious Value Transfers in Poker](https://www.kaggle.com/competitions/detect-suspicious-value-transfers-in-poker).

Given 2M hands of six-max poker, identify pairs of players who are colluding,
classify how they are doing it, and cite the five hands that prove it.

**Final: 0.84537 public, 109/359.** Median entry scored 0.8034.

## The task

Each of 112,540 candidate player-pairs needs three predictions, scored as
`0.70 x pair AP + 0.20 x evidence MAP@5 + 0.10 x behaviour macro-AP`:

| Component | Weight | What it asks |
|---|---|---|
| Risk score | 70% | How likely is this pair colluding? |
| Evidence | 20% | Which five hands demonstrate it? |
| Behaviour | 10% | Which of three collusion patterns is it? |

Training labels cover only 1,860 of 137,739 development pairs — 372 positive.
Everything else is unlabelled, and some of it is colluding.

## Results

| Submission | Change | Public |
|---|---|---|
| v1 | first end-to-end pipeline | 0.83232 |
| v2 | 3 changes at once | 0.83492 |
| v3 | 3 changes at once | 0.83472 |
| v4 | evidence only: context-feature blend | 0.83942 |
| v5 | risk only: family-specialist blend | 0.84409 |
| v6 | evidence only: per-family blend weight | 0.84436 |
| **v7** | **risk only: context features into the risk model** | **0.84537** |

Every submission that changed exactly one thing improved. Neither submission
that changed three things at once did.

## Two things worth reading even if you don't care about poker

**The obvious validation metric was measuring the wrong problem.** Scoring
locally by treating every unlabelled pair as innocent gave 0.72 while the
leaderboard implied 0.92. The gap is structural, not noise: development labels
1.3% of pairs, so undisclosed colluders are counted as our mistakes, whereas the
real solution labels every colluding pair. Censoring the most suspicious
unlabelled pairs closes it. Judging changes at censor-0 alone would have
rejected the change that produced the largest single gain.

**Collusion does not persist across the phase boundary.** 99.94% of the pairs to
be scored also played together in the training window, which looks like free
extra data. A pair's behaviour in one phase predicts its label in the other at
AP 0.198 against a base rate of 0.200 — pure noise. Testing this first avoided
building a feature set that would have diluted the real signal.

## Layout

```
src/
  prepare.py          partition raw hands, build folds (hash of table_id)
  features.py         identity-free symmetric pair-hand features
  population.py       label-free entry-policy features and contrasts
  hindsight.py        what the cards actually were: folded the winning hand, victim of isolation
  context_features.py action-local context: immediate yielding, folded a private pair, role consistency
  pipeline.py         one assembly path for both phases, so train and inference cannot drift
  final.py            the frozen configuration: fit, lockbox, export
  validate_submission.py  13 pre-submission checks, including that every cited hand
                          was really shared by that pair in the scored window
reports/              every experiment's metrics, including the failures
OPUS_HANDOFF.md       full working log: what was tried, what failed, and why
```

## Reproducing

Competition data is not included and must be downloaded from Kaggle. Then:

```bash
python src/prepare.py          # partition, folds
python src/features.py         # pair-hand features
python src/population.py       # entry-policy features
python src/build_cache.py      # cache the assembled tables
python src/final.py            # fit and export
python src/validate_submission.py artifacts/final/submission.csv
```

One caveat recorded honestly: re-running the pipeline does not reproduce the
v2 predictions bit-for-bit, so the winning lineage is built on the file
downloaded back from Kaggle rather than a local rebuild.

## What didn't work

Documented with numbers in `OPUS_HANDOFF.md`, because the negative results took
as long to establish as the positive ones: phase pooling, PU relabelling of
suspicious unlabelled pairs, seed averaging the evidence rankers (actively
harmful), pooling behaviour families, reverse-engineering the organisers'
evidence-selection rule, graded relevance, and count-vs-rate features.
