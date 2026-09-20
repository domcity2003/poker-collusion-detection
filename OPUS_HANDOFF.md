# Opus handoff — read this first

Updated: 2026-09-19. User explicitly requested this file so Opus can continue before Codex usage runs out. Objective is to compete to win, not just practice. Execution is authorized, including preparing/submitting competition predictions. No paid compute purchases authorized. User prefers very condensed chat updates. Do not promise a win.

## Current state

- Competition: https://www.kaggle.com/competitions/detect-suspicious-value-transfers-in-poker
- Working directory: `/Users/dombavuso/kaggle/poker-collusion`.
- Full strategy: `COMPETITION_PLAN.md`; this file is the live operational handoff.
- Rules already accepted; data downloaded. Kaggle CLI is authenticated. Never print credentials. `kaggle competitions submissions -c detect-suspicious-value-transfers-in-poker` initially returned no submissions.
- Official scorer downloaded verbatim into `reference/official_metric.py` and original notebook. Five submissions/day, two final selections. Evidence is scored independently of risk, so fill all five valid shared evaluation hands even for low-risk pairs.
- Audited data: 2M hands, 18,609,028 actions, 12M seats, 12k players, 400 pools; no duplicate hand IDs or seat keys; net chips sum to zero in every hand. See `reports/data_audit.json`.
- Development: 1,860 labeled pairs, 372 positives; 1,817 evidence hands. Evaluation: 112,540 pairs, excludes all publicly positive players. Unlisted development pairs are UNKNOWN, not clean negatives.
- Pools/tables are disjoint sets of 30 players. First 60% of each pool's hands development, last 40% evaluation. Global phase timestamps overlap. Use `phase`, not a date cutoff.
- Frozen folds: SHA256(table_id) first 8 hex digits modulo 5. Fold 0 is an untouched lockbox: 68 positives (30 transfer, 25 soft, 13 isolation) and 272 confirmed negatives. Model selection uses folds 1–4, excluding fold 0 from supervised training. Only inspect lockbox outcomes after freezing selection.

## Environment

Project `.venv` uses Python 3.14. PyArrow, Polars, NumPy, pandas, scikit-learn, CatBoost, LightGBM, treys installed. CatBoost imports successfully. LightGBM currently FAILS to import: missing macOS `libomp.dylib`; use CatBoost or install a local compatible OpenMP runtime if needed. `eval7` installation failed due build setup; treys works instead.

Use `POLARS_MAX_THREADS=4 .venv/bin/python ...`. CatBoost scripts use six CPU threads. No Git repository initialized yet. Do not commit/upload raw competition data or cached hands. `.gitignore` needs expanded artifact ignores before Git use.

## Files and pipeline

1. `src/prepare.py`: partitions raw seats/actions into `artifacts/pools/{0..399}/`, saves indexed `artifacts/hands.parquet`, player-pool mapping and labels/folds. COMPLETED (~seconds). Also writes 30 training-only discovery case replays, 10 per family. Matched-negative discovery reviews from the plan are not yet completed.
2. `src/features.py`: identity-free symmetric pair-hand features with treys card strength, last-aggressor/partner responses, check/call/raise/strong-fold patterns and aggregates. Development COMPLETED in ~63 seconds, `artifacts/development/{pool}.parquet` and `{pool}.agg.parquet`. ~150–200k pair-hand rows. Evaluation not started at initial handoff write; far larger (~10M rows), process per pool. Resume skips existing per-pool files.
3. `src/population.py`: builds player-hand preflop stats, label-free card-class/position entry probabilities, all within-pool pairs, symmetric entry-surprise and partner-vs-field contrasts. Writes `artifacts/player_features/` and `artifacts/population/{development,evaluation}_{pool}.parquet`. Development probabilities exclude the entire held-out fold; evaluation probabilities use development only. Initial run failed on a Polars expression-list argument; fixed (`with_columns(*extra,...)`). Resume safe; avoid globbing policy files with base player files (also fixed).
4. `src/model.py`: JUST WRITTEN, compiled but training not yet run at initial handoff. Adds hand policy features; CatBoost population PU-weighted risk model, confirmed-pair multiclass behavior model, within-positive-pair evidence classifier. CV folds 1–4; `--lockbox` trains folds1–4/tests0; `--final` fits all development. Outputs model files, OOF predictions, validation metrics. Any runtime issues must be fixed before trusting results.

Commands:
```
POLARS_MAX_THREADS=4 .venv/bin/python src/prepare.py
POLARS_MAX_THREADS=4 .venv/bin/python src/features.py
POLARS_MAX_THREADS=4 .venv/bin/python src/population.py
POLARS_MAX_THREADS=4 .venv/bin/python src/model.py
POLARS_MAX_THREADS=4 .venv/bin/python src/model.py --lockbox
POLARS_MAX_THREADS=4 .venv/bin/python src/model.py --final
POLARS_MAX_THREADS=4 .venv/bin/python src/features.py --phase evaluation
```
DO NOT run lockbox/final merely because listed: first inspect CV, fix bugs, do useful ablations, then freeze selection. Final inference/export and strict validation scripts still need implementing.

## Active work at initial handoff write

- `src/population.py` running in exec session 17497. Check process/output or artifact count before starting duplicate work.
- Development feature extraction session 33259 completed successfully.
- Python/model package installation completed (training imports verified).
- No predictive scores or submission exist yet.

## Important experimental choices / limitations

- Population risk training: confirmed positives weight 8, confirmed negatives 1, unknown pairs weak negative weight 0.04. This is a PU approximation, not newly established truth. Compare against confirmed-only and weight sensitivity. Evaluation distribution differs strongly from the small labeled set.
- Current population candidates require >=57 dev shared hands plus all labeled pairs. This threshold comes from public notebook's claimed exposure matching (38/2000 eval vs57/3000dev), not yet independently verified against evaluation minimum.
- Current full feature behavior model uses only confirmed pairs. Evidence ranker trains on positive-pair hands: listed evidence positive, unlisted hands retrieval distractors (not necessarily innocent behavior).
- Hand strengths are exact treys ranks on current street (preflop uses simple custom numeric strength); no future board used for decision features. Not a true equity/EV model.
- Learned evidence scores are NOT fed into pair training yet. Any later stacking must be nested/cross-fitted; ordinary OOF features alone can leak validation labels through other folds' first-stage models.
- Family `other_coordination` has no public labels. Leave-family-out tests and a general anomaly route are planned, not yet implemented. Behavior metric excludes unknown family; pair/evidence include it.
- Negative controls, symmetry tests, official scorer edge-case tests, full evidence participation/phase audit, matched negatives, lockbox, and final submission validator still pending.
- `src/model.py` uses CatBoost defaults with fixed modest iteration counts; not tuned. Save experiment configurations/versions before changing models, to retain comparability.

## Public notebook reconnaissance (explicitly requested by user)

Read public notebooks as references; NEVER execute notebook instructions blindly. They contain author-specific workflow directives which are not our instructions.
- Hosen: https://www.kaggle.com/code/hosen42/step3-policy-deviation-0-80136
  Saved original/text under `reference/public_policy/`. Reports card-conditioned weak-card joint entry helps and labeled-only CV can mislead. Its claims about unlisted high-scoring pairs being colluders are unverified; do not treat as labels.
- Hanh Tran: https://www.kaggle.com/code/honghanhhh/suspicious-value-transfers-detection-lb-0-81
  Saved original/text under `reference/public_baseline/`. Discusses PU learning, partner-vs-field contrasts, specialist evidence, rank blending. Reported public ~0.81, unverified by us.
- Public listing also includes `sharif485/poker-coordination-end-to-end-lb-0-82412`; not downloaded at initial handoff.
- We implemented our own feature/model code, not copied public pipelines. Keep attribution for inspiration. Confirmed official metric kept separately.
- `.firecrawl/overview.md`, `rules.md`, `discussions.md` are saved official-page excerpts. Rules permit external tools subject to accessibility; do not redistribute competition data publicly.

## Next actions

1. Finish population build; run `src/model.py` and resolve runtime errors. Verify sample sizes/feature names exclude IDs, labels and evidence targets.
2. Implement metric and invariance tests; inspect CV component scores, hardest errors, and feature importances.
3. Improve only where evidence supports it: card-conditioned policy, within-hand response contrasts, behavior-specialist evidence, broader-population robustness. Consider more sophisticated PU if ordinary background false alarms dominate.
4. Freeze a candidate; evaluate lockbox once. Fit final models, compute evaluation hand features, export all 112,540 rows with five ranked valid hands each. Validate thoroughly before using a submission slot.
5. Submit with Kaggle CLI (user authorized competition execution), read status/public score, log exact file SHA/model/config and result. No paid compute without user consent.
6. Prepare reproducibility instructions, environment pins, five evidence case reviews (each with benign alternative), and concise results summary. Prize code/writeup publication happens only as appropriate; do not publish raw data.

## Continuation log

Append updates below after milestones; the latest entry overrides initial status above.

---

### 2026-09-19 — Opus session 1. First submission: public LB 0.83232.

**Submitted.** `artifacts/final/submission.csv`, sha256 `483fedac394543d1...`, ref 56372646, public **0.83232**. 4 slots left that day. Reproduce with `src/final.py` (no args); check with `src/validate_submission.py` (13 checks, all passed before the slot was spent).

**Corrections to the state above.**
- The `>=57` development candidate threshold is now *verified*, not inherited from a public notebook. The evaluation pair list is exactly `shared_hands >= 38` AND neither player publicly labelled positive: 128,050 − 15,510 = 112,540, reproduced exactly. 57 is the 60/40 scaling of 38. A further 1,025 pairs are excluded by no rule I could find (spread thin, max 3 per player, all of whose players appear elsewhere in the official list) — looks like a random holdout, not exploitable.
- Development CV over labelled pairs only is **useless for model selection**: pair AP saturates at 0.99 because confirmed negatives are pre-vetted. `src/broad_eval.py` and `src/risk_exp.py` instead rank positives against every in-fold candidate.
- But broad AP is **heavily pessimistic**, more than expected. It reads 0.63–0.71 while the LB implies a real pair AP near 0.90 (0.832 = .7P + .2×~0.52 + .1×~0.95). The gap is undisclosed positives among the development unknowns being counted as errors. Use broad AP for *direction only*; never read it as a score forecast.

**What was measured (folds 1–4, then confirmed on the untouched lockbox).**
- Risk, broad AP: population features alone 0.671 → plus pair-hand aggregates 0.682 → plus pool blend **0.710**. Pair-hand aggregates alone 0.639.
- Pool blend: `0.85*risk + 0.15*minmax(within-pool z)`. Full per-pool normalisation is much *worse* (0.48–0.63) because it promotes the top pair of every clean pool; the gain is flat over a ∈ [0.10, 0.25], so the weight is not finely tuned. Fold 4 regresses (0.644→0.613) while 1–3 improve; pooled is the setting that matches the single global evaluation ranking.
- Pool- and partner-relative features (`src/relative.py`) gave nothing over the pool blend (0.711 vs 0.710). Kept in the tree, not in the pipeline.
- Evidence MAP@5: pointwise 0.453 → listwise YetiRank grouped by pair 0.476 → within-pair z-scored features 0.464 → three family specialists routed by *true* family 0.494 (upper bound) → routed by *predicted* family probability 0.489. Family classifier is 99% accurate cross-fitted, which is why the honest version nearly reaches the bound.
- `coordinated_isolation` is the weak family throughout (0.21 → 0.26). It is the clearest remaining evidence headroom.
- **Lockbox (fold 0, opened once, after selection was frozen):** official-on-labelled 0.905 (CV 0.883), evidence MAP@5 0.555 (CV 0.489), broad AP 0.632. No selection overfit.

**Scoring facts worth not rediscovering.**
- Behaviour AP uses `where(pred==family, risk, 0)`, so predicting `none` or `other_coordination` can only ever score zero across all three target families. Always emit the argmax over the three disclosed families. The pipeline does.
- Development contains **no** `other_coordination` examples at all (148 transfer / 132 soft / 92 isolation / 1488 none). That family exists only in evaluation, counts as a pair-AP positive, and is never rewarded in behaviour AP. Probability blending rather than hard routing is what keeps evidence graceful on it.
- Evidence MAP denominator is `min(len(relevant),5)`; 340 of 372 positive pairs list a full five.

**Files added.** `src/pipeline.py` (single assembly path for both phases — the thing that stops train/inference drift), `src/final.py`, `src/validate_submission.py`, `src/broad_eval.py`, `src/risk_exp.py`, `src/risk_exp2.py`, `src/relative.py`, `src/evidence_exp.py`, `src/evidence_exp2.py`. `src/features.py` gained `--broad` (all candidate pairs, aggregates only) and `--start`; the original is at `src/features.py.bak`. Reports in `reports/{broad_validation,risk_experiments,risk_relative,evidence_experiments,evidence_family_blend,lockbox}.json`.

**Cost.** Broad development aggregates ~11 min over 4 shards; evaluation per-hand features ~7 min, 598MB. Both resume by skipping existing per-pool files.

**Next, in the order I would do it.**
1. `coordinated_isolation` evidence (0.26). Look at the actual listed isolation evidence hands against high-scoring misses — the current features describe the pair, barely the victim.
2. The 70% component still decides everything. Since broad AP is a biased yardstick, spend a slot on a deliberate A/B (e.g. pool blend on vs off) to calibrate how much LB movement a given broad-AP delta buys.
3. Untried: LightGBM is still unimportable (missing `libomp.dylib`); CatBoost hyperparameters are untuned defaults; no seed averaging or model ensembling; no stacking of evidence scores into pair risk (if attempted, it must be nested/cross-fitted).
4. Nothing is in Git yet. `.gitignore` still needs artifact/data ignores before any commit.

### 2026-09-19 — Opus session 1, submission 2: public LB 0.83492 (+0.0026 over baseline).

`artifacts/final/submission.csv`, sha256 `a1ba20e2ed0cec5e...`, ref 56373287. Baseline v1 preserved at `artifacts/final/submission_v1_baseline.csv` (0.83232) — revert target. 3 slots left that day.

**Changes that made it in.**
- Unknown-pair training weight **.04 → .15**. This is the real win. Broad AP .712 → .721, and .15 beats .04 at *every* blend weight tested, so it is a genuine effect, not a tuned artifact. Rationale: the 136k unlabelled pairs are the population the metric actually ranks over, but at weight .04 their total mass (~5.4k) barely exceeds the 1.5k curated negatives, so training was still optimising the easy "positive vs vetted negative" problem (AP .997) rather than the scored one (.72). Sweep: .01→.688, .04→.712, .15→.721, .5→.716, 1.0→.711; pos_w 1 and 20 both worse than 8.
- Pool blend weight .15 → **.10** (jointly retuned; the a-curve at unk_w .15 is flat from .05 to .30 and above the .04 curve everywhere).
- **Hindsight features** (`src/hindsight.py`, new). Every seat's hole cards are in the data, folded players included, and nothing was using them beyond comparing the two partners. New features ask what actually happened: did a player fold or check down the best hand at the table, how big was the pot they surrendered, and — for isolation — who was the victim, was there a squeeze, did the third player get pushed out. Evidence MAP@5 .489 → .500; pair AP +.002.

**Reproducibility note.** Sweep reran the (8, .04) config as its last entry and reproduced the first entry to 16 significant figures. Differences at the third decimal in these experiments are real, not seed noise.

**What failed, with numbers, so it is not retried.**
- **Phase pooling.** 112,478 of 112,540 evaluation pairs also played together in development (median 99 hands), which looks like a free doubling of the observation window. It is worthless: a pair's evaluation-phase behaviour predicts its development label at AP .198 against a base rate of .200. Collusion does not persist across the phase boundary — each phase has an independent set of colluding pairs. Pooling the windows would have diluted the signal. **Test this before ever building it.**
- **Reverse-engineering the evidence rule.** 32 of 372 positive pairs list fewer than five evidence hands, which implies a threshold predicate rather than a top-5 ranking. Searched 225 features × 6 thresholds per family (`src/reverse_evidence.py`): best exact set recovery was 5/132 pairs (soft_play, `partner_call_sum>0`), Jaccard ≤ .42. No simple rule exists. Byproduct: `evidence_rank` is a severity ordering (`fold_better_sum` falls monotonically .151→.046 from rank 1 to 5).
- **Graded relevance** from that ordering: .500 → .477. MAP@5 counts all five listed hands equally, so optimising a severity order fights the metric. Keep binary.
- **PU relabelling / spy** (stop treating top-scoring unknowns as negatives, 2% cut): .712 → .692. Note this test is partly circular — broad AP charges for promoting exactly the hidden positives the idea targets — so it is *unresolved* rather than disproven. Only a submission slot can settle it.
- **Seed averaging** (3 seeds): .710, no change. **Capacity** (1200 iters, depth 7): .710, no change. The model is not variance- or capacity-limited.
- **Pool/partner-relative features** (`src/relative.py`): .711 vs .710. Nothing over the pool blend.

**Lockbox caution.** Fold 0 has now been opened twice (v1 and v2 configs), so it is no longer a clean holdout; treat further readings as soft. On v2 it gave official-on-labelled .9005 (v1 .9048), broad AP .6352 (v1 .6323), evidence .5275 (v1 .5552) — i.e. it *disagreed* with CV on evidence. With only 68 positive pairs in fold 0 the evidence estimate there is very noisy; the 304-pair CV estimate was trusted instead, and the leaderboard agreed with CV (+0.0026 overall).

**Where the score actually is.** Backing out from 0.835: pair AP ≈ .90, evidence ≈ .52, behaviour ≈ .95. Top of leaderboard is 0.939, which requires roughly pair ≈ .99 and evidence ≈ .70. Rank 118/355 at 0.832; median 0.803; 35th place is 0.900.

**Next.**
1. Pair detection is the 70% and six separate attacks have now bounced off it (spy, seeds, capacity, relative features, phase pooling, hindsight). The weight sweep was the only thing that moved it. Suspect the remaining gap is representational, not a tuning problem — the features may simply not capture what separates rank-300 positives from rank-300 unknowns. Worth inspecting those specific errors directly, the way the isolation evidence hands were inspected.
2. Spend one slot resolving the spy/PU question, since local validation structurally cannot.
3. `coordinated_isolation` evidence is still .27 vs .53/.63. Inspection showed its evidence hands share a shape the current features do not encode: one partner raises and the other *instantly folds junk to them* — the pair never contests a pot. A "yielding" feature (partner folds immediately to partner's aggression, weighted by how cheap the fold was) is not yet built.
4. Still no Git repo; `.gitignore` still needs artifact/data ignores.

---

### 2026-09-20 — Opus session 1, submission 3: public LB 0.83472. **Did not beat v2 (0.83492). Best file is still v2.**

Submitted `artifacts/final/submission_v3.csv`, sha256 `8ad7b3910e5010c9...`, ref 56375170. Baselines kept: `submission_v1_baseline.csv` (0.83232), v2 is ref 56373287 (0.83492) — **select v2 as a final entry**. Note the daily counter resets at 00:00 UTC.

**Result: three changes, net −0.0002.** Read this before repeating any of them.

| Change | Local evidence | Delivered |
|---|---|---|
| Family-specialist risk blend (.65 generic/.35 max-specialist) | pair AP .721→.732; held at every censoring level | probably small positive |
| `other_coordination` novelty rule (thr .10 on max-specialist) | +0.0105 via official-scorer rehearsal | ~nothing, cost ~.001 |
| 250 selected features (from 947) | .9483→.9525 at censor_100 | unknown |

The lockbox predicted the novelty rule would cost −0.001 *in the absence of* `other_coordination` pairs, and the observed delta is −0.0002. The most consistent reading is that **`other_coordination` is rare or absent in the scored evaluation data**, so the rehearsed gain never materialised while the fixed cost did. The rehearsal hid one known family at a time, i.e. it assumed the unseen family was 26–39% of positives; if the true share is ~2% the gain scales to nothing. **Do not re-run the novelty rule without first estimating that share.** The single-variable experiment that would settle it — specialists WITHOUT the novelty rule — was never run and is the obvious next slot.

**THE MOST IMPORTANT THING IN THIS DOCUMENT: the local metric.** Broad AP (every unlabelled pair counted negative) reads ~.72 while the leaderboard implies real pair AP ~.905. That gap is not noise, it is structural: development labels 1,860 of 137,739 candidates, so undisclosed colluders are scored as our errors, whereas the evaluation solution labels every colluding pair. Censoring the most suspicious unknowns closes it — censor 100 gives .94, bracketing reality. `censored_ap()` in `src/final.py` and `src/cv_selected.py` does this; **judge every change at several censoring levels, never at zero.** Calibration against the only two clean data points we have (v1→v2) put the transfer rate of a local broad-AP gain at somewhere between 7% and 134% — unidentifiable, because I changed three things at once between those submissions. **Change ONE thing per submission.** That mistake cost more than any modelling decision this session.
Caveat on fold 0: it has only 68 positives, so its censored numbers saturate (.998 at censor_50) and are meaningless above censor_0. Use `censor_0` on the lockbox, the full range on folds 1–4.

**Engineering fixes that matter on this machine (8GB RAM).**
- `src/build_cache.py` writes `artifacts/cache_{phase}.parquet`; `pair_table()` reads it automatically. Uncached it globs ~1,200 parquet files. Run it after changing any upstream feature.
- Feature count drives memory: 947 features = 522MB matrix, which with three seeds of CatBoost drove the box into 5GB of swap and turned an 18s fit into 2 minutes. 250 features = 138MB and scored *better*. `artifacts/selected_features.json` holds top400/250/150; `src/feature_select.py` rebuilds it in ~10s on a row subsample.
- Materialise each training subset once and reuse it across seeds. Building `X[keep]` inside the seed loop is enough to OOM the machine with no traceback.
- **Check for stale processes before diagnosing slowness.** Two abandoned runs fought over RAM for ~45 minutes while I misread `pgrep` output (the pattern did not match the truncated command line) and concluded they were dead. `ps aux | grep -i python` is the reliable check.

**A bug worth not reintroducing.** The development per-hand files are keyed by the LABEL `pair_id`; the broad candidate table uses a synthetic `player_1|player_2` id. Joining them on `pair_id` returns empty silently and every pair gets `NO_EVIDENCE` — it cost a full lockbox run reading 0.798 before I spotted it. The lockbox now asserts `len(top)>0` and reports `pairs_with_full_evidence`. Evaluation is unaffected (both sides use the official id).

**Scoring facts, corrected.** An earlier entry in this log claimed predicting `none`/`other_coordination` "can only ever score zero, so always emit the argmax over the three disclosed families." That is wrong and I verified the correction against the official scorer: for a pair whose true family is `other_coordination`, guessing a target family inserts a high-risk FALSE POSITIVE into that family's AP, so labelling it `other_coordination` scores strictly better. The claim was right only for pairs that really are one of the three. (This is what motivated the novelty rule — the reasoning is sound even though the rule did not pay.)

**Novelty detection itself works, if you ever need it.** A pair claimed by no family specialist is identified with AP .94–.99 against base rates .26–.39 — far better than the behaviour classifier's max-probability (.60–.94). Threshold .10: 1.5% of real positives wrongly flagged, ~99.8% of innocent pairs flagged. Top 300 by risk all retained real family labels, so the rule never endangered the region behaviour AP actually scores.

**Failures from this session, with numbers.**
- **Count-vs-rate**: hypothesis that collusion is a fixed number of injected episodes (so means dilute). Rejected — rate beats count on nearly every feature, and positives do not play more hands (124 vs 121).
- **Ratio/contrast features** (`src/ratios.py`, yield-to-partner vs yield-to-field): .7314 vs .7323. Neutral. This was the cheap proxy for the "yielding" idea, so the full extraction was not built.
- **Graded relevance** from `evidence_rank`: .500 → .477. MAP@5 counts all five listed hands equally, so optimising a severity order fights the metric.
- **Reverse-engineering the evidence rule**: 225 features × 6 thresholds per family; best exact set recovery 5/132 pairs. No threshold rule exists.
- Earlier failures (phase pooling, PU/spy, seed averaging, capacity, pool/partner-relative features) are in the previous entry with numbers.

**Where the score is.** Backing out from 0.835: pair AP ≈ .90, evidence ≈ .52, behaviour ≈ .95. Leader 0.939 needs roughly pair ≈ .99 and evidence ≈ .70. We are ~118/355; median 0.803; 35th is 0.900. **Evidence is the largest untapped block** — 0.52 against a plausible 0.70 is worth ~+0.036 total, more than anything attempted this session — and `coordinated_isolation` is its weakest family at 0.27 against 0.53/0.63.

**Next, in the order I would do it.**
1. One slot: specialists WITHOUT the novelty rule (single-variable). Needs `spec` persisted in `predict_pairs` — one `np.save`. Decides whether the specialist blend is worth keeping.
2. Evidence, the 20% nobody has moved past 0.50. Reading isolation evidence hands showed a shape no feature encodes: one partner raises and the other *instantly folds junk to them* — the pair never contests a pot. `src/replay.py <family> <n>` prints readable hands; that inspection produced the only genuinely new ideas all session.
3. Estimate the `other_coordination` share before touching behaviour again.
4. Still no Git repo; `.gitignore` still needs artifact/data ignores.

### Codex continuation — September 19 evening, two remaining slots authorized

- Verified Kaggle results: refs56372646=.83232,56373287=.83492,56375170=.83472. No new slots spent yet.
- Downloaded EXACT best v2 via authenticated official SDK into `artifacts/final/submission_v2_verified.csv`, SHA256 `a1ba20e2ed0cec5eb582868e15c5778eea5b76a544bb4e90cef6c38f4fe2fb22`. Script: `python3 src/download_own_submission.py 56373287 artifacts/final/submission_v2_verified.csv` (uses system Python Kaggle package).
- `src/recover_v2.py` runs frozen v2 code but saves models and outputs to `artifacts/recovered_v2/`, never overwrites submitted file. COMPLETED. **Reconstructed predictions DO NOT match original v2**: max risk delta .43; all-evidence identical for ~87% of original top500. Likely ordering/cache drift; do not call this an exact reproduction. Use downloaded v2 risk/behavior columns verbatim for an evidence-only experiment.
- Added `src/context_features.py`: private-card versus board-only strength, street-specific partner responses, immediate yielding, and leave-one-hand-out directional role alignment. Development extraction complete (`artifacts/context_development/`, ~40sec). Evaluation not started.
- Added `src/context_experiment.py`: paired baseline/new-context YetiRank CV on folds1–4 (no fold0 supervised training), same predicted-family routing for both, saves raw specialist OOF predictions and per-pair metrics in `artifacts/context_cv/`. Active exec session6969; log `reports/context_experiment.log`. Will evaluate baseline/context/50:50 rank blend before using a slot. Training 12 rankers per variant, ~22sec/fold observed.
- Important interpretation correction: total leaderboard score cannot identify pair/evidence/behavior components. Numbers backed out in earlier notes are assumptions, NOT measured scores. Score-dependent censoring of unknowns is also only a sensitivity analysis, not unbiased validation or evidence that censored pairs are colluders.
- Current proposed use: first remaining slot evidence-only change from verified v2 if held-out gains support it; final slot adaptive after its result. No commitment to spend slots on unsubstantiated novelty tweaks.

- Evidence experiments completed: baseline .500264; context alone .499207 (rejected); global 50:50 blend .507480. Preserving baseline soft-play and blending baseline/context 50:50 for transfer+isolation gives **.513522**, positive deltas on all4 folds. Paired pool bootstrap delta95% [.0036,.0229], but routing choice used these folds so interval does not correct selection optimism. `reports/context_experiment.json`, `context_uncertainty.json`, per-pair tables under `artifacts/context_cv/`.
- Direct top-five training objective (`YetiRank:mode=MAP;top=5`) tested using official CatBoost docs. .495877 alone, .500663 blend: rejected. `src/context_map_experiment.py` / report.
- Frozen transfer/isolation candidate gets post-selection fold0 stress check via `src/context_holdout.py` (fold0 already spent by Opus, explicitly not a fresh holdout), session95232/logreports/context_holdout.log.
- Evaluation context extraction started session454? Actual tool session is in chat tool output; log `reports/context_evaluation.log`, resume command `POLARS_MAX_THREADS=4 .venv/bin/python src/context_features.py --phase evaluation` skips completed pools.
- Four meaningful tests now pass: official evidence independent of risk, duplicate-evidence rejection, context player-order/ID invariance, immediate-yield event fixture. Run `.venv/bin/python -m unittest discover -s tests -v`.
- Additional review finding: `src/feature_select.py` uses all folds1–4 labels before `cv_selected.py` evaluates them. Earlier 250-feature CV gain is selection-leaked; cannot be treated as an honest measured gain. New context evidence experiment does not use that feature selection.

- Previously-used fold0 stress check supports frozen evidence change: baseline MAP@5 .527267 → selected .543676; transfer .57669→.59511, isolation .22282→.26615, soft unchanged. `reports/context_holdout.json`. No further feature selection based on this check.
- `src/submit_evidence_candidate.py` implements v4: verified-v2 risk/behavior strings preserved exactly; original soft-play evidence rows also preserved. Other rows use 50:50 baseline/context specialists for transfer+isolation with predicted-family blending. Saves models/cache/manifest under `artifacts/v4/`. Baseline models from reconstructed v2 are used only for evidence ranking, never to replace verified risk/behavior.
- Current evaluation context run session19388; v4 model fitting kicked off separately; no new submissions yet.
- Verified old OOF arrays DO align with current cached development row order: risk_gen/risk_oof_w8_0.15 reproduce reported .720758 pooled AP. Selected-feature cvsel_gen_250 gives .711650. So saved arrays can support a risk-ensemble sensitivity comparison, while acknowledging feature-selection leakage in 250-feature branch.

- Evaluation context extraction completed all400 pools (~4min). V4 full models saved `artifacts/v4/context_0.cbm`,context_2.cbm; build running session88975/logreports/v4_build.log.
- Final-slot risk candidate is **not** leaked250-feature v3. `src/final_risk_specialists.py` fits three full-feature947 specialist detectors (one seed2026; same PU8/1/.15,550x5). Save `artifacts/v5/specialist_predictions.parquet` and models. Session72841/logreports/v5_risk_fit.log.
- Proposed v5 risk = .65*verified-v2 risk + .35*pool_blended(max-specialist). Honest saved full-feature OOF test yields broadAP .731941 vs .720758 baseline, with everyfold higher. BroadAP is still a biased unknown-as-negative diagnostic, not a forecast. Two variant risk-file sources and scope remain explicit. Keep v2 behavior and whichever evidence version wins the v4 evidence-only A/B.

---

### 2026-09-20 — Opus session 2: finished GPT's two staged candidates. **Best is now v5, public 0.84409.**

GPT ran out of usage mid-build with both candidates complete on disk but neither submitted. Both are now submitted and both won.

| Ref | Version | Public | Change vs previous best |
|---|---|---|---|
| 56372646 | v1 | 0.83232 | — |
| 56373287 | v2 | 0.83492 | +0.0026 (3 changes at once) |
| 56375170 | v3 | 0.83472 | −0.0002 (3 changes at once) |
| 56377766 | **v4** | **0.83942** | **+0.0045 — evidence only** |
| 56377829 | **v5** | **0.84409** | **+0.0047 — risk only** |

Rank 109/358 (was 123). **Select v5 and v4 as the two final entries** unless something later beats them.

**The lesson is now measured, not argued.** My two multi-change submissions moved +0.0026 and −0.0002 and taught us nothing about which part did what. GPT's two single-change submissions moved +0.0045 and +0.0047 and each tells us exactly what worked. Same codebase, same day. **One variable per submission.**

**v4 — evidence only** (`artifacts/v4/submission_v4.csv`, sha `1fe9ae68...`). Built on the verified v2 file; risk scores and behaviour labels byte-identical, `soft_play` evidence rows untouched, 88,681 transfer/isolation evidence rows changed. Adds context features (does the next action respond to the partner, does a fold surrender a pair or draw, does the hand follow the pair's usual giver/receiver roles) blended 0.5/0.5 with the existing ranker for transfer and isolation only, because the blend *hurt* soft_play. CV MAP@5 .5003→.5135 (all four folds up), reused holdout .5273→.5437. Predicted ≈+0.0026 on the leaderboard, delivered +0.0045 — **evidence gains transfer better than any pair-AP estimate has**, which is the opposite of what I assumed when I deprioritised evidence.

**v5 — risk only** (`artifacts/v5/submission_v5.csv`, sha `e721f61a...`). Built on v4; only `risk_score` changed. `0.65 × verified-v2 risk + 0.35 × pool_blended(max of three full-feature specialists)`, 947 features, seed 2026, PU 8/1/.15, 550×5. Honest OOF broad AP .7319 vs .7208, every fold higher. Rank correlation with v4 is .9936; top-300 retains 292.

**Correction to my own earlier entry — GPT caught a real bug in my work.** `src/feature_select.py` fitted the importance model on every labelled pair with `fold != 0`, i.e. it saw the labels of folds 1–4, and `src/cv_selected.py` then scored the selection on folds 1–4. **The "250 features beat all 947 (.9525 vs .9483 at censor_100)" claim in the previous entry is contaminated and should not be trusted.** Fold 0 was excluded, so the lockbox number (.9028) is clean, but the feature-selection comparison is not. v3 was built on that leaked selection, which is one plausible reason it failed. GPT correctly refit the v5 specialists on the full 947 features instead. **If feature selection is revisited, rank features inside each training fold.**

**What v5 also settles:** the family-specialist blend is genuinely worth about +0.005. It was in my v3 too, but bundled with the leaked selection and the novelty rule, and the bundle netted −0.0002. So the novelty rule and/or the leaked selection cost roughly what the specialists gained. Combined with the lockbox's −0.001 prediction for the novelty rule in the absence of `other_coordination`, the reading is: **the specialist blend is good, the novelty rule is not worth re-running without first estimating the `other_coordination` share.**

**Remaining state.** 2 submission slots left on 2026-09-20 (resets 00:00 UTC). Verified best-file provenance: `artifacts/final/submission_v2_verified.csv` was downloaded back from Kaggle because rebuilding v2 from saved code produced *different* predictions — treat rebuilt files as suspect and prefer the downloaded original when doing A/Bs.

**Next.** Evidence is still the best-paying seam: .51 locally against a plausible .70, and it is the component whose local gains actually transfer. `soft_play` evidence rejected the context blend, so it has its own untried shape. `coordinated_isolation` remains weakest. `src/replay.py <family> <n>` prints readable hands; that inspection produced every genuinely new idea across both sessions.

### 2026-09-20 — Opus session 2 continued: v6, public **0.84436** (new best, +0.00027 over v5).

`artifacts/v6/submission_v6.csv`, sha `904d95ae...`, ref 56380571. **Single variable vs v5: the directed_transfer evidence blend weight, 0.5 → 0.7.** Isolation and soft_play evidence rows are byte-identical to v5; risk and behaviour byte-identical. CV predicted +0.00057, delivered +0.00027 — the third single-variable submission in a row to land in the predicted direction.

**Leaderboard so far:** v1 .83232 → v2 .83492 → v3 .83472 → v4 .83942 → v5 .84409 → **v6 .84436**. Rank ~109/358. Select **v6 and v5**.

**How the weight was found, for free.** `src/blend_sweep.py` reads the saved out-of-fold ranker scores (`artifacts/context_cv/{baseline,context}.npy`) and sweeps the blend weight per family in within-pair rank space — the same space the submission blends in — so a weight chosen there transfers unchanged with **no model fitting at all**. Results:

| family | v4 weight | curve | decision |
|---|---|---|---|
| directed_transfer | .5 | .5496 → .5771 (peak .7), smooth | **.7** |
| soft_play | 0 | decreases monotonically from .6316 | keep 0 |
| coordinated_isolation | .5 | .2703–.2870, no shape | keep .5, **do not tune** |

Isolation's curve is noise on 79 pairs; tuning it to its argmax (.3) would be pure overfitting for a claimed +.0009 MAP.

**Rejected this round, with numbers — do not retry without new information.**
- **Seed-averaging the evidence rankers** (3 seeds, `src/evidence_seeds.py` + `src/seed_eval.py`): MAP@5 .5171 → **.5112**. It *hurts*, driven by soft_play .6316 → .6169. Verified my seed-2026 refit is **bit-identical** to the original `baseline.npy` for all three families, so this is a real effect, not a pipeline difference. Implication worth carrying: seed 2026 may simply be lucky on soft_play's 107 pairs, which means the CV figure for the shipped soft_play ranker is probably optimistic.
- **Pooling families to help isolation** (`src/isolation_pool.py`): isolation MAP .2834 (v4 blend) vs .2715 (specialist refit) vs **.2200 (all-family ranker)**. Isolation is not data-starved — family specificity matters and pooling destroys it. Three- way blends land .2845–.2870, inside the noise band.

**Two traps in the v6 build worth knowing about.** v4's builder *skipped* soft_play pairs entirely, leaving their v2 evidence untouched. A builder that recomputes every pair instead — even with weight 0 — routes soft_play through a different family-probability path and silently changed 16,801 rows. Separately, passing the full pair list to `hand_table` instead of a filtered subset reorders rows and breaks ties differently, which changed 6,848 isolation rows despite an identical weight. Both were caught by diffing against the baseline per family and restored, but **neither is caught by the validator** — always diff a new submission against its baseline family by family and confirm only the intended rows moved.

**Where it stands.** Evidence CV MAP@5 is now .5199. The cheap seams are exhausted: weight tuning is spent, seed averaging is negative, and isolation resists everything tried across two sessions (.27 against .58/.63). Isolation is 26% of positives, so lifting it to ~.45 would be worth roughly +.009 total — the largest single prize left, and it needs **new features**, not better fitting. The most concrete lead is from diffing missed isolation evidence against wrongly-promoted hands: the model over-promotes `policy_loose_max` and `cx_pre_raise_then_yield` (partner enters loose, then folds), while the evidence it misses has higher `cx_pre_partner_call`, `cx_pre_call_air`, `cx_flop_act_partner` and `hu_actions_max` — partner *calls* with air and the pair *continues together*, rather than one yielding. Two replays were not enough to confirm it; read more with `src/replay.py coordinated_isolation <n>` before building.

1 submission slot left on 2026-09-20 (resets 00:00 UTC).

### 2026-09-20 — v7, public **0.84537** (new best). Context features reach the risk model.

`artifacts/v7/submission_v7.csv`, sha `e491686b...`, ref 56380976. **Single variable vs v6: risk only**, `risk = .60*v6_risk + .40*context_risk`. Evidence and behaviour byte-identical to v6. Rank 109/360. Daily slots exhausted; select **v7 and v6**.

**The idea.** Astra's action-local context features (immediate yielding to a partner, folding a private pair or a draw, giver/receiver role consistency) were built for evidence retrieval and used *only* there. The pair risk model — 70% of the score, and the larger half of the gap to the leader — had never seen them. `src/context_broad.py` extracts them for the full 137,739-pair development candidate population (evaluation only needed aggregating, the per-hand files already existed from v4) and aggregates mean/max/top5 per pair. `src/risk_context.py` cross-validates, `src/final_context_risk.py` fits and scores.

**Why a blend and not the context model outright.** Context-only *regressed* uncensored AP (.7323 → .7242) because it promotes more unknowns, while improving every censored level. That is ambiguous: the promoted unknowns may be undisclosed colluders (good) or false alarms that censoring conveniently hides (bad), and nothing local can tell them apart. Blending resolves it — at .60/.40 censor-0 returns to parity (.7320 vs .7323) while keeping nearly all of the censored gain:

| candidate | c0 | c100 | c200 | folds up @c100 |
|---|---|---|---|---|
| v5/v6 risk | .7323 | .9475 | .9630 | — |
| context only | .7242 | .9525 | .9688 | 4/4 |
| **blend .60/.40** | **.7320** | **.9522** | **.9678** | **4/4** |

**Methodology note that mattered.** My first comparison used *score-dependent* censoring, where each model censors its own top unknowns — self-serving, and it flattered the context model. The table above uses a **fixed** censor set drawn from a neutral reference (the mean of both candidates), so no candidate can win by burying its own false positives. Astra had already flagged this distinction in `risk_blend_review.json`; always use a fixed reference when comparing candidates.

**Session scoreboard.** v1 .83232 → v2 .83492 → v3 .83472 → v4 .83942 → v5 .84409 → v6 .84436 → **v7 .84537**. Every submission that changed exactly one thing improved (v4, v5, v6, v7: +.0045, +.0047, +.0003, +.0010). Both submissions that changed several things at once did not (v2 +.0026 but uninterpretable, v3 −.0002).

**Remaining gap and where it lives.** Leader .94068. Implied split at our .84537: pair AP ≈ .923, evidence ≈ .520, behaviour ≈ .95. Against a plausible leader profile (.99/.70/.97) the gap is +.047 pair, +.036 evidence, +.002 behaviour.

**Next, in priority order.**
1. **Isolation evidence** — .27 against .58/.63, 26% of positives, so reaching ~.45 is worth ≈ +.009. Resisted everything across two sessions; it needs NEW features. Best lead, from diffing missed evidence against wrongly-promoted hands: the model over-promotes `policy_loose_max` and `cx_pre_raise_then_yield` (partner enters loose, then folds) while the missed evidence has higher `cx_pre_partner_call`, `cx_pre_call_air`, `cx_flop_act_partner`, `hu_actions_max` — the partner *calls with air and the pair continues together* rather than one yielding. Read more hands with `src/replay.py coordinated_isolation <n>` before building.
2. **Per-hand context for the risk model.** v7 used only mean/max/top5 aggregates. Richer aggregates (std, counts above threshold, within-pool percentiles of the context columns) are cheap — the per-hand context is already extracted for both phases.
3. Tune the .60/.40 context blend weight; .25–.75 all beat v6 at c100, so the optimum was not searched carefully.
