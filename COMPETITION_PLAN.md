# Poker collusion competition: competitive execution plan

Objective: maximize the final competition score with reproducible models, strong validation, and inspectable evidence. This replaces the README's original practice-only ambition. No deadline-based shortcut is assumed.

## 1. Verified starting point

Inspected locally on 2026-09-19:

| Item | Observed |
| --- | --- |
| Hands | 2,000,000 |
| Actions | 18,609,028 |
| Seat records | 12,000,000 |
| Players | 12,000 |
| Development pairs | 1,860: 372 positive, 1,488 negative |
| Positive behaviors | 148 directed transfer, 132 soft play, 92 coordinated isolation |
| Other coordination labels | None |
| Evidence | 1,817 rows across 372 pairs; 3–5 hands per pair |
| Evaluation pairs | 112,540 |
| Pair overlap between labeled and evaluation pairs | None |
| Players in labeled / evaluation pairs | 3,127 / 11,302; 2,433 overlap |
| Development / evaluation hands | 1,200,000 / 800,000 |
| Hole cards | Both columns present, no nulls across all seats |

Development timestamps span January 1–22; evaluation spans January 10–February 2. The periods overlap: use `phase`, not a global timestamp cutoff.

The [official overview](https://www.kaggle.com/competitions/detect-suspicious-value-transfers-in-poker/overview), retrieved during planning, confirms 0.70 PairAP + 0.20 EvidenceMAP@5 + 0.10 BehaviorMAP. BehaviorMAP averages one-vs-rest AP across the three disclosed families, using risk_score for the predicted family and zero for other families; other_coordination is excluded from this component. Evidence must be evaluation-phase hands containing both players; duplicate evidence makes a submission invalid. Equal risk scores are resolved by pair_id. The public/private split is approximately 30%/70%, stratified by behavior.

Crucially, evaluation excludes pairs containing ANY publicly labeled positive player. The observed shared players therefore must not be interpreted as overlap with known colluders. There are 400 persistent pools of 30 players, one table per pool, with the first 60% of each pool's hands in development and the last 40% in evaluation. Coordination is episodic. Unlisted development pairs are unknown, not negative. The fourth behavior is intentionally undisclosed.

Still to retrieve and inspect: full rules, data dictionary, and the [reference metric notebook](https://www.kaggle.com/code/florianderoofr/slash-poker-competition-metric), including exact evidence AP denominators and missed-target logic.

## 2. Strategic thesis

Learn how a player's decisions change around a particular partner, conditional on cards, betting context, and ordinary playing style. Combine that signal with repeated directional losses, coordination against third parties, and hand-level evidence.

Three outputs deserve linked but distinct models:

1. Pair suspicion: primarily controls the score.
2. Behavior: identifies the mechanism of suspicious play.
3. Evidence ranking: identifies the specific hands that substantiate it.

Hole cards make card-aware analysis possible. Availability does not prove the generator follows realistic poker dynamics: verify card and action consistency before investing in expensive equity calculations. Synthetic-data signatures are hypotheses to test across independent splits, not a substitute for validation.

## 3. First deliverable: data and scoring contract

- Retrieve the official evaluation description, rules, data dictionary, permitted data usage, submission limits, and any scoring code or organizer clarifications.
- Enforce the confirmed requirement that submitted evidence comes exclusively from evaluation-phase hands.
- Resolve label scope: phase-wide collusion, episode-specific behavior, or another definition. Confirm whether supplied evidence lists are exhaustive or selected examples.
- Audit unique keys, foreign keys, duplicate records, both members' participation in every labeled evidence hand, action ordering, and pair-ID mapping.
- Check card validity and duplicates within hands, partial boards, pot/contribution/net-chip consistency, and side-pot behavior. Determine whether `amount` and `amount_to` represent increments or totals for each action type.
- Measure phase distributions, co-play exposures, stake changes, player coverage, table structure, missing/sentinel values, and metadata differences by label.
- Check how labeled pairs were selected. A 20% positive development sample does not establish a 20% evaluation prevalence.

Deliverables: `reports/data_audit.md`, a machine-readable schema, exact local metric implementation, and a strict submission validator. Test metric edge cases against official examples/code where available.

## 4. Validation before model selection

With only 372 positive pairs, split design is more consequential than a large hyperparameter search.

- Primary development evaluation: fixed pool-grouped folds, balanced for positive counts and behavior families. Hold out entire tables/pools, including all their supervised evidence. This prevents memorizing known colluders and holds out shared events together. Verify the pool/player mapping and class coverage first.
- Secondary player-disjoint evaluation within pools, if label coverage supports it: assign players to train/validation groups and exclude cross-boundary pairs; no player may occur on both sides. This distinguishes player generalization from pool generalization. Merely grouping on player_1 is insufficient. A random pair split is only an optimistic diagnostic, never the selection criterion.
- Temporal stress test: build features from earlier and later windows within development. Use it as a distribution/stability test until label timing supports a valid temporal target; do not automatically label every positive pair's earlier window positive.
- Reserve an untouched set of development pools as a lockbox, sized after checking class counts. Do not repeatedly select features against it. If strict separation leaves too few positives, explicitly document the compromise.
- Fit supervised player profiles, graph label propagation, hand rankers, scalers, calibration, and any target encoding within folds. A held-out label must not influence a neighboring training feature through a shared graph or hand.
- Audit shared-hand overlap between train and validation pairs. Prevent labeled hand/evidence targets from entering training when those same events are reserved for validation.
- Compare phase-specific feature distributions and test exposure-matched windows: 1.2M development hands versus 0.8M evaluation hands creates a count mismatch. Never inherit a positive pair label for a shortened window without supporting label semantics. Known evidence within a window can establish an observed positive, but a window without listed evidence is not automatically negative.

Report both mean fold scores and the score over pooled out-of-fold predictions. AP depends on prevalence, and pooled AP additionally depends on comparable score scales between folds. Inspect per-pool score shifts: strong within-pool ranking can still produce a poor global submission. Any fold/model calibration must be learned without its validation targets. Report sensitivity to negative weighting as a diagnostic rather than claiming the development AP predicts the private score.

Report PairAP, EvidenceMAP@5, BehaviorMAP, the weighted score, precision at top K, behavior-specific results, and performance by co-play exposure and familiar/unfamiliar player status. Estimate uncertainty with pair resampling and, where supported, player/component-aware resampling. Do not report tiny gains as decisive.

Use leaderboard results as external checks. Keep a submission log with its hypothesis, version, local results, and public result. Select final models primarily through validation robustness rather than repeated leaderboard tuning.

## 5. Efficient data architecture

Use a project-local environment with pinned dependencies. Start with columnar processing (Polars or DuckDB), Parquet caches, and CPU tree models. Profile before buying or requiring additional compute.

Processing layers:

1. Hand/player facts: phase, positions, stakes, stack sizes, actions, contributions, cards, outcomes.
2. Candidate pair/hand facts: both players' states and interactions, other players' responses, directed A-to-B and B-to-A views.
3. Pair/phase aggregates: exposure-adjusted rates, tails, top suspicious hands, temporal concentration, and graph context.

Encode string IDs internally; retain exact original identifiers for export. Join candidate pairs through hand participation and filter early. Avoid expanding all action rows into all possible player pairs. Partition intermediate work by phase and hand batches, and record peak memory and runtime. Cache expensive card features by card/state configuration.

## 6. Feature program

### A. Co-play and value movement

- Shared hands, tables, sessions, stake-normalized contributions and net chips.
- Directional loss/win coincidence, repeated beneficiary patterns, large-loss frequency, concentration, and persistence across sessions/windows. Players belong to one persistent pool/table; cross-table persistence is not an assumed available signal.
- Partner-conditioned outcomes versus the player's other opponents, matched for stake, position, stack, and available card strength.
- Shrink sparse estimates toward player/population baselines; carry exposure and uncertainty features.

A losing player's contribution and a partner's winnings do not uniquely identify a transfer in a multiway pot. Treat such quantities as transfer proxies, with heads-up and multiway cases distinguished.

### B. Card-aware decision anomalies

- Preflop strength and street-specific made-hand/draw features.
- Folding strong holdings to a partner, unusual weak calls or raises that benefit a partner, and expensive folds after prior investment.
- Checks and under-aggression against a partner relative to comparable nonpartner opportunities.
- Price-to-call, pot odds, effective stack, position, number of active players, and action history.
- Separate decision-time features, using only the board revealed by that street, from retrospective outcome/equity features using all supplied cards.

Begin with inexpensive strength features. Add exact showdown evaluation and selective Monte Carlo equity for high-value candidates if the audit and ablations justify it. Do not claim to calculate poker regret or true expected value without an appropriate opponent model.

### C. Sequential coordination

- Raise/reraise patterns involving both partners and subsequent third-party folds.
- Squeezes, isolation frequency, reciprocal pressure, and partner passivity after outsiders exit.
- Action-transition features by street, aggressor, target, and active-player count.
- Separate opportunities to coordinate from successful outcomes so common situations are not automatically suspicious.

### D. Partner-specific deviation

Fit a population action model on training-fold data, preferably with cross-fitting. Predict ordinary fold/call/raise behavior conditional on the observable state and player style. Aggregate residual anomalies when the candidate partner is involved. A recurring deviation benefiting the same partner is more informative than globally unusual play.

### E. Graph and episode features

- Exposure-normalized co-play graph and directed value-movement proxy graph.
- Reciprocity, common neighbors, small groups, directional imbalance, and concentration.
- Suspicious bursts versus diffuse activity; top-window and top-hand aggregates.
- Keep identity/metadata features in a separately ablated branch. Test whether apparent gains survive player-disjoint validation.

## 7. Modeling ladder and experiment order

1. Transparent heuristic pair score and evidence ranking; establish a complete working submission pipeline.
2. Regularized linear model plus a strong boosted-tree baseline on pair features.
3. Compare CatBoost and LightGBM using identical folds and modest searches, contingent on runtime compatibility.
4. Behavior-specific binary experts for directed transfer, soft play, and coordinated isolation, plus a general suspicion model.
5. Evidence-based pair aggregation: top hand scores, top-K means, counts above thresholds, and temporal concentration. Generate training features out of fold so the hand model cannot leak evidence labels into pair validation.
6. Partner-conditioned action residuals and graph features, each added through ablation.
7. Small, diverse ensemble selected from complementary out-of-fold errors. Compare rank averaging with probability blending; AP mostly rewards ordering, but cross-model scale and behavior scoring still matter.
8. Only consider sequence neural networks, graph neural networks, or heavier equity computation if simpler models plateau and diagnostics identify a problem they can solve.

For every experiment save configuration, feature version, seed, folds, all component scores, out-of-fold predictions, runtime, and error analysis. Promote feature blocks on repeatable gains and failure-mode improvements.

## 8. Evidence is a supervised retrieval problem

For each labeled pair, candidate hands must contain both players and satisfy the official phase requirements. Rank the known evidence hands using pair-hand features and behavior-specific signals.

- Confirm evidence semantics first. If the supplied 3–5 hands are only selected examples, other hands from positive pairs are unlabeled, not established negatives.
- Use confirmed-negative pairs for clean negatives, supplemented by carefully weighted within-positive-pair candidates and hard negatives.
- Train a lightweight classifier/ranker with the same pool-held-out folds as the pair model; ranking groups are individual pairs. Measure exact official MAP@5 and candidate recall separately.
- Preserve distinct hand IDs and return the five highest-ranked valid hands, or the permitted sentinel when appropriate. Never sacrifice ranked relevance for artificial diversity unless validation supports it.
- Test reciprocal interaction: whether hand scores improve pair ranking and pair behavior probabilities improve evidence ranking. Use out-of-fold stacking for both.

Build an inspectable hand replay showing cards, bets, outcomes, and why the hand was selected. Review top false positives, missed positives, and confusing behavior assignments.

## 9. Unknown behavior and score allocation

There are no development examples of `other_coordination`; the official overview confirms this is deliberate. A four-class supervised classifier cannot learn that class from these labels. Detecting these pairs and their evidence can improve 90% of the weighted objective, while the behavior component covers only the three known classes.

Start with the three observed behaviors and an independently estimated suspicion score. Add a parallel general-coordination anomaly branch using partner-conditioned deviations and observable action patterns, calibrated against confirmed hard negatives. Simulate unseen behavior by withholding one known family from training and evaluating its detection and evidence retrieval; rotate across all three families. This is a proxy for unknown-family robustness, not validation on the actual hidden family. Examine high-suspicion, low-known-class-fit examples. Do not assign `other_coordination` solely because confidence is low. Any unvalidated heuristic must remain explicitly labeled as such.

For this simulation, exclude the withheld family's labels AND evidence from all supervised training, feature selection, calibration, and blend tuning. Retain pool separation as well. Otherwise the supposedly unknown family can leak into the evidence model. Keep its pair/evidence targets solely for the final diagnostic on that run.

Prioritize pair ranking because its reported weight is 70%, but develop evidence early because evidence quality can also improve pair detection. Use the component weights to compare measured gains; a 0.05 evidence improvement contributes 0.01 overall under the reported formula.

## 10. Review and finalization

- Review representative directed transfer, soft play, coordinated isolation, high-risk unknowns, hard negatives, and low-exposure pairs.
- Diagnose false positives due to ordinary aggression, strong cards, stake changes, repeated table assignment, or low sample size.
- Run feature ablations, player-disjoint stress tests, window/exposure sensitivity, and the held-out lockbox evaluation before final selection.
- Refit selected configurations on eligible development data; construct evaluation features under the confirmed rules.
- Validate exactly 112,540 unique pair IDs, template order, finite risk scores in [0,1], allowed behavior strings, distinct valid evidence IDs, required phase and player participation, and sentinel formatting.
- Produce immutable submission versions, a manifest linking predictions to models/configuration/data checksums, and a reproducible run command.
- Keep code and five case reviews ready for the prize writeup requirement reported in the README, pending official verification.

## 11. Concrete milestones

| Milestone | Deliverable | Exit condition |
| --- | --- | --- |
| M0: Contract and audit | Official metric/rules notes, audit report, validation plan | Critical semantics resolved and data joins trustworthy |
| M1: End-to-end baseline | Feature cache, baseline model, local scores, valid CSV | Pipeline runs reproducibly and beats trivial references |
| M2: Behavioral features | Card-aware, sequential, partner-relative models | Ablations show repeatable gains |
| M3: Evidence model | Hand ranker, replays, pair aggregation | Candidate recall and MAP measured with leakage controls |
| M4: Robustness and ensemble | Complementary models, stress tests, error reviews | Improvements survive more than one split |
| M5: Final artifact | Validated submission, reproducible code, case reviews | Full submission contract passes |

## 12. Sharpened priorities and decision gates

### Read the planted examples before designing hundreds of features

Before broad modeling, replay 30 training-only evidence hands (10 per disclosed behavior), plus 30 matched hands from confirmed-negative pairs. Match negatives on opportunities such as street, stake, card strength, and active players where possible. Record the observable action pattern, beneficiary, plausible benign explanation, and which proposed feature would distinguish them. Select across multiple pools. Keep validation/lockbox cases out of this discovery set.

Deliverable: a compact behavior-to-feature specification with executable examples. If a feature does not distinguish its motivating examples, revise it before extracting it across 18.6M actions. Do not assume every planted pattern matches textbook poker intuition.

### Bring evidence forward

Build the first evidence heuristic and top-hand aggregates in M1, and the first supervised hand scorer alongside M2. M3 is their refinement, not their first appearance. Collusion is episodic: a few manipulated hands may disappear inside whole-period means. The primary representation should preserve the strongest hands, repeated motifs, direction, and contiguous bursts alongside ordinary aggregates.

Check that top-K and maximum features do not merely reward pairs with more shared hands. Compare them against exposure-matched negative distributions, include opportunity counts, and test stability when benign hands are added. Preserve the mapping from every aggregate back to its contributing hands.

### Distinguish a model failure from a pipeline failure

Measure evidence candidate recall before ranking. Run an oracle diagnostic that puts supplied evidence first whenever it survives candidate generation; compare its achievable metric with the real ranker. If evidence is missing from the candidates, repair joins/filters before tuning the model. A full shared-hand candidate set should retain every valid supplied evidence hand.

Separate behavior assignment from risk thresholding. The official behavior score uses risk_score for the single predicted family, so classification accuracy alone is not the selection objective. Compare known-family assignment, `none`/unknown policies, and evidence-output policies against the exact composite scorer. Do not suppress valid evidence just because pair risk is low unless the reference metric or validation demonstrates a reason.

### Make each expensive experiment earn its place

| Hypothesis | Smallest decisive experiment | Promotion / stop decision |
| --- | --- | --- |
| A few manipulated hands carry the signal | Compare means-only pair features with top-hand/burst features on fixed folds | Promote if held-out pair/evidence gains survive exposure controls |
| Partner-specific decisions beat generic bad-play detection | Add matched action residuals to the current baseline | Stop expanding the action model if gains vanish on unseen pools or hard negatives |
| Card knowledge improves discrimination | Add cheap strength features before any full equity engine | Implement expensive equity only for documented residual errors and measurable incremental gain |
| Novel coordination is detectable without known-family labels | Run three fully held-out-family experiments | Restrict unknown-detector blend weight if it mostly raises hard-negative scores |
| Graph features add information beyond fixed pools | Compare graph additions against pool, exposure, and player-style controls | Drop graph complexity if it reproduces table membership or exposure |
| An ensemble improves ranking | Blend saved out-of-fold predictions and inspect rescued/misranked pairs | Keep only components with repeatable complementary value |

Require a paired comparison against the current champion on the same folds, per-fold/component deltas, uncertainty, and runtime. A small uncertain gain may remain an ensemble candidate; it is not grounds for replacing a simpler champion. Freeze model selection before opening the lockbox. If lockbox results force redesign, record that the lockbox is spent rather than repeatedly calling it untouched.

Maintain one experiment ledger with hypothesis, expected score component, implementation cost, result, and decision. Run one interpretable feature-block change at a time before combinations. Review the largest errors after each promoted model, and use those errors to choose the next experiment. Stop tuning a branch after two successive additions fail to produce repeatable improvement unless a new failure diagnosis justifies returning to it.

### Prove invariances that should be true

- Swapping player_1/player_2 must preserve the undirected pair score and behavior; directed features must swap consistently. Use symmetric pooling of directional features or evaluate both orientations.
- Relabeling arbitrary player, hand, or table IDs must not change predictions in the identity-free model. Raw ID strings and hashes must not accidentally become numeric signals.
- Use within-training-pool, exposure-matched partner shuffles as a negative-control diagnostic: if partner-specific features retain the same apparent advantage, inspect whether they measure general player style or shared opportunity. These shuffles are diagnostics, not automatically valid negative training labels.
- After verifying rules, compare evaluation-phase behavioral features alone against those augmented with development-phase unlabeled history for the same eligible players. Keep the history branch explicit and mimic its information availability in validation; evaluation labels are never available.

Immediate next work: inspect the reference metric, replay the training-only case set, complete the audit, freeze pool folds, and build the first pair-hand table with evidence heuristics. The plan is complete; no predictive model has been trained yet.
