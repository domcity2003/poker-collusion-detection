"""Ratio and contrast features the trees cannot build for themselves.

Reading the coordinated_isolation evidence hands showed the pair never contests
a pot against each other: one raises, the other instantly folds junk. The raw
counts for that already exist (`partner_fold_*` against `facing_partner_*`, and
`other_fold_*` against `facing_other_*`) but only as per-hand means. A gradient
boosted tree can threshold and add features, it cannot divide one by another, so
the *rate* of yielding - and crucially the yielding rate against the partner
minus the same player's yielding rate against everyone else - is invisible to it.

That difference is the whole signal: folding a lot is normal, folding a lot
specifically to one player is not.
"""
import polars as pl

def add_ratios(df):
 def col(c):return pl.col(c) if c in df.columns else pl.lit(0.0)
 e=1e-6
 ex=[
  # How often a player folds when facing the partner's aggression, versus when
  # facing anyone else's. The gap is partner-specific deference.
  (col('partner_fold_sum__mean')/(col('facing_partner_sum__mean')+e)).alias('r_yield_to_partner'),
  (col('other_fold_sum__mean')/(col('facing_other_sum__mean')+e)).alias('r_yield_to_field'),
  # Passive responses to the partner, same contrast.
  (col('partner_call_sum__mean')/(col('facing_partner_sum__mean')+e)).alias('r_call_partner'),
  (col('partner_raise_sum__mean')/(col('facing_partner_sum__mean')+e)).alias('r_reraise_partner'),
  # Do they ever actually fight each other after the flop?
  (col('post_together_aggression_sum__mean')/(col('together_actions_sum__mean')+e)).alias('r_post_contest'),
  (col('together_aggression_sum__mean')/(col('together_actions_sum__mean')+e)).alias('r_contest'),
  (col('hu_check_sum__mean')/(col('hu_actions_sum__mean')+e)).alias('r_hu_check'),
  (col('hu_aggression_sum__mean')/(col('hu_actions_sum__mean')+e)).alias('r_hu_aggression'),
  # Folding cheaply and folding the better hand, as rates rather than counts.
  (col('fold_cheap_strong_sum__mean')/(col('partner_fold_sum__mean')+e)).alias('r_fold_cheap'),
  (col('fold_better_sum__mean')/(col('partner_fold_sum__mean')+e)).alias('r_fold_better'),
  (col('hs_fold_with_best__mean')/(col('hs_acts_with_best__mean')+e)).alias('r_fold_with_best'),
  (col('hs_passive_with_best__mean')/(col('hs_acts_with_best__mean')+e)).alias('r_passive_with_best'),
  # Isolation shape: both in, third players pushed out.
  (col('third_folds__mean')/(col('third_actions__mean')+e)).alias('r_third_fold'),
  (col('isolation__mean')/(col('both_pre_raise__mean')+e)).alias('r_isolation_given_both_raise'),
  (col('hs_isolated_pot__mean')/(col('hs_pair_both_agg_pre__mean')+e)).alias('r_isolated_given_squeeze'),
 ]
 df=df.with_columns(ex)
 return df.with_columns([
  (pl.col('r_yield_to_partner')-pl.col('r_yield_to_field')).alias('r_yield_excess'),
  (pl.col('r_yield_to_partner')/(pl.col('r_yield_to_field')+e)).alias('r_yield_lift'),
  (pl.col('r_contest')-pl.col('r_hu_aggression')).alias('r_contest_gap'),
 ])
