"""
Shared budget-enforcement mechanism used by the one-shot baselines. It keeps
training and inference under the same global-depletion/free-view-fallback
convention as the proposed methods without importing their acquisition code.

Two independent things are being ported here, deliberately kept separate:

  1. TRAINING: dual-ascent (a first-order / Euclidean-mirror-map special
     case of "OMD") update of a single scalar price `omd_lambda`, exactly
     using the shared update rule:

         raw_lambda = omd_lambda + step_size * (instant_cost - spending_ratio)
         omd_lambda = clip(raw_lambda, 0, lambda_max)

     `omd_lambda` prices cost into whatever score each baseline already
     uses to decide what to acquire (DIME's pred_cmi, EDDI's per-feature
     criterion, CAE's selection logits) -- see `penalize_scores` below.
     This is a LAGRANGIAN RELAXATION of the budget constraint: lambda
     reacts to realized spend, but there is no hard per-round cutoff
     enforced through lambda alone.

  2. GLOBAL DEPLETION + FALLBACK: a literal remaining_budget counter that
     depletes as cost is realized (training OR inference), with a forced
     fallback to the free-only feature once it hits zero -- matching both
     the proposed methods. This is what actually turns a PER-SAMPLE budget
     into a GLOBAL one: the pool is shared and stateful across an ordered
     stream of samples/batches, not reset for each new sample.

Calibration (spending_ratio, lambda_max, step_size) is FIXED for now, per
your instruction -- no auto-tuning. Pass different BudgetState instances
(with dataset-appropriate spending_ratio) per dataset/method rather than
sharing one instance's constants across differently-scaled cost models.
"""

from __future__ import annotations


class BudgetState:
    """
    Shared training-time (Lagrangian) and inference-time (depleting-pool)
    budget-enforcement state.

    Args:
      spending_ratio: target mean cost per round/sample (the analogue of
        run_phase1_training's `spending_ratio = training_budget / n_train`
        or Phase 2's `inference_budget / n_inference`). Used ONLY to drive
        the OMD dual-ascent update -- it does not by itself cap anything.
      total_budget: if given, initializes `remaining_budget` to this value
        and enables global depletion + fallback (item 2 above). If None,
        only the OMD lambda-pricing mechanism (item 1) is active and
        nothing is ever hard-capped -- useful if you want the Lagrangian
        relaxation without also imposing a literal depleting pool (e.g.
        while still deciding on (b) vs (a) train-time semantics).
      lambda_max: clip ceiling for omd_lambda (matches run_phase1_training's
        `lambda_max = 10`).
      step_size: dual-ascent step size (matches `step_size = 1.0`).
    """

    def __init__(self, spending_ratio, total_budget=None, lambda_max=10.0, step_size=1.0):
        self.spending_ratio = float(spending_ratio)
        self.lambda_max = float(lambda_max)
        self.step_size = float(step_size)
        self.omd_lambda = 0.0

        self.total_budget = None if total_budget is None else float(total_budget)
        self.remaining_budget = self.total_budget

        # Cumulative spend across the entire lifetime of this state.
        self.cumulative_spent = 0.0

    # ------------------------------------------------------------------
    # Item 1: OMD / dual-ascent update on the price lambda
    # ------------------------------------------------------------------
    def update_lambda(self, realized_cost):
        """
        One dual-ascent step, given the realized mean cost of the round/
        batch just completed. Identical update rule to
        run_phase1_training's per-round lambda update, just called once
        per training round (which may be a batch, not a single sample --
        see per-file docstrings for what counts as a "round" there).
        """
        raw_lambda = self.omd_lambda + self.step_size * (float(realized_cost) - self.spending_ratio)
        self.omd_lambda = max(0.0, min(self.lambda_max, raw_lambda))

    def penalize_scores(self, scores, costs):
        """
        Subtract the current lambda-price from a batch of per-feature (or
        per-subset) scores, e.g. DIME's pred_cmi or EDDI's per-feature
        criterion, BEFORE ranking/argmax. Shapes broadcast the same way
        `scores / feature_costs` already does elsewhere in this codebase
        -- this is a straight substitution of that division for a
        subtraction-by-lambda-times-cost, i.e. the Lagrangian penalty
        instead of a hard per-sample cost-normalized ranking.
        """
        return scores - self.omd_lambda * costs

    # ------------------------------------------------------------------
    # Item 2: global depletion + forced fallback (training or inference)
    # ------------------------------------------------------------------
    @property
    def is_exhausted(self):
        """True once the global pool has hit zero. Always False if this
        BudgetState was constructed without a total_budget (pure
        Lagrangian mode, item 1 only)."""
        return self.remaining_budget is not None and self.remaining_budget <= 0

    def spend(self, realized_cost):
        """Deplete the global pool by a realized cost (scalar, e.g. a
        batch's mean cost during training, or a single sample's realized
        cost during inference). No-op on remaining_budget if total_budget
        was never set, but cumulative_spent is always tracked regardless."""
        realized_cost = float(realized_cost)
        self.cumulative_spent += realized_cost
        if self.remaining_budget is not None:
            self.remaining_budget = max(0.0, self.remaining_budget - realized_cost)

    def summary(self):
        """
        Snapshot of this state's current lambda/spend, for logging or
        saving to results -- e.g. into a CSV row, so you can check
        post-hoc whether lambda actually moved/settled and how much was
        actually spent, rather than assuming the mechanism behaved as
        intended.

        cumulative_spent is the total realized cost across this object's
        lifetime. spent_total is the amount depleted from its global pool.
        spent_total/total_budget/remaining_budget are None if this
        BudgetState was never given a total_budget (pure Lagrangian mode).
        """
        spent_total = None
        if self.total_budget is not None:
            spent_total = self.total_budget - self.remaining_budget
        return {
            "lambda_final": self.omd_lambda,
            "cumulative_spent": self.cumulative_spent,
            "total_budget": self.total_budget,
            "remaining_budget": self.remaining_budget,
            "spent_total": spent_total,
        }


# Inference-side global depletion + fallback wrapper.
#
# Each baseline keeps its OWN selection mechanism (DIME's ranking, EDDI's
# subset scoring, CAE's learned mask) -- this wrapper only decides whether
# a sample's ALREADY-CHOSEN subset is affordable against the SHARED,
# depleting pool, exactly mirroring run_phase2_inference's:
#
#     if remaining_budget - cost < 0:
#         subset = fallback_subset
#         cost = fallback_cost
#     remaining_budget -= cost
#
# -- i.e. forced fallback to the free-only subset, NOT skip-and-try-a-
# cheaper-option (that distinction was flagged earlier as a deliberate
# design choice in the GMM script; this wrapper matches it exactly so
# baseline inference-time accounting is the same mechanism).
# ----------------------------------------------------------------------
def apply_global_budget_fallback(cost, state: BudgetState, fallback_cost=0.0):
    """
    Args:
      cost: realized cost of the subset a baseline ALREADY selected for
        one sample.
      state: BudgetState with total_budget set (i.e. remaining_budget is
        being tracked). If state.total_budget is None, this is a no-op
        that always accepts the given cost (no global pool to check).
      fallback_cost: cost of the free-only fallback (0.0 in every
        convention used across these files).

    Returns: (accepted_cost, used_fallback) -- accepted_cost is what
    actually gets charged to the pool (either `cost` or `fallback_cost`);
    used_fallback tells the caller whether to swap its chosen mask/subset
    for the free-only one before returning predictions.
    """
    if state.remaining_budget is None:
        return cost, False

    if state.remaining_budget - cost < 0:
        # BUG FIX: this used to mutate state.remaining_budget directly,
        # which skipped state.cumulative_spent entirely -- meaning
        # inference-side "actual spending" silently read 0 regardless of
        # what was really spent. Routing through state.spend() keeps
        # both counters consistent, the same as every training-side call.
        state.spend(fallback_cost)
        return fallback_cost, True

    state.spend(cost)
    return cost, False
