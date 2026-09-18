import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from core.utils import MaskLayerGrouped, get_entropy
from core.budget_state import apply_global_budget_fallback
# Note: this file has NO functional dependency on cmi_estimator.py -- it
# works with ANY object exposing .mask_layer/.value_network/.predictor/
# .cmi_scaling (duck-typed, not isinstance-checked). An import of
# CMIEstimator used to sit here for documentation purposes only; removed
# since it was unused and just created an extra file dependency.


class DIMEOneShot(nn.Module):
    '''
    One-shot (open-loop) version of DIME's selection rule, built from a
    TRAINED CMIEstimator. Training is left completely untouched --
    CMIEstimator is used exactly as-is, fit via its usual
    pytorch_lightning Trainer. Nothing here subclasses or modifies it.

    How it relates to DIME's own adaptive inference (predict_step):

      DIME (adaptive, closed-loop):
        repeat until stopping criterion fires:
          1. query value_network on current x_masked -> per-feature CMI
          2. pick argmax(CMI / feature_costs) among unobserved features
          3. acquire it, update the mask, RE-QUERY the value network

      This class (one-shot, open-loop):
          1. query value_network ONCE, at the initial state (free
             modality only) -> per-feature CMI estimates
          2. rank paid modalities by CMI / feature_costs (DIME's exact
             per-step rule)
          3. commit to acquiring them greedily in that order, in a
             single decision, until the budget is exhausted or predicted
             CMI stops being positive (DIME's own positive-CMI check)

    Everything DIME-specific is preserved: the learned value network IS
    the selection mechanism, the same cmi_scaling transforms are applied
    ('bounded' -> sigmoid * entropy, 'positive' -> softplus, 'none' ->
    raw), the same cost normalization, the same positive-CMI acceptance
    check, and the same budget-mode stopping semantics as predict_step's
    mode='budget'. The ONLY change is that the CMI estimates are frozen
    at the initial mask instead of being re-estimated after each
    acquisition -- i.e. the feedback loop is removed, which is precisely
    the one-shot restriction.

    If no paid modality has positive predicted CMI, or none fits the
    budget, the sample falls back to the free modality (index 0) alone.

    Args:
      cmi_estimator: a TRAINED CMIEstimator instance. Its value_network,
        predictor, mask_layer, and cmi_scaling are all reused.
      feature_costs: list of per-modality costs; feature_costs[0] must be
        0 (the free modality, always observed).
    '''
    def __init__(self, cmi_estimator, feature_costs=None):
        super().__init__()
        mask_layer = cmi_estimator.mask_layer
        assert isinstance(mask_layer, MaskLayerGrouped)
        if feature_costs is not None:
            assert feature_costs[0] == 0, "modality 0 must be the free modality (cost 0)"

        self.value_network = cmi_estimator.value_network
        self.predictor = cmi_estimator.predictor
        self.mask_layer = mask_layer
        self.cmi_scaling = cmi_estimator.cmi_scaling
        self.num_modalities = mask_layer.mask_size

        if feature_costs is None:
            # Match CMIEstimator's default: uniform cost. Keep modality 0
            # free to preserve the free-modality convention.
            feature_costs = [0] + [1.0] * (self.num_modalities - 1)
        self.feature_costs = torch.tensor(feature_costs, dtype=torch.float32)

    def _estimate_cmi(self, x_masked):
        '''Predicted per-feature CMI, using the same cmi_scaling transform
        as CMIEstimator.training_step / predict_step.'''
        pred = self.predictor(x_masked)
        if self.cmi_scaling == 'bounded':
            entropy = get_entropy(pred).unsqueeze(1)
            pred_cmi = self.value_network(x_masked).sigmoid() * entropy
        elif self.cmi_scaling == 'positive':
            pred_cmi = torch.nn.functional.softplus(self.value_network(x_masked))
        else:
            pred_cmi = self.value_network(x_masked)
        return pred_cmi

    def select_features(self, x, budget=None, verbose=False, global_budget_state=None):
        '''
        One-shot selection: a single value-network query per sample at the
        initial state, then commit to the full subset.

        Args:
          x: input batch (num_samples, d_in).
          budget: max total cost allowed per sample (same semantics as
            predict_step's mode='budget'). None = only the positive-CMI
            check limits acquisition. IGNORED if global_budget_state is
            given -- see below.
          verbose:
          global_budget_state: optional budget_state.BudgetState with
            total_budget set. If given, the per-sample `budget` argument
            is NOT used to cap selection at all -- instead, each sample's
            already-chosen subset (ranked/committed exactly as before) is
            checked against a SHARED, depleting pool via
            apply_global_budget_fallback, forcing a fallback to the free
            modality once the pool is exhausted -- mirroring
            run_phase2_inference's global depletion + forced-fallback
            semantics (not skip-and-continue-to-cheaper-option; see that
            function's docstring). Samples are processed in the order
            given in `x`, so results are order-dependent under this mode,
            same as the GMM-bandit method's own Phase 2.
        '''
        device = next(self.predictor.parameters()).device
        feature_costs = self.feature_costs.to(device)
        num_modalities = self.num_modalities

        # Initial mask: free modality only.
        m = torch.zeros((x.shape[0], num_modalities), device=device)
        m[:, 0] = 1

        total_cost = 0.0
        with torch.no_grad():
            for i in tqdm(range(len(x)), disable=not verbose):
                x_row = x[i:i+1]
                m_row = m[i:i+1]

                # --- The single value-network query (this is the one-shot part) ---
                x_masked = self.mask_layer(x_row, m_row)
                pred_cmi = self._estimate_cmi(x_masked)[0]  # (num_modalities,)

                # Never select already-observed modalities (mirrors
                # predict_step's `pred_cmi -= 1e6 * mask`).
                pred_cmi = pred_cmi - 1e6 * m_row[0]

                # DIME's selection score: CMI / cost. Free modality has
                # cost 0 and is already observed, so exclude index 0.
                scores = pred_cmi / torch.clamp(feature_costs, min=1e-12)
                scores[0] = -np.inf

                # Rank paid modalities by score, then commit greedily in
                # that order -- open-loop version of DIME's per-step argmax.
                order = torch.argsort(scores, descending=True).tolist()

                spent = 0.0
                selected = []
                for j in order:
                    # Positive-CMI acceptance check (as in predict_step).
                    # Since costs are positive, sign(CMI/cost) == sign(CMI),
                    # so once we hit a non-positive CMI in the ranking,
                    # everything after it is non-positive too -> break.
                    if pred_cmi[j] <= 0:
                        break
                    cost_j = feature_costs[j].item()
                    if global_budget_state is None:
                        # Per-sample budget (original behavior).
                        if budget is not None and spent + cost_j > budget:
                            continue
                    selected.append(j)
                    spent += cost_j

                if global_budget_state is not None:
                    # Global depletion + forced fallback, checked ONCE
                    # against the sample's fully-committed subset cost
                    # (not per-feature) -- matches run_phase2_inference's
                    # whole-subset accept/fallback decision.
                    accepted_cost, used_fallback = apply_global_budget_fallback(
                        spent, global_budget_state, fallback_cost=0.0)
                    if used_fallback:
                        selected = []
                    spent = accepted_cost

                for j in selected:
                    m[i, j] = 1
                total_cost += spent

                if verbose:
                    print(f"Sample {i}: CMI/cost ranking = {order[:5]}..., "
                          f"selected = {selected}, spent = {spent:.4f}")

        x_masked = self.mask_layer(x, m)
        if verbose:
            print(f"x_masked.shape = {x_masked.shape}, m.shape = {m.shape}, total_cost = {total_cost}")
        return x_masked, m, total_cost
