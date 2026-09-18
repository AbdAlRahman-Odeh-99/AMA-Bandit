"""
EDDI: one-shot (open-loop) feature/modality selection.
"""
import torch
import torch.nn as nn
from tqdm import tqdm

from core.utils import MaskLayerGrouped
from baselines.iterative import calculate_criterion, Imputer
from core.budget_state import apply_global_budget_fallback


class EDDI(nn.Module):
    '''
    One-shot EDDI, converted as a MINIMAL reduction of the original
    (sequential) EDDI's own per-feature scoring loop -- structurally
    parallel to how DIMEOneShot was derived from CMIEstimator.

    Original (sequential) EDDI, per step k of max_features:
      1. impute unobserved features at the CURRENT mask (re-sampled
         every step)
      2. for each remaining feature j: tentatively add it, predict,
         score via calculate_criterion -- ONE FEATURE AT A TIME
      3. argmax over those per-feature scores -> acquire ONE feature
      4. repeat from 1, now conditioned on the larger mask

    This class (one-shot, open-loop):
      1. impute unobserved features ONCE, at the initial (free-only) state
      2. score every paid feature INDIVIDUALLY, exactly like the
         original's own per-feature loop -- but only ONCE, not
         re-run after each acquisition
      3. rank all paid features by that single-pass score (optionally
         cost-normalized via the cost_normalized flag)
      4. commit to a subset via a single greedy walk down that fixed
         ranking, filling the budget (skip an unaffordable feature,
         keep checking cheaper ones further down)

    The only thing frozen/removed relative to the original sequential
    EDDI is the RE-IMPUTATION-AND-RE-SCORING loop between acquisitions --
    exactly analogous to what DIMEOneShot removes from CMIEstimator's
    per-step re-querying. The per-feature (not per-subset) scoring
    mechanism itself is preserved unchanged, matching the original's own
    algorithm shape.

    Args:
      sampler: model with an `.impute(x, mask)` method (e.g. PVAE).
      predictor: trained classifier/regressor over (masked_x, mask).
      mask_layer: MaskLayerGrouped instance.
      task: 'classification' or 'regression'.
      feature_costs: list of per-modality costs, or None. feature_costs[0]
        must be 0 (the free modality, always observed) if given.
      cost_normalized: if True (default), rank by score/cost (gain-per-cost).
        If False, rank by raw score under the hard budget filter (spends
        up to the budget instead of optimizing efficiency).
    '''
    def __init__(self, sampler, predictor, mask_layer, task='classification',
                 feature_costs=None, cost_normalized=True):
        super().__init__()
        assert hasattr(sampler, 'impute')
        self.sampler = sampler
        self.model = predictor
        self.mask_layer = mask_layer
        assert task in ('regression', 'classification')
        self.task = task
        if feature_costs is not None:
            assert feature_costs[0] == 0, "modality 0 must be the free modality (cost 0)"
        self.feature_costs = feature_costs
        self.cost_normalized = cost_normalized

        assert isinstance(mask_layer, MaskLayerGrouped)
        self.data_imputer = Imputer(self.mask_layer.group_matrix.cpu().data.numpy())

    def forward(self, x, budget=None, verbose=False, global_budget_state=None):
        '''Select a modality subset (one-shot) and make a prediction.'''
        x_masked, _, _ = self.select_features(x, budget, verbose, global_budget_state=global_budget_state)
        return self.model(x_masked)

    def select_features(self, x, budget=None, verbose=False, global_budget_state=None):
        '''
        One-shot selection: score every paid feature individually ONCE
        (mirroring the original EDDI's own per-feature loop), rank them,
        then commit to a budget-filling subset via a single greedy walk.

        Args:
          x: input batch (num_samples, d_in).
          budget: max total cost allowed per sample. None = unconstrained.
            IGNORED if global_budget_state is given.
          verbose:
          global_budget_state: optional budget_state.BudgetState with
            total_budget set. If given, the per-sample greedy budget-fill
            walk below is UNCHANGED (still ranks and fills as before),
            but the resulting subset's total cost is then checked ONCE
            against a SHARED, depleting pool, with forced fallback to the
            free modality on exhaustion -- same global-depletion
            semantics as DIMEOneShot's version; see
            apply_global_budget_fallback's docstring.
        '''
        model = self.model
        mask_layer = self.mask_layer
        sampler = self.sampler
        data_imputer = self.data_imputer
        device = next(model.parameters()).device

        assert isinstance(mask_layer, MaskLayerGrouped)
        num_modalities = mask_layer.mask_size
        candidate_modalities = list(range(1, num_modalities))  # exclude the free modality

        feature_costs = self.feature_costs
        m = torch.zeros((x.shape[0], num_modalities), device=device)
        m[:, 0] = 1  # free modality always observed

        total_cost = 0.0

        for i in tqdm(range(len(x)), disable=not verbose):
            x_row = x[i:i+1]
            m_row = m[i:i+1]

            # --- Single imputation at the initial (free-only) state ---
            x_sampled = sampler.impute(x_row, m_row)[0]
            mc_samples = x_sampled.shape[0]

            for g in range(num_modalities):
                if m_row[0, g] == 1:
                    inds = torch.where(mask_layer.group_matrix[g])[0].cpu().numpy()
                    original = x_row[:, inds]
                    x_sampled = data_imputer.impute(x_sampled, original, g)

            # --- Score every paid feature INDIVIDUALLY, ONCE -- mirrors
            # the original EDDI's own per-feature tentative-add-and-score
            # loop, just not repeated after each acquisition. ---
            m_expand = m_row.repeat(mc_samples, 1)
            scores = {}
            for j in candidate_modalities:
                m_expand[:, j] = 1
                x_expand_masked = mask_layer(x_sampled, m_expand)
                with torch.no_grad():
                    preds = model(x_expand_masked)
                criterion = calculate_criterion(preds, self.task)

                cost_j = feature_costs[j] if feature_costs is not None else 1.0
                if self.cost_normalized and cost_j > 0:
                    criterion = criterion / cost_j
                scores[j] = criterion

                m_expand[:, j] = 0  # revert before scoring the next feature

            # --- Rank once by that single-pass score ---
            order = sorted(candidate_modalities, key=lambda j: scores[j], reverse=True)

            if verbose:
                print(f"Sample {i}: per-feature scores = "
                      f"{[(j, round(float(scores[j]), 4)) for j in order[:5]]}...")

            # --- Greedy budget-fill walk down the fixed ranking (per-
            # sample cap only used when NOT under global-depletion mode)
            spent = 0.0
            selected = []
            for j in order:
                cost_j = feature_costs[j] if feature_costs is not None else 1.0
                if global_budget_state is None and budget is not None and spent + cost_j > budget:
                    continue  # skip, keep checking cheaper features further down
                selected.append(j)
                spent += cost_j

            if global_budget_state is not None:
                # Global depletion + forced fallback, checked ONCE
                # against the sample's fully-committed subset cost.
                accepted_cost, used_fallback = apply_global_budget_fallback(
                    spent, global_budget_state, fallback_cost=0.0)
                if used_fallback:
                    selected = []
                spent = accepted_cost

            for j in selected:
                m[i, j] = 1
            total_cost += spent

            if verbose:
                print(f"Selected subset {selected}, spent = {spent:.4f}")

        x_masked = mask_layer(x, m)
        if verbose:
            print(f"x_masked.shape = {x_masked.shape}, m.shape = {m.shape}, total_cost = {total_cost}")
        return x_masked, m, total_cost
