"""One-shot adaptation of Greedy Dynamic Feature Selection (GDFS).

Upstream GDFS repeatedly re-scores after each acquisition.  This open-loop
version evaluates the selector once from the free-only state and converts the
resulting cost-normalized scores into one budget-feasible joint subset.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from core.budget_state import BudgetState, apply_global_budget_fallback
from core.utils import MaskLayerGrouped


def budgeted_masks_from_scores(scores, feature_costs, budget):
    """Greedily pack each row by score-per-cost, always retaining view 0."""
    costs = torch.as_tensor(
        feature_costs, dtype=scores.dtype, device=scores.device
    )
    if scores.ndim != 2 or scores.shape[1] != len(costs):
        raise ValueError("scores must have one column per modality")
    limit = float(costs[1:].sum().item()) if budget is None else float(budget)
    masks = torch.zeros_like(scores)
    masks[:, 0] = 1.0
    ratios = scores[:, 1:] / costs[1:].clamp_min(1e-12)
    order = ratios.argsort(dim=1, descending=True)
    for row in range(len(scores)):
        remaining = max(0.0, limit)
        for paid_index in order[row].tolist():
            modality = paid_index + 1
            cost = float(costs[modality].item())
            if cost <= remaining + 1e-7:
                masks[row, modality] = 1.0
                remaining -= cost
    return masks


class GDFSOneShot(nn.Module):
    """Input-dependent, open-loop GDFS subset selector and predictor."""

    def __init__(self, selector, predictor, mask_layer, feature_costs):
        super().__init__()
        if not isinstance(mask_layer, MaskLayerGrouped):
            raise TypeError("GDFSOneShot requires MaskLayerGrouped")
        costs = torch.as_tensor(feature_costs, dtype=torch.float32)
        if len(costs) != mask_layer.mask_size:
            raise ValueError("feature_costs must match the modality count")
        if float(costs[0]) != 0.0:
            raise ValueError("modality 0 must be free")
        if bool((costs[1:] <= 0).any()):
            raise ValueError("paid modality costs must be positive")
        self.selector = selector
        self.predictor = predictor
        self.mask_layer = mask_layer
        self.num_modalities = int(mask_layer.mask_size)
        self.register_buffer("feature_costs", costs)

    def free_mask(self, batch_size, dtype, device):
        mask = torch.zeros(
            batch_size, self.num_modalities, dtype=dtype, device=device
        )
        mask[:, 0] = 1.0
        return mask

    def selection_scores(self, x):
        free = self.free_mask(len(x), x.dtype, x.device)
        return self.selector(self.mask_layer(x, free))

    def hard_masks(self, x, budget):
        return budgeted_masks_from_scores(
            self.selection_scores(x), self.feature_costs, budget
        )

    def select_features(
        self,
        x,
        budget=None,
        verbose=False,
        global_budget_state: BudgetState | None = None,
    ):
        device = self.feature_costs.device
        x = x.to(device)
        with torch.no_grad():
            proposed = self.hard_masks(x, budget)
        masks = self.free_mask(len(x), x.dtype, device)
        total_cost = 0.0
        for row in range(len(x)):
            proposed_cost = float(
                (proposed[row] * self.feature_costs).sum().item()
            )
            accepted_cost = proposed_cost
            used_fallback = False
            if global_budget_state is not None:
                accepted_cost, used_fallback = apply_global_budget_fallback(
                    proposed_cost, global_budget_state, fallback_cost=0.0
                )
            if not used_fallback:
                masks[row] = proposed[row]
            total_cost += float(accepted_cost)
            if verbose:
                chosen = torch.nonzero(masks[row]).flatten().tolist()
                print(
                    f"GDFS sample {row}: subset={chosen}, "
                    f"cost={accepted_cost:.4f}"
                )
        return self.mask_layer(x, masks), masks, total_cost

    def sample_features(
        self,
        x,
        budget,
        global_budget_state: BudgetState | None = None,
    ):
        """Sample hard one-shot subsets and retain differentiable log-probs."""
        device = self.feature_costs.device
        x = x.to(device)
        scores = self.selection_scores(x)
        masks = self.free_mask(len(x), x.dtype, device)
        log_probs = []
        entropies = []
        total_cost = 0.0
        limit = float(self.feature_costs[1:].sum().item()) if budget is None else float(budget)
        for row in range(len(x)):
            remaining = max(0.0, limit)
            available = torch.ones(
                self.num_modalities, dtype=torch.bool, device=device
            )
            available[0] = False
            row_log_prob = torch.zeros((), device=device)
            row_entropy = torch.zeros((), device=device)
            while True:
                affordable = available & (
                    self.feature_costs <= remaining + 1e-7
                )
                if not bool(affordable.any()):
                    break
                logits = scores[row] - torch.log(
                    self.feature_costs.clamp_min(1e-12)
                )
                logits = logits.masked_fill(~affordable, -torch.inf)
                distribution = Categorical(logits=logits)
                choice = distribution.sample()
                row_log_prob = row_log_prob + distribution.log_prob(choice)
                row_entropy = row_entropy + distribution.entropy()
                masks[row, choice] = 1.0
                available[choice] = False
                remaining -= float(self.feature_costs[choice].item())

            proposed_cost = float(
                (masks[row] * self.feature_costs).sum().item()
            )
            accepted_cost = proposed_cost
            used_fallback = False
            if global_budget_state is not None:
                accepted_cost, used_fallback = apply_global_budget_fallback(
                    proposed_cost, global_budget_state, fallback_cost=0.0
                )
            if used_fallback:
                masks[row].zero_()
                masks[row, 0] = 1.0
                row_log_prob = row_log_prob * 0.0
                row_entropy = row_entropy * 0.0
            total_cost += float(accepted_cost)
            log_probs.append(row_log_prob)
            entropies.append(row_entropy)
        return (
            self.mask_layer(x, masks),
            masks,
            total_cost,
            torch.stack(log_probs),
            torch.stack(entropies),
        )

    def forward(self, x, budget=None):
        x_masked, _, _ = self.select_features(x, budget=budget)
        return self.predictor(x_masked)
