"""Cost-aware one-shot Permutation Training (PT) component.

The online adapter updates global permutation-style importance estimates from
past acquired observations. Those estimates define one ranking shared by every
sample; view 0 stays free and paid modalities are packed into the cost budget.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.gdfs_oneshot import budgeted_masks_from_scores
from core.budget_state import BudgetState, apply_global_budget_fallback
from core.utils import MaskLayerGrouped


class PTOneShot(nn.Module):
    """Static cost-aware subset selector driven by permutation importance."""

    def __init__(
        self,
        predictor,
        mask_layer,
        feature_costs,
        feature_importance=None,
        *,
        importance_decay=0.9,
        seed=42,
    ):
        super().__init__()
        if not isinstance(mask_layer, MaskLayerGrouped):
            raise TypeError("PTOneShot requires MaskLayerGrouped")
        costs = torch.as_tensor(feature_costs, dtype=torch.float32)
        if len(costs) != mask_layer.mask_size:
            raise ValueError("feature_costs must match the modality count")
        if float(costs[0]) != 0.0:
            raise ValueError("modality 0 must be free")
        if bool((costs[1:] <= 0).any()):
            raise ValueError("paid modality costs must be positive")
        if not 0 <= importance_decay < 1:
            raise ValueError("importance_decay must lie in [0, 1)")
        self.predictor = predictor
        self.mask_layer = mask_layer
        self.num_modalities = int(mask_layer.mask_size)
        self.importance_decay = float(importance_decay)
        self.rng = np.random.default_rng(seed)
        self.register_buffer("feature_costs", costs)
        if feature_importance is None:
            importance = torch.zeros(self.num_modalities)
        else:
            importance = torch.as_tensor(
                feature_importance, dtype=torch.float32
            )
        if len(importance) != self.num_modalities:
            raise ValueError("feature_importance has the wrong length")
        importance[0] = torch.inf
        self.register_buffer("feature_importance", importance)
        self.register_buffer(
            "importance_updates", torch.zeros(self.num_modalities)
        )
        self.reservoir_values = [[] for _ in range(self.num_modalities)]

    def masks_for_budget(self, batch_size, budget, dtype, device):
        scores = self.feature_importance.to(device=device, dtype=dtype)
        scores = scores.unsqueeze(0).expand(batch_size, -1)
        return budgeted_masks_from_scores(scores, self.feature_costs, budget)

    def select_features(
        self,
        x,
        budget=None,
        verbose=False,
        global_budget_state: BudgetState | None = None,
    ):
        device = self.feature_costs.device
        x = x.to(device)
        proposed = self.masks_for_budget(len(x), budget, x.dtype, device)
        masks = torch.zeros_like(proposed)
        masks[:, 0] = 1.0
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
                    f"PT sample {row}: subset={chosen}, "
                    f"cost={accepted_cost:.4f}"
                )
        return self.mask_layer(x, masks), masks, total_cost

    def update_causal_importance(self, x, y, mask):
        """Update streaming permutation scores using only past acquired values."""
        device = self.feature_costs.device
        x = x.detach().to(device)
        y = y.detach().long().to(device)
        mask = mask.detach().to(device)
        self.predictor.eval()
        with torch.no_grad():
            baseline_loss = F.cross_entropy(
                self.predictor(self.mask_layer(x, mask)), y
            )
            for modality in range(1, self.num_modalities):
                if not bool(mask[0, modality]) or not self.reservoir_values[modality]:
                    continue
                replacement = self.rng.choice(
                    self.reservoir_values[modality]
                )
                permuted = x.clone()
                permuted[:, modality] = float(replacement)
                permuted_loss = F.cross_entropy(
                    self.predictor(self.mask_layer(permuted, mask)), y
                )
                delta = permuted_loss - baseline_loss
                old = self.feature_importance[modality]
                if self.importance_updates[modality] == 0:
                    updated = delta
                else:
                    updated = (
                        self.importance_decay * old
                        + (1 - self.importance_decay) * delta
                    )
                self.feature_importance[modality] = updated
                self.importance_updates[modality] += 1

        for modality in range(self.num_modalities):
            if bool(mask[0, modality]):
                self.reservoir_values[modality].append(
                    float(x[0, modality].item())
                )

    def forward(self, x, budget=None):
        x_masked, _, _ = self.select_features(x, budget=budget)
        return self.predictor(x_masked)
