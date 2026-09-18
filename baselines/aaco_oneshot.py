"""One-shot Acquisition Conditioned Oracle (AACO) baseline.

This is a compact adaptation of the AACO implementation in AFABench:
https://github.com/Linusaronsson/AFA-Benchmark/tree/main/afabench/components/methods/oracle/aaco

The published oracle optimizes a *joint* set of useful future acquisitions,
then its sequential wrapper chooses one member of that set and recomputes.
Here the joint set found at the initial free-only state is acquired directly,
which is the natural open-loop/one-shot reduction used by this comparison.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.budget_state import BudgetState, apply_global_budget_fallback
from core.utils import MaskLayerGrouped


class AACOOneShot(nn.Module):
    """Nonparametric, KNN-based one-shot AACO subset selector."""

    def __init__(
        self,
        predictor: nn.Module,
        mask_layer: MaskLayerGrouped,
        feature_costs,
        *,
        k_neighbors=5,
        acquisition_cost=0.05,
        max_candidates=100,
        exact_max_paid=12,
        seed=42,
    ):
        super().__init__()
        if not isinstance(mask_layer, MaskLayerGrouped):
            raise TypeError("AACOOneShot requires MaskLayerGrouped")
        costs = [float(c) for c in feature_costs]
        if len(costs) != mask_layer.mask_size:
            raise ValueError("feature_costs must match the modality count")
        if costs[0] != 0:
            raise ValueError("modality 0 must be the free modality (cost 0)")
        if any(c <= 0 for c in costs[1:]):
            raise ValueError("all paid modality costs must be positive")
        if k_neighbors < 1:
            raise ValueError("k_neighbors must be positive")
        if acquisition_cost < 0:
            raise ValueError("acquisition_cost must be nonnegative")
        if max_candidates < 1:
            raise ValueError("max_candidates must be positive")

        self.predictor = predictor
        self.mask_layer = mask_layer
        self.k_neighbors = int(k_neighbors)
        self.acquisition_cost = float(acquisition_cost)
        self.max_candidates = int(max_candidates)
        self.exact_max_paid = int(exact_max_paid)
        self.seed = int(seed)
        self.num_modalities = int(mask_layer.mask_size)
        self.register_buffer(
            "feature_costs", torch.tensor(costs, dtype=torch.float32)
        )
        self.register_buffer(
            "candidate_masks", self._make_candidate_masks()
        )
        self.register_buffer(
            "reference_x", torch.empty(0, self.num_modalities)
        )
        self.register_buffer(
            "reference_y", torch.empty(0, dtype=torch.long)
        )
        self.register_buffer(
            "reference_masks",
            torch.empty(0, self.num_modalities, dtype=torch.bool),
        )

    def _make_candidate_masks(self):
        """Create cached candidate masks, always including the free view."""
        n_paid = self.num_modalities - 1
        if n_paid <= self.exact_max_paid:
            subset_ids = torch.arange(1 << n_paid, dtype=torch.long)
            shifts = torch.arange(n_paid, dtype=torch.long)
            paid = ((subset_ids[:, None] >> shifts[None, :]) & 1).bool()
        else:
            rng = np.random.default_rng(self.seed)
            candidates = {tuple([False] * n_paid): None}
            # Singletons and cost-ordered prefixes make tight budgets robust;
            # the remainder follows AFABench's random-mask approximation.
            for j in range(n_paid):
                row = [False] * n_paid
                row[j] = True
                candidates.setdefault(tuple(row), None)
            order = np.argsort(self.feature_costs[1:].detach().cpu().numpy())
            row = np.zeros(n_paid, dtype=bool)
            for j in order:
                row[j] = True
                candidates.setdefault(tuple(row.tolist()), None)
            target = min(self.max_candidates, 1 << n_paid)
            while len(candidates) < target:
                size = int(rng.integers(0, n_paid + 1))
                row = np.zeros(n_paid, dtype=bool)
                if size:
                    row[rng.choice(n_paid, size=size, replace=False)] = True
                candidates.setdefault(tuple(row.tolist()), None)
            paid = torch.tensor(
                list(candidates)[:target], dtype=torch.bool
            )
        free = torch.ones(len(paid), 1, dtype=torch.bool)
        return torch.cat([free, paid], dim=1)

    @property
    def has_reference_data(self):
        return len(self.reference_y) > 0

    def add_reference(self, x, y, mask):
        """Append one or more causally acquired observations to the bank."""
        mask = mask.detach().bool().to(self.feature_costs.device)
        x = x.detach().to(self.feature_costs.device) * mask
        y = y.detach().long().to(self.feature_costs.device)
        self.reference_x = torch.cat([self.reference_x, x], dim=0)
        self.reference_y = torch.cat([self.reference_y, y], dim=0)
        self.reference_masks = torch.cat(
            [self.reference_masks, mask], dim=0
        )

    def _affordable_candidates(self, budget):
        masks = self.candidate_masks
        costs = (masks.float() * self.feature_costs.unsqueeze(0)).sum(dim=1)
        if budget is not None:
            keep = costs <= float(budget) + 1e-12
            masks = masks[keep]
            costs = costs[keep]
        if not len(masks):
            # The free-only candidate has cost zero, so this is defensive.
            masks = self.candidate_masks[:1]
            costs = torch.zeros(1, device=masks.device)
        return masks, costs

    def choose_mask(
        self,
        x_row,
        observed_mask=None,
        budget=None,
        exclude_reference_index=None,
    ):
        """Return AACO's minimum expected-loss joint subset for one row."""
        device = self.feature_costs.device
        if observed_mask is None:
            observed_mask = torch.zeros(
                self.num_modalities, dtype=torch.bool, device=device
            )
            observed_mask[0] = True
        else:
            observed_mask = observed_mask.to(device).bool().view(-1)
        free_only = torch.zeros_like(observed_mask)
        free_only[0] = True
        if not self.has_reference_data:
            return free_only

        candidate_masks, candidate_costs = self._affordable_candidates(budget)
        candidate_masks = candidate_masks | observed_mask.unsqueeze(0)
        candidate_masks = candidate_masks.unique(dim=0)
        candidate_costs = (
            candidate_masks.float() * self.feature_costs.unsqueeze(0)
        ).sum(dim=1)

        # A reference may support a candidate only if every required value was
        # actually acquired when that reference arrived. Reference masks may
        # therefore be partial in the online protocol.
        coverage = (
            self.reference_masks.unsqueeze(1)
            | ~candidate_masks.unsqueeze(0)
        ).all(dim=2)
        if exclude_reference_index is not None:
            reference_index = int(exclude_reference_index)
            if not 0 <= reference_index < len(self.reference_y):
                raise IndexError("exclude_reference_index is out of range")
            coverage[reference_index] = False
        eligible_counts = coverage.sum(dim=0)
        valid = eligible_counts > 0
        if not bool(valid.any()):
            return free_only
        candidate_masks = candidate_masks[valid]
        candidate_costs = candidate_costs[valid]
        coverage = coverage[:, valid]
        eligible_counts = eligible_counts[valid]

        x_row = x_row.to(device).view(1, -1)
        distances = (
            (self.reference_x - x_row).square()
            * observed_mask.float().unsqueeze(0)
        ).sum(dim=1)
        distance_table = distances.unsqueeze(1).expand(-1, len(candidate_masks))
        distance_table = distance_table.masked_fill(~coverage, torch.inf)
        k_eff = min(self.k_neighbors, int(eligible_counts.min().item()))
        neighbor_idx = torch.topk(
            distance_table, k_eff, dim=0, largest=False
        ).indices.T

        neighbor_x = self.reference_x[neighbor_idx]
        neighbor_y = self.reference_y[neighbor_idx]
        expanded_masks = candidate_masks[:, None, :].expand(
            -1, k_eff, -1
        )
        flat_x = (neighbor_x * expanded_masks).reshape(
            -1, self.num_modalities
        )
        flat_masks = expanded_masks.reshape(-1, self.num_modalities).float()
        self.predictor.eval()
        with torch.no_grad():
            logits = self.predictor(self.mask_layer(flat_x, flat_masks))
            losses = F.cross_entropy(
                logits, neighbor_y.reshape(-1), reduction="none"
            ).view(len(candidate_masks), k_eff)

        # Match AACO's inverse-frequency weighting, computed from only the
        # causal reference bank available at this decision.
        counts = torch.bincount(self.reference_y)
        weights = len(self.reference_y) / counts.clamp_min(1).float()
        losses = losses * weights[neighbor_y]
        expected_loss = losses.mean(dim=1)
        objective = expected_loss + self.acquisition_cost * candidate_costs
        return candidate_masks[int(torch.argmin(objective).item())]

    def select_features(
        self,
        x,
        budget=None,
        verbose=False,
        global_budget_state: BudgetState | None = None,
        exclude_reference_indices=None,
    ):
        """Choose one complete subset per row, then apply budget fallback."""
        device = self.feature_costs.device
        x = x.to(device)
        masks = torch.zeros(
            len(x), self.num_modalities, dtype=x.dtype, device=device
        )
        masks[:, 0] = 1.0
        if (exclude_reference_indices is not None
                and len(exclude_reference_indices) != len(x)):
            raise ValueError(
                "exclude_reference_indices must contain one index per row"
            )
        total_cost = 0.0
        for i in range(len(x)):
            exclude_index = (
                None if exclude_reference_indices is None
                else exclude_reference_indices[i]
            )
            chosen = self.choose_mask(
                x[i], budget=budget,
                exclude_reference_index=exclude_index,
            )
            proposed_cost = float(self.feature_costs[chosen].sum().item())
            accepted_cost = proposed_cost
            used_fallback = False
            if global_budget_state is not None:
                accepted_cost, used_fallback = apply_global_budget_fallback(
                    proposed_cost, global_budget_state, fallback_cost=0.0
                )
            if not used_fallback:
                masks[i] = chosen.to(masks.dtype)
            total_cost += float(accepted_cost)
            if verbose:
                selected = torch.nonzero(masks[i]).flatten().tolist()
                print(
                    f"AACO sample {i}: subset={selected}, "
                    f"cost={accepted_cost:.4f}"
                )
        return self.mask_layer(x, masks), masks, total_cost

    def forward(self, x, budget=None):
        x_masked, _, _ = self.select_features(x, budget=budget)
        return self.predictor(x_masked)
