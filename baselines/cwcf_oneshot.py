"""One-shot Classification with Costly Features (CwCF).

The published method is a sequential dueling-DQN whose actions either acquire
one feature or terminate with a class prediction.  This adaptation compresses
the acquisition part to one action representing a complete modality subset;
the terminal class action is still taken only after the subset is observed.
Consequently subset selection uses only the initially free modality and never
looks at paid values before acquisition.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from core.budget_state import apply_global_budget_fallback
from core.utils import MaskLayerGrouped


class CwCFDuelingNetwork(nn.Module):
    """Original-style three-layer dueling Q-network for grouped features."""

    def __init__(self, mask_layer, num_classes, num_subsets, hidden_size=128,
                 hidden_layers=3):
        super().__init__()
        if not isinstance(mask_layer, MaskLayerGrouped):
            raise TypeError("CwCF requires MaskLayerGrouped")
        if min(num_classes, num_subsets, hidden_size, hidden_layers) < 1:
            raise ValueError("CwCF dimensions must be positive")
        self.mask_layer = mask_layer
        self.num_features = int(mask_layer.group_matrix.shape[1])
        self.num_modalities = int(mask_layer.mask_size)
        self.num_classes = int(num_classes)
        self.num_subsets = int(num_subsets)
        self.num_actions = self.num_classes + self.num_subsets

        layers = []
        previous = 2 * self.num_features
        for _ in range(int(hidden_layers)):
            layers.extend([nn.Linear(previous, int(hidden_size)), nn.ReLU()])
            previous = int(hidden_size)
        self.trunk = nn.Sequential(*layers)
        self.value_head = nn.Linear(previous, 1)
        self.advantage_head = nn.Linear(previous, self.num_actions)

    def _state(self, masked_input):
        expected = self.num_features + self.num_modalities
        if masked_input.ndim != 2 or masked_input.shape[1] != expected:
            raise ValueError(f"masked_input must have shape [batch, {expected}]")
        values = masked_input[:, :self.num_features]
        modality_mask = masked_input[:, self.num_features:]
        feature_mask = modality_mask @ self.mask_layer.group_matrix.to(values)
        return torch.cat([values, feature_mask], dim=1)

    def q_values(self, masked_input):
        hidden = self.trunk(self._state(masked_input))
        value = self.value_head(hidden)
        advantage = self.advantage_head(hidden)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def forward(self, masked_input):
        """Terminal classification-action values, used as class logits."""
        return self.q_values(masked_input)[:, :self.num_classes]


class CwCFOneShot(nn.Module):
    """CwCF with one complete-subset acquisition action per example."""

    def __init__(self, mask_layer, feature_costs, num_classes, *,
                 hidden_size=128, hidden_layers=3, cost_weight=1.0,
                 max_candidates=512, exact_max_paid=12, seed=42):
        super().__init__()
        costs = torch.as_tensor(feature_costs, dtype=torch.float32)
        if not isinstance(mask_layer, MaskLayerGrouped):
            raise TypeError("CwCF requires MaskLayerGrouped")
        if len(costs) != mask_layer.mask_size:
            raise ValueError("feature_costs must match the modality count")
        if float(costs[0]) != 0.0 or bool((costs[1:] <= 0).any()):
            raise ValueError("modality 0 must be free and paid costs positive")
        if cost_weight < 0 or min(max_candidates, exact_max_paid) < 1:
            raise ValueError("invalid CwCF cost/candidate settings")
        self.mask_layer = mask_layer
        self.num_modalities = int(mask_layer.mask_size)
        self.num_classes = int(num_classes)
        self.cost_weight = float(cost_weight)
        self.seed = int(seed)
        self.register_buffer("feature_costs", costs)
        candidates = self._make_candidate_masks(max_candidates, exact_max_paid)
        self.register_buffer("candidate_masks", candidates)
        self.register_buffer(
            "candidate_costs", (candidates.float() * costs).sum(dim=1)
        )
        self.q_network = CwCFDuelingNetwork(
            mask_layer, num_classes, len(candidates), hidden_size, hidden_layers
        )

    @property
    def predictor(self):
        return self.q_network

    def _make_candidate_masks(self, max_candidates, exact_max_paid):
        n_paid = self.num_modalities - 1
        if n_paid <= int(exact_max_paid):
            ids = torch.arange(1 << n_paid, dtype=torch.long)
            shifts = torch.arange(n_paid, dtype=torch.long)
            paid = ((ids[:, None] >> shifts[None, :]) & 1).bool()
        else:
            rng = np.random.default_rng(self.seed)
            items = {tuple([False] * n_paid): None}
            for index in range(n_paid):
                row = [False] * n_paid
                row[index] = True
                items.setdefault(tuple(row), None)
            order = np.argsort(self.feature_costs[1:].cpu().numpy())
            row = np.zeros(n_paid, dtype=bool)
            for index in order:
                row[index] = True
                items.setdefault(tuple(row.tolist()), None)
            target = min(int(max_candidates), 1 << n_paid)
            while len(items) < target:
                size = int(rng.integers(n_paid + 1))
                row = np.zeros(n_paid, dtype=bool)
                if size:
                    row[rng.choice(n_paid, size=size, replace=False)] = True
                items.setdefault(tuple(row.tolist()), None)
            paid = torch.tensor(list(items)[:target], dtype=torch.bool)
        return torch.cat([torch.ones(len(paid), 1, dtype=torch.bool), paid], 1)

    def free_mask(self, batch_size, *, dtype, device):
        mask = torch.zeros(batch_size, self.num_modalities,
                           dtype=dtype, device=device)
        mask[:, 0] = 1
        return mask

    def subset_q_values(self, x):
        free = self.free_mask(len(x), dtype=x.dtype, device=x.device)
        q = self.q_network.q_values(self.mask_layer(x, free))
        return q[:, self.num_classes:]

    def feasible_candidates(self, budget):
        if budget is None:
            return torch.ones_like(self.candidate_costs, dtype=torch.bool)
        return self.candidate_costs <= float(budget) + 1e-7

    def choose_action_indices(self, subset_q, *, budget, omd_lambda=0.0,
                              epsilon=0.0):
        feasible = self.feasible_candidates(budget)
        scores = subset_q - float(omd_lambda) * self.candidate_costs
        scores = scores.masked_fill(~feasible.unsqueeze(0), -torch.inf)
        greedy = scores.argmax(1)
        if epsilon <= 0:
            return greedy
        indices = torch.nonzero(feasible, as_tuple=False).flatten()
        random = indices[torch.randint(len(indices), (len(subset_q),),
                                       device=subset_q.device)]
        explore = torch.rand(len(subset_q), device=subset_q.device) < epsilon
        return torch.where(explore, random, greedy)

    def apply_global_budget(self, proposed_masks, budget_state):
        masks = self.free_mask(len(proposed_masks), dtype=proposed_masks.dtype,
                               device=proposed_masks.device)
        actions = torch.zeros(len(proposed_masks), dtype=torch.long,
                              device=proposed_masks.device)
        total_cost = 0.0
        for row, proposed in enumerate(proposed_masks):
            matches = (self.candidate_masks == proposed.bool()).all(1)
            action = torch.nonzero(matches, as_tuple=False)[0, 0]
            cost = float(self.candidate_costs[action])
            fallback = False
            if budget_state is not None:
                cost, fallback = apply_global_budget_fallback(
                    cost, budget_state, fallback_cost=0.0
                )
            if not fallback:
                masks[row] = proposed
                actions[row] = action
            total_cost += float(cost)
        return masks, actions, total_cost

    def select_features(self, x, budget=None, verbose=False,
                        global_budget_state=None):
        x = x.to(self.feature_costs.device)
        self.eval()
        with torch.no_grad():
            subset_q = self.subset_q_values(x)
        masks = self.free_mask(len(x), dtype=x.dtype, device=x.device)
        total_cost = 0.0
        for row in range(len(x)):
            price = 0.0 if global_budget_state is None else global_budget_state.omd_lambda
            action = self.choose_action_indices(
                subset_q[row:row + 1], budget=budget, omd_lambda=price
            )[0]
            cost = float(self.candidate_costs[action])
            fallback = False
            if global_budget_state is not None:
                cost, fallback = apply_global_budget_fallback(
                    cost, global_budget_state, fallback_cost=0.0
                )
            if not fallback:
                masks[row] = self.candidate_masks[action].to(x.dtype)
            total_cost += float(cost)
            if verbose:
                chosen = torch.nonzero(masks[row], as_tuple=False).flatten().tolist()
                print(f"CwCF sample {row}: subset={chosen}, cost={cost:.4f}")
        return self.mask_layer(x, masks), masks, total_cost

    @staticmethod
    def classification_targets(y, num_classes, *, dtype):
        targets = -torch.ones(len(y), num_classes, dtype=dtype, device=y.device)
        return targets.scatter(1, y[:, None], 0.0)
