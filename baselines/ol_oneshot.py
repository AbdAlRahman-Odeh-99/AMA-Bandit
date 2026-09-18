"""One-shot Opportunistic Learning (OL) baseline.

The published OL method is a sequential DQN.  It couples a classifier
(``P-Net``) to a Q-network and rewards acquisition by the change in predictive
confidence per unit cost.  This module retains that architecture and reward,
but makes each action a complete modality subset chosen once from the initial
free-only state.  The resulting policy is a one-step Q learner, which is the
open-loop counterpart needed by this repository's one-shot comparison.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.budget_state import apply_global_budget_fallback
from core.utils import MaskLayerGrouped


class OLPQNetwork(nn.Module):
    """OL's shared P/Q architecture for grouped tabular modalities."""

    def __init__(
        self,
        mask_layer,
        num_classes,
        num_actions,
        *,
        hidden_sizes=(64, 32, 16),
        dropout=0.5,
        use_feature_mask=False,
    ):
        super().__init__()
        if not isinstance(mask_layer, MaskLayerGrouped):
            raise TypeError("OLPQNetwork requires MaskLayerGrouped")
        hidden_sizes = tuple(int(size) for size in hidden_sizes)
        if not hidden_sizes or min(hidden_sizes) < 1:
            raise ValueError("hidden_sizes must contain positive integers")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        if num_classes < 1 or num_actions < 1:
            raise ValueError("num_classes and num_actions must be positive")

        self.mask_layer = mask_layer
        self.num_features = int(mask_layer.group_matrix.shape[1])
        self.num_modalities = int(mask_layer.mask_size)
        self.num_classes = int(num_classes)
        self.num_actions = int(num_actions)
        self.hidden_sizes = hidden_sizes
        self.dropout = float(dropout)
        self.use_feature_mask = bool(use_feature_mask)
        input_size = self.num_features * (2 if use_feature_mask else 1)

        self.layers_p = nn.ModuleList()
        previous = input_size
        for size in (*hidden_sizes, self.num_classes):
            self.layers_p.append(nn.Linear(previous, size))
            previous = size

        q_hidden_sizes = []
        for index, p_size in enumerate(hidden_sizes):
            if index == 0:
                q_hidden_sizes.append(p_size)
            else:
                previous_p = hidden_sizes[index - 1]
                previous_q = q_hidden_sizes[-1]
                q_hidden_sizes.append(
                    max(1, previous_p * p_size // (previous_p + previous_q))
                )
        self.layers_q = nn.ModuleList()
        previous = input_size
        for p_size, q_size in zip(hidden_sizes, q_hidden_sizes):
            self.layers_q.append(nn.Linear(previous, q_size))
            previous = p_size + q_size
        self.layers_q.append(nn.Linear(previous, self.num_actions))

    def _split_masked_input(self, masked_input):
        expected = self.num_features + self.num_modalities
        if masked_input.ndim != 2 or masked_input.shape[1] != expected:
            raise ValueError(
                f"masked_input must have shape [batch, {expected}]"
            )
        return (
            masked_input[:, :self.num_features],
            masked_input[:, self.num_features:],
        )

    def _network_input(self, masked_features, modality_mask):
        if not self.use_feature_mask:
            return masked_features
        group_matrix = self.mask_layer.group_matrix.to(masked_features)
        feature_mask = modality_mask.to(masked_features) @ group_matrix
        return torch.cat([masked_features, feature_mask], dim=1)

    def _p_activations(self, network_input, *, force_dropout=False):
        activation = network_input
        activations = []
        for layer in self.layers_p[:-1]:
            activation = F.relu(layer(activation))
            activation = F.dropout(
                activation,
                p=self.dropout,
                training=self.training or force_dropout,
            )
            activations.append(activation)
        return activations

    def forward_pq(
        self,
        masked_features,
        modality_mask,
        *,
        force_dropout=False,
    ):
        network_input = self._network_input(masked_features, modality_mask)
        p_activations = self._p_activations(
            network_input, force_dropout=force_dropout
        )
        class_logits = self.layers_p[-1](p_activations[-1])

        q_activation = F.relu(self.layers_q[0](network_input))
        for layer, p_activation in zip(
            self.layers_q[1:-1], p_activations[:-1]
        ):
            q_activation = F.relu(
                layer(torch.cat([q_activation, p_activation.detach()], dim=1))
            )
        q_values = self.layers_q[-1](
            torch.cat([q_activation, p_activations[-1].detach()], dim=1)
        )
        return class_logits, q_values

    def p_logits(
        self,
        masked_features,
        modality_mask,
        *,
        force_dropout=False,
    ):
        """Run only P-Net when Q-values are not needed."""
        network_input = self._network_input(masked_features, modality_mask)
        activations = self._p_activations(
            network_input, force_dropout=force_dropout
        )
        return self.layers_p[-1](activations[-1])

    def q_only(self, masked_features, modality_mask):
        network_input = self._network_input(masked_features, modality_mask)
        with torch.no_grad():
            p_activations = self._p_activations(network_input)
        q_activation = F.relu(self.layers_q[0](network_input))
        for layer, p_activation in zip(
            self.layers_q[1:-1], p_activations[:-1]
        ):
            q_activation = F.relu(
                layer(torch.cat([q_activation, p_activation], dim=1))
            )
        return self.layers_q[-1](
            torch.cat([q_activation, p_activations[-1]], dim=1)
        )

    def forward(self, masked_input):
        masked_features, modality_mask = self._split_masked_input(masked_input)
        return self.p_logits(masked_features, modality_mask)

    def q_from_masked_input(self, masked_input):
        masked_features, modality_mask = self._split_masked_input(masked_input)
        return self.q_only(masked_features, modality_mask)

    def confidence(self, masked_input, mcdrop_samples=100):
        """Estimate class confidence using OL's Monte Carlo dropout."""
        if mcdrop_samples < 1:
            raise ValueError("mcdrop_samples must be positive")
        masked_features, modality_mask = self._split_masked_input(masked_input)
        repeated_features = masked_features.repeat_interleave(
            mcdrop_samples, dim=0
        )
        repeated_mask = modality_mask.repeat_interleave(
            mcdrop_samples, dim=0
        )
        logits = self.p_logits(
            repeated_features,
            repeated_mask,
            force_dropout=True,
        )
        probabilities = logits.softmax(dim=1).view(
            len(masked_input), mcdrop_samples, self.num_classes
        )
        return probabilities.mean(dim=1)

    def p_parameters(self):
        return self.layers_p.parameters()

    def q_parameters(self):
        return self.layers_q.parameters()


def confidence_change(confidence_before, confidence_after, method):
    """OL equation-7 confidence-change reward before cost normalization."""
    normalized = method.strip().lower()
    if normalized == "softmax":
        return (
            confidence_before.max(dim=1).values
            - confidence_after.max(dim=1).values
        ).abs()
    difference = confidence_before - confidence_after
    if normalized in {"bayesian-l1", "bayesian_l1", "l1"}:
        return difference.abs().sum(dim=1)
    if normalized in {"bayesian-l2", "bayesian_l2", "l2"}:
        return difference.square().sum(dim=1)
    raise ValueError(
        "reward_method must be softmax, Bayesian-L1, or Bayesian-L2"
    )


class OLOneShot(nn.Module):
    """One-step OL policy whose actions are complete modality subsets."""

    def __init__(
        self,
        mask_layer,
        feature_costs,
        num_classes,
        *,
        hidden_sizes=(64, 32, 16),
        dropout=0.5,
        use_feature_mask=False,
        reward_method="Bayesian-L1",
        mcdrop_samples=100,
        max_candidates=512,
        exact_max_paid=12,
        seed=42,
    ):
        super().__init__()
        costs = torch.as_tensor(feature_costs, dtype=torch.float32)
        if not isinstance(mask_layer, MaskLayerGrouped):
            raise TypeError("OLOneShot requires MaskLayerGrouped")
        if len(costs) != mask_layer.mask_size:
            raise ValueError("feature_costs must match the modality count")
        if float(costs[0]) != 0.0:
            raise ValueError("modality 0 must be free")
        if bool((costs[1:] <= 0).any()):
            raise ValueError("paid modality costs must be positive")
        if min(mcdrop_samples, max_candidates, exact_max_paid) < 1:
            raise ValueError("OL sample and candidate limits must be positive")

        self.mask_layer = mask_layer
        self.num_modalities = int(mask_layer.mask_size)
        self.reward_method = str(reward_method)
        self.mcdrop_samples = int(mcdrop_samples)
        self.seed = int(seed)
        self.register_buffer("feature_costs", costs)
        candidates = self._make_candidate_masks(
            int(max_candidates), int(exact_max_paid)
        )
        self.register_buffer("candidate_masks", candidates)
        self.register_buffer(
            "candidate_costs",
            (candidates.float() * costs.unsqueeze(0)).sum(dim=1),
        )
        self.pq_module = OLPQNetwork(
            mask_layer,
            num_classes,
            len(candidates),
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            use_feature_mask=use_feature_mask,
        )

    @property
    def predictor(self):
        return self.pq_module

    def _make_candidate_masks(self, max_candidates, exact_max_paid):
        n_paid = self.num_modalities - 1
        if n_paid <= exact_max_paid:
            subset_ids = torch.arange(1 << n_paid, dtype=torch.long)
            shifts = torch.arange(n_paid, dtype=torch.long)
            paid = ((subset_ids[:, None] >> shifts[None, :]) & 1).bool()
        else:
            rng = np.random.default_rng(self.seed)
            candidates = {tuple([False] * n_paid): None}
            for index in range(n_paid):
                singleton = [False] * n_paid
                singleton[index] = True
                candidates.setdefault(tuple(singleton), None)
            order = np.argsort(self.feature_costs[1:].cpu().numpy())
            prefix = np.zeros(n_paid, dtype=bool)
            for index in order:
                prefix[index] = True
                candidates.setdefault(tuple(prefix.tolist()), None)
            target = min(max_candidates, 1 << n_paid)
            while len(candidates) < target:
                size = int(rng.integers(0, n_paid + 1))
                subset = np.zeros(n_paid, dtype=bool)
                if size:
                    subset[rng.choice(n_paid, size=size, replace=False)] = True
                candidates.setdefault(tuple(subset.tolist()), None)
            paid = torch.tensor(list(candidates)[:target], dtype=torch.bool)
        free = torch.ones(len(paid), 1, dtype=torch.bool)
        return torch.cat([free, paid], dim=1)

    def free_mask(self, batch_size, *, dtype, device):
        mask = torch.zeros(
            batch_size, self.num_modalities, dtype=dtype, device=device
        )
        mask[:, 0] = 1.0
        return mask

    def q_values(self, x):
        free = self.free_mask(len(x), dtype=x.dtype, device=x.device)
        return self.pq_module.q_from_masked_input(self.mask_layer(x, free))

    def _feasible_candidates(self, budget):
        if budget is None:
            return torch.ones_like(self.candidate_costs, dtype=torch.bool)
        return self.candidate_costs <= float(budget) + 1e-7

    def choose_action_indices(
        self,
        q_values,
        *,
        budget,
        omd_lambda=0.0,
        epsilon=0.0,
    ):
        feasible = self._feasible_candidates(budget)
        if not bool(feasible.any()):
            return torch.zeros(
                len(q_values), dtype=torch.long, device=q_values.device
            )
        adjusted = q_values - float(omd_lambda) * self.candidate_costs
        adjusted = adjusted.masked_fill(~feasible.unsqueeze(0), -torch.inf)
        greedy = adjusted.argmax(dim=1)
        if epsilon <= 0:
            return greedy
        feasible_indices = torch.nonzero(feasible, as_tuple=False).flatten()
        positions = torch.randint(
            len(feasible_indices), (len(q_values),), device=q_values.device
        )
        random_actions = feasible_indices[positions]
        explore = torch.rand(len(q_values), device=q_values.device) < epsilon
        return torch.where(explore, random_actions, greedy)

    def apply_global_budget(self, proposed_masks, budget_state):
        masks = self.free_mask(
            len(proposed_masks),
            dtype=proposed_masks.dtype,
            device=proposed_masks.device,
        )
        actions = torch.zeros(
            len(proposed_masks), dtype=torch.long, device=proposed_masks.device
        )
        total_cost = 0.0
        for row, proposed in enumerate(proposed_masks):
            proposed_cost = float((proposed * self.feature_costs).sum().item())
            accepted_cost = proposed_cost
            used_fallback = False
            if budget_state is not None:
                accepted_cost, used_fallback = apply_global_budget_fallback(
                    proposed_cost, budget_state, fallback_cost=0.0
                )
            if not used_fallback:
                masks[row] = proposed
                matches = (self.candidate_masks == proposed.bool()).all(dim=1)
                actions[row] = torch.nonzero(matches, as_tuple=False)[0, 0]
            total_cost += float(accepted_cost)
        return masks, actions, total_cost

    def rewards_for_masks(self, x, masks):
        free = self.free_mask(len(x), dtype=x.dtype, device=x.device)
        with torch.no_grad():
            paired_inputs = torch.cat(
                [self.mask_layer(x, free), self.mask_layer(x, masks)], dim=0
            )
            paired_confidence = self.pq_module.confidence(
                paired_inputs, self.mcdrop_samples
            )
            confidence_before, confidence_after = paired_confidence.chunk(2)
            change = confidence_change(
                confidence_before, confidence_after, self.reward_method
            )
            costs = (masks * self.feature_costs).sum(dim=1)
            reward = torch.where(
                costs > 0,
                change / costs.clamp_min(torch.finfo(costs.dtype).eps),
                torch.zeros_like(change),
            )
        return reward

    def select_features(
        self,
        x,
        budget=None,
        verbose=False,
        global_budget_state=None,
    ):
        """Select the greedy online OL subset for each evaluation row."""
        x = x.to(self.feature_costs.device)
        self.eval()
        with torch.no_grad():
            q_values = self.q_values(x)
        masks = self.free_mask(len(x), dtype=x.dtype, device=x.device)
        total_cost = 0.0
        for row in range(len(x)):
            price = (
                0.0
                if global_budget_state is None
                else global_budget_state.omd_lambda
            )
            action = self.choose_action_indices(
                q_values[row:row + 1],
                budget=budget,
                omd_lambda=price,
            )[0]
            accepted_cost = float(self.candidate_costs[action].item())
            used_fallback = False
            if global_budget_state is not None:
                accepted_cost, used_fallback = apply_global_budget_fallback(
                    accepted_cost, global_budget_state, fallback_cost=0.0
                )
            if not used_fallback:
                masks[row] = self.candidate_masks[action].to(x.dtype)
            total_cost += float(accepted_cost)
            if verbose:
                selected = torch.nonzero(
                    masks[row], as_tuple=False
                ).flatten().tolist()
                print(
                    f"OL sample {row}: subset={selected}, "
                    f"cost={accepted_cost:.4f}"
                )
        return self.mask_layer(x, masks), masks, total_cost

    def forward(self, x, budget=None):
        masked, _, _ = self.select_features(x, budget=budget)
        return self.pq_module(masked)
