"""One-shot Joint Active Feature Acquisition and Classification (JAFA).

The original JAFA policy uses n-step Q-learning to acquire one feature at a
time or stop.  The paper formally defines actions as subsets, however, so the
open-loop reduction used here makes one decision from the initial free-only
state: choose a complete affordable subset, acquire it, and predict.

This implementation retains JAFA's defining model structure:

* an order-invariant read-process set encoder for observed feature groups;
* a classifier trained on partially observed states;
* a Q-network whose gradient does not update the shared encoder; and
* reward equal to negative classification loss minus acquisition cost.

With one acquisition decision the Bellman target is the immediate terminal
reward, making this a contextual-bandit/single-step-Q version of JAFA rather
than the published sequential policy.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from core.budget_state import apply_global_budget_fallback
from core.utils import MaskLayerGrouped


def _mlp(input_size, output_size, hidden_size, *, depth=2):
    layers = []
    current = input_size
    for _ in range(depth):
        layers.extend([nn.Linear(current, hidden_size), nn.ReLU()])
        current = hidden_size
    layers.append(nn.Linear(current, output_size))
    return nn.Sequential(*layers)


class ReadProcessSetEncoder(nn.Module):
    """JAFA's attention-based, order-invariant read-process set encoder."""

    def __init__(
        self,
        element_size,
        output_size=16,
        hidden_size=32,
        memory_size=16,
        processing_steps=5,
    ):
        super().__init__()
        if min(output_size, hidden_size, memory_size, processing_steps) < 1:
            raise ValueError("all encoder dimensions and steps must be positive")
        self.output_size = int(output_size)
        self.memory_size = int(memory_size)
        self.processing_steps = int(processing_steps)
        self.reader = _mlp(element_size, memory_size, hidden_size)
        self.process = nn.LSTMCell(memory_size, memory_size)
        self.writer = _mlp(2 * memory_size, output_size, hidden_size)
        self.empty_set = nn.Parameter(torch.zeros(output_size))

    def forward(self, elements, observed_mask):
        if elements.ndim != 3:
            raise ValueError("elements must have shape [batch, set, element]")
        if observed_mask.shape != elements.shape[:2]:
            raise ValueError("observed_mask must match the first two dimensions")
        observed_mask = observed_mask.bool()
        batch_size = len(elements)
        nonempty = observed_mask.any(dim=1)
        safe_mask = observed_mask.clone()
        safe_mask[~nonempty, 0] = True

        memories = self.reader(elements)
        query = torch.zeros(
            batch_size, self.memory_size,
            dtype=elements.dtype, device=elements.device,
        )
        cell = torch.zeros_like(query)
        read = torch.zeros_like(query)
        for _ in range(self.processing_steps):
            query, cell = self.process(read, (query, cell))
            attention_logits = torch.bmm(
                memories, query.unsqueeze(-1)
            ).squeeze(-1)
            attention_logits = attention_logits.masked_fill(
                ~safe_mask, -torch.inf
            )
            attention = torch.softmax(attention_logits, dim=1)
            read = torch.bmm(attention.unsqueeze(1), memories).squeeze(1)

        output = self.writer(torch.cat([query, read], dim=1))
        if bool((~nonempty).any()):
            output = output.clone()
            output[~nonempty] = self.empty_set
        return output


class JAFAPredictor(nn.Module):
    """Shared JAFA set encoder and classification head."""

    def __init__(
        self,
        mask_layer,
        num_classes,
        *,
        embedding_size=16,
        hidden_size=32,
        memory_size=16,
        processing_steps=5,
        linear_classifier=True,
    ):
        super().__init__()
        if not isinstance(mask_layer, MaskLayerGrouped):
            raise TypeError("JAFAPredictor requires MaskLayerGrouped")
        self.mask_layer = mask_layer
        self.num_modalities = int(mask_layer.mask_size)
        self.num_features = int(mask_layer.group_matrix.shape[1])
        self.register_buffer(
            "modality_identity", torch.eye(self.num_modalities)
        )
        self.encoder = ReadProcessSetEncoder(
            self.num_features + self.num_modalities,
            output_size=embedding_size,
            hidden_size=hidden_size,
            memory_size=memory_size,
            processing_steps=processing_steps,
        )
        if linear_classifier:
            self.classifier = nn.Linear(embedding_size, num_classes)
        else:
            self.classifier = _mlp(
                embedding_size, num_classes, hidden_size
            )

    def encode(self, feature_values, modality_mask):
        modality_mask = modality_mask.bool()
        group_matrix = self.mask_layer.group_matrix.to(feature_values)
        grouped_values = (
            feature_values.unsqueeze(1) * group_matrix.unsqueeze(0)
        )
        identities = self.modality_identity.to(feature_values).unsqueeze(0)
        identities = identities.expand(len(feature_values), -1, -1)
        elements = torch.cat([grouped_values, identities], dim=2)
        return self.encoder(elements, modality_mask)

    def encode_masked_input(self, masked_input):
        expected = self.num_features + self.num_modalities
        if masked_input.ndim != 2 or masked_input.shape[1] != expected:
            raise ValueError(
                f"masked_input must have shape [batch, {expected}]"
            )
        feature_values = masked_input[:, :self.num_features]
        modality_mask = masked_input[:, self.num_features:]
        return self.encode(feature_values, modality_mask)

    def forward(self, masked_input):
        return self.classifier(self.encode_masked_input(masked_input))


class JAFAOneShot(nn.Module):
    """Single-step JAFA subset-Q policy with a shared set classifier."""

    def __init__(
        self,
        predictor,
        mask_layer,
        feature_costs,
        *,
        embedding_size=16,
        hidden_size=32,
        acquisition_cost_weight=0.05,
        max_candidates=512,
        exact_max_paid=12,
        seed=42,
    ):
        super().__init__()
        if not isinstance(predictor, JAFAPredictor):
            raise TypeError("predictor must be JAFAPredictor")
        costs = torch.as_tensor(feature_costs, dtype=torch.float32)
        if len(costs) != mask_layer.mask_size:
            raise ValueError("feature_costs must match the modality count")
        if float(costs[0]) != 0.0:
            raise ValueError("modality 0 must be free")
        if bool((costs[1:] <= 0).any()):
            raise ValueError("paid modality costs must be positive")
        if acquisition_cost_weight < 0:
            raise ValueError("acquisition_cost_weight must be nonnegative")
        if min(max_candidates, exact_max_paid) < 1:
            raise ValueError("candidate limits must be positive")

        self.predictor = predictor
        self.mask_layer = mask_layer
        self.num_modalities = int(mask_layer.mask_size)
        self.acquisition_cost_weight = float(acquisition_cost_weight)
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
        self.q_network = _mlp(
            embedding_size, len(candidates), hidden_size
        )

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

    def q_values(self, x, *, detach_encoder=True):
        free = self.free_mask(len(x), dtype=x.dtype, device=x.device)
        free_input = self.mask_layer(x, free)
        embedding = self.predictor.encode_masked_input(free_input)
        if detach_encoder:
            embedding = embedding.detach()
        return self.q_network(embedding)

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
            return torch.zeros(len(q_values), dtype=torch.long, device=q_values.device)
        adjusted = q_values - float(omd_lambda) * self.candidate_costs
        adjusted = adjusted.masked_fill(~feasible.unsqueeze(0), -torch.inf)
        greedy = adjusted.argmax(dim=1)
        if epsilon <= 0:
            return greedy
        feasible_indices = torch.nonzero(feasible, as_tuple=False).flatten()
        random_positions = torch.randint(
            len(feasible_indices), (len(q_values),), device=q_values.device
        )
        random_actions = feasible_indices[random_positions]
        explore = torch.rand(len(q_values), device=q_values.device) < epsilon
        return torch.where(explore, random_actions, greedy)

    def _apply_global_budget(self, proposed_masks, budget_state):
        masks = self.free_mask(
            len(proposed_masks),
            dtype=proposed_masks.dtype,
            device=proposed_masks.device,
        )
        action_indices = torch.zeros(
            len(proposed_masks), dtype=torch.long, device=proposed_masks.device
        )
        total_cost = 0.0
        for row in range(len(proposed_masks)):
            proposed = proposed_masks[row]
            proposed_cost = float(
                (proposed * self.feature_costs).sum().item()
            )
            accepted_cost = proposed_cost
            used_fallback = False
            if budget_state is not None:
                accepted_cost, used_fallback = apply_global_budget_fallback(
                    proposed_cost, budget_state, fallback_cost=0.0
                )
            if not used_fallback:
                masks[row] = proposed
                matches = (self.candidate_masks == proposed.bool()).all(dim=1)
                action_indices[row] = torch.nonzero(
                    matches, as_tuple=False
                )[0, 0]
            total_cost += float(accepted_cost)
        return masks, action_indices, total_cost

    def select_features(
        self,
        x,
        budget=None,
        verbose=False,
        global_budget_state=None,
    ):
        x = x.to(self.feature_costs.device)
        with torch.no_grad():
            q_values = self.q_values(x)
        masks = self.free_mask(len(x), dtype=x.dtype, device=x.device)
        total_cost = 0.0
        for row in range(len(x)):
            omd_lambda = (
                0.0 if global_budget_state is None
                else global_budget_state.omd_lambda
            )
            action = self.choose_action_indices(
                q_values[row:row + 1], budget=budget,
                omd_lambda=omd_lambda,
            )[0]
            proposed = self.candidate_masks[action].to(x.dtype)
            accepted_cost = float(self.candidate_costs[action].item())
            used_fallback = False
            if global_budget_state is not None:
                accepted_cost, used_fallback = apply_global_budget_fallback(
                    accepted_cost, global_budget_state, fallback_cost=0.0
                )
            if not used_fallback:
                masks[row] = proposed
            total_cost += float(accepted_cost)
            if verbose:
                chosen = torch.nonzero(masks[row]).flatten().tolist()
                print(
                    f"JAFA sample {row}: subset={chosen}, "
                    f"cost={accepted_cost:.4f}"
                )
        return self.mask_layer(x, masks), masks, total_cost

    def forward(self, x, budget=None):
        x_masked, _, _ = self.select_features(x, budget=budget)
        return self.predictor(x_masked)
