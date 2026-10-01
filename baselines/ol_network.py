"""The P/Q network and confidence reward used by sequential OL."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

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
