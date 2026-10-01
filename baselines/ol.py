"""Original causal Opportunistic Learning (OL) baseline.

This is a grouped-modality port of the authors' ``Demo_OL_DQN.ipynb`` from
https://github.com/mkachuee/Opportunistic. Each training encounter is causal:

1. acquire modalities sequentially and predict;
2. record the prediction before exposing that encounter to learning;
3. append its transitions to replay memory;
4. update P-Net and, after the warm-up phase, Q-Net from replay.

The port keeps the original shared P/Q architecture, Monte-Carlo-dropout
confidence reward, P-only warm-up, Double-DQN update, cost-normalized action
ranking, transition replay, and soft target update. The intentional repository
adaptations are grouped modalities, a zero-cost free modality, strict budget
fractions, and the project's dataset/metric/CSV conventions.

The default stream contains one encounter per training row, which makes
``training_error`` directly comparable to LPChain's prequential error. Pass
``--episodes 40000`` to reproduce the original diabetes notebook's training
length (the balanced stream will then cycle over the training split).
"""

from __future__ import annotations

import argparse
import copy
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.ol_network import OLPQNetwork, confidence_change
from baselines.ol_common import (
    accuracy_metric,
    auroc_metric,
    baseline_output_filename,
    build_experiment,
    f1_metric,
)
from core.datasets import (
    ALL_DATASETS,
    DEFAULT_IMAGE_POOL_SIDE,
    DEFAULT_SAMPLING_MODE,
    SAMPLING_MODES,
    SYNTHETIC_N_CLASSES,
    SYNTHETIC_N_VIEWS,
)


@dataclass
class OLTransition:
    """A single original-OL replay tuple, stored on CPU."""

    features: torch.Tensor
    label: int
    mask: torch.Tensor
    action: int
    reward: float
    next_mask: torch.Tensor
    done: bool
    next_allowed_actions: torch.Tensor


class ExperienceReplayMemory:
    """Fixed-size transition replay with sampling with replacement."""

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("replay capacity must be positive")
        self.capacity = int(capacity)
        self.buffer: deque[OLTransition] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def extend(self, transitions: list[OLTransition]) -> None:
        self.buffer.extend(transitions)

    def sample(self, size: int) -> list[OLTransition]:
        if not self.buffer:
            return []
        indices = torch.randint(len(self.buffer), (int(size),)).tolist()
        return [self.buffer[index] for index in indices]


class OriginalOnlineOL(nn.Module):
    """The authors' online DPQN adapted to grouped, budgeted modalities."""

    def __init__(
        self,
        mask_layer,
        feature_costs,
        num_classes: int,
        *,
        hidden_sizes=(64, 32, 16),
        dropout=0.5,
        use_feature_mask=False,
        reward_method="Bayesian-L1",
        mcdrop_samples=100,
        gamma=0.0,
        target_tau=0.001,
        replay_size=None,
        replay_batch_size=128,
        learning_rate=1e-3,
        max_grad_norm=None,
        force_acquisition=True,
        cost_normalized_actions=True,
    ):
        super().__init__()
        costs = torch.as_tensor(feature_costs, dtype=torch.float32)
        if costs.ndim != 1 or len(costs) != mask_layer.mask_size:
            raise ValueError("feature_costs must match the modality count")
        if not torch.isclose(costs[0], torch.tensor(0.0)):
            raise ValueError("modality 0 must be the zero-cost free modality")
        if bool((costs[1:] <= 0).any()):
            raise ValueError("all paid modality costs must be positive")
        if mcdrop_samples < 1 or replay_batch_size < 1:
            raise ValueError("MC-dropout and replay batch sizes must be positive")
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must lie in [0, 1]")
        if not 0 < target_tau <= 1:
            raise ValueError("target_tau must lie in (0, 1]")

        self.mask_layer = mask_layer
        self.num_modalities = int(mask_layer.mask_size)
        self.num_paid = self.num_modalities - 1
        # Original actions: one per paid modality, followed by prediction/stop.
        self.stop_action = self.num_paid
        self.num_actions = self.num_paid + 1
        self.num_classes = int(num_classes)
        self.reward_method = str(reward_method)
        self.mcdrop_samples = int(mcdrop_samples)
        self.gamma = float(gamma)
        self.target_tau = float(target_tau)
        self.replay_batch_size = int(replay_batch_size)
        self.max_grad_norm = (
            None if max_grad_norm is None else float(max_grad_norm)
        )
        self.force_acquisition = bool(force_acquisition)
        self.cost_normalized_actions = bool(cost_normalized_actions)
        self.register_buffer("feature_costs", costs)

        self.network = OLPQNetwork(
            mask_layer,
            num_classes,
            self.num_actions,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            use_feature_mask=use_feature_mask,
        )
        self.target_network = copy.deepcopy(self.network)
        self.target_network.requires_grad_(False)
        capacity = (
            self.num_modalities * 1000 if replay_size is None else int(replay_size)
        )
        self.replay = ExperienceReplayMemory(capacity)
        # The original notebook uses one Adam optimizer over both P and Q.
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=float(learning_rate)
        )

    @property
    def device(self) -> torch.device:
        return self.feature_costs.device

    def free_mask(self, batch_size: int) -> torch.Tensor:
        mask = torch.zeros(
            batch_size, self.num_modalities, dtype=torch.bool, device=self.device
        )
        mask[:, 0] = True
        return mask

    def _masked_features(
        self, features: torch.Tensor, modality_mask: torch.Tensor
    ) -> torch.Tensor:
        feature_mask = (
            modality_mask.to(features) @ self.mask_layer.group_matrix.to(features)
        )
        return features * feature_mask

    def _p_logits(
        self,
        network: OLPQNetwork,
        features: torch.Tensor,
        modality_mask: torch.Tensor,
        *,
        force_dropout: bool,
    ) -> torch.Tensor:
        masked = self._masked_features(features, modality_mask)
        return network.p_logits(
            masked,
            modality_mask.to(features),
            force_dropout=force_dropout,
        )

    def _q_values(
        self,
        network: OLPQNetwork,
        features: torch.Tensor,
        modality_mask: torch.Tensor,
    ) -> torch.Tensor:
        masked = self._masked_features(features, modality_mask)
        _logits, q_values = network.forward_pq(
            masked,
            modality_mask.to(features),
            # The notebook explicitly keeps P-Net dropout active for Q.
            force_dropout=True,
        )
        return q_values

    def confidence(
        self, features: torch.Tensor, modality_mask: torch.Tensor
    ) -> torch.Tensor:
        masked_input = self.mask_layer(
            features, modality_mask.to(dtype=features.dtype)
        )
        return self.network.confidence(masked_input, self.mcdrop_samples)

    def allowed_actions(
        self,
        modality_mask: torch.Tensor,
        *,
        spent: float,
        sample_budget: float,
        global_remaining: float,
    ) -> torch.Tensor:
        """Return affordable unobserved acquisitions plus the stop action."""
        if modality_mask.ndim != 1:
            raise ValueError("allowed_actions expects one sample mask")
        allowed = torch.zeros(
            self.num_actions, dtype=torch.bool, device=modality_mask.device
        )
        remaining_sample = max(0.0, float(sample_budget) - float(spent))
        remaining = min(remaining_sample, max(0.0, float(global_remaining)))
        allowed[: self.num_paid] = (
            (~modality_mask[1:])
            & (self.feature_costs[1:] <= remaining + 1e-7)
        )
        allowed[self.stop_action] = True
        return allowed

    def choose_action(
        self,
        features: torch.Tensor,
        modality_mask: torch.Tensor,
        allowed: torch.Tensor,
        *,
        epsilon: float,
    ) -> int:
        """Original epsilon-greedy action choice with Q/cost exploitation."""
        valid = torch.nonzero(allowed, as_tuple=False).flatten()
        if len(valid) == 0:
            return self.stop_action
        if torch.rand((), device=self.device).item() < float(epsilon):
            selected = valid[torch.randint(len(valid), (1,), device=self.device)]
            return int(selected.item())

        q_values = self._q_values(
            self.network, features.unsqueeze(0), modality_mask.unsqueeze(0)
        )[0]
        scores = q_values.clone()
        if self.cost_normalized_actions:
            scores[: self.num_paid] /= self.feature_costs[1:]
        scores[~allowed] = -torch.inf
        if self.force_acquisition and bool(allowed[: self.num_paid].any()):
            # The original notebook sets prediction/stop to -1e20 until no
            # feature remains. Here "remains" also means affordable.
            scores[self.stop_action] = -torch.inf
        return int(scores.argmax().item())

    @torch.no_grad()
    def collect_episode(
        self,
        features: torch.Tensor,
        label: int,
        *,
        sample_budget: float,
        global_remaining: float,
        epsilon: float,
    ) -> dict:
        """Collect one sample trajectory without learning from that sample."""
        features = features.to(self.device)
        mask = self.free_mask(1)[0]
        spent = 0.0
        transitions: list[OLTransition] = []
        selected_modalities = [0]
        current_confidence = self.confidence(
            features.unsqueeze(0), mask.unsqueeze(0)
        )[0]

        for _step in range(self.num_paid + 1):
            allowed = self.allowed_actions(
                mask,
                spent=spent,
                sample_budget=sample_budget,
                global_remaining=global_remaining - spent,
            )
            action = self.choose_action(
                features, mask, allowed, epsilon=epsilon
            )
            next_mask = mask.clone()
            reward = 0.0
            done = action == self.stop_action

            if not done:
                modality = action + 1
                next_mask[modality] = True
                next_confidence = self.confidence(
                    features.unsqueeze(0), next_mask.unsqueeze(0)
                )[0]
                reward = float(
                    confidence_change(
                        current_confidence.unsqueeze(0),
                        next_confidence.unsqueeze(0),
                        self.reward_method,
                    ).item()
                )
                spent += float(self.feature_costs[modality].item())
                selected_modalities.append(modality)
                current_confidence = next_confidence

            next_allowed = self.allowed_actions(
                next_mask,
                spent=spent,
                sample_budget=sample_budget,
                global_remaining=global_remaining - spent,
            )
            if done:
                next_allowed[:] = False

            transitions.append(
                OLTransition(
                    features=features.detach().cpu(),
                    label=int(label),
                    mask=mask.detach().cpu(),
                    action=int(action),
                    reward=float(reward),
                    next_mask=next_mask.detach().cpu(),
                    done=bool(done),
                    next_allowed_actions=next_allowed.detach().cpu(),
                )
            )
            mask = next_mask
            if done:
                break

        prediction = int(current_confidence.argmax().item())
        return {
            "transitions": transitions,
            "prediction": prediction,
            "probabilities": current_confidence.detach().cpu(),
            "mask": mask.detach().cpu(),
            "selected_modalities": selected_modalities,
            "cost": float(spent),
        }

    def _stack_replay(self, transitions: list[OLTransition]):
        features = torch.stack([item.features for item in transitions]).to(self.device)
        labels = torch.tensor(
            [item.label for item in transitions], dtype=torch.long, device=self.device
        )
        masks = torch.stack([item.mask for item in transitions]).to(self.device)
        actions = torch.tensor(
            [item.action for item in transitions], dtype=torch.long, device=self.device
        )
        rewards = torch.tensor(
            [item.reward for item in transitions], dtype=torch.float32, device=self.device
        )
        next_masks = torch.stack([item.next_mask for item in transitions]).to(
            self.device
        )
        dones = torch.tensor(
            [item.done for item in transitions], dtype=torch.bool, device=self.device
        )
        next_allowed = torch.stack(
            [item.next_allowed_actions for item in transitions]
        ).to(self.device)
        return (
            features,
            labels,
            masks,
            actions,
            rewards,
            next_masks,
            dones,
            next_allowed,
        )

    def replay_update(self, *, train_q: bool) -> tuple[float, float]:
        """Perform the notebook's P update and optional Double-DQN Q update."""
        sampled = self.replay.sample(self.replay_batch_size)
        if not sampled:
            return math.nan, math.nan
        (
            features,
            labels,
            masks,
            actions,
            rewards,
            next_masks,
            dones,
            next_allowed,
        ) = self._stack_replay(sampled)

        self.network.train()
        logits = self._p_logits(
            self.network, features, masks, force_dropout=True
        )
        p_loss = F.cross_entropy(logits, labels)
        self.optimizer.zero_grad()
        p_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(
                self.network.parameters(), self.max_grad_norm
            )
        self.optimizer.step()

        q_loss_value = math.nan
        if train_q:
            q_values = self._q_values(self.network, features, masks)
            chosen_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                online_next_q = self._q_values(
                    self.network, features, next_masks
                )
                target_allowed = next_allowed.clone()
                # Terminal rows need one finite placeholder for argmax; their
                # bootstrapped value is removed immediately afterward.
                target_allowed[dones, self.stop_action] = True
                if self.force_acquisition:
                    has_acquisition = target_allowed[:, : self.num_paid].any(dim=1)
                    target_allowed[has_acquisition, self.stop_action] = False
                online_next_q = online_next_q.masked_fill(
                    ~target_allowed, -torch.inf
                )
                next_actions = online_next_q.argmax(dim=1)
                target_next_q = self._q_values(
                    self.target_network, features, next_masks
                ).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                targets = rewards + self.gamma * (~dones).float() * target_next_q

            q_loss = F.mse_loss(chosen_q, targets)
            self.optimizer.zero_grad()
            q_loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.max_grad_norm
                )
            self.optimizer.step()
            q_loss_value = float(q_loss.item())

        self._soft_update_target()
        return float(p_loss.item()), q_loss_value

    @torch.no_grad()
    def _soft_update_target(self) -> None:
        for target, current in zip(
            self.target_network.parameters(), self.network.parameters()
        ):
            target.mul_(1.0 - self.target_tau).add_(
                current, alpha=self.target_tau
            )


def _balanced_stream_indices(
    labels: torch.Tensor,
    n_episodes: int,
    *,
    rng: np.random.Generator,
    balanced: bool,
) -> np.ndarray:
    """Create the notebook's cycling stream, optionally alternating classes."""
    labels_np = labels.detach().cpu().numpy().astype(int)
    if n_episodes < 1 or len(labels_np) < 1:
        raise ValueError("the stream and episode count must be non-empty")
    if not balanced:
        output = []
        while len(output) < n_episodes:
            output.extend(rng.permutation(len(labels_np)).tolist())
        return np.asarray(output[:n_episodes], dtype=int)

    classes = np.unique(labels_np)
    pools = {label: np.flatnonzero(labels_np == label) for label in classes}
    for label in classes:
        rng.shuffle(pools[label])
    positions = {label: 0 for label in classes}
    output = np.empty(n_episodes, dtype=int)
    for episode in range(n_episodes):
        label = classes[episode % len(classes)]
        if positions[label] >= len(pools[label]):
            rng.shuffle(pools[label])
            positions[label] = 0
        output[episode] = pools[label][positions[label]]
        positions[label] += 1
    return output


def _classification_metrics(
    probabilities: list[torch.Tensor], labels: list[int]
) -> dict[str, float]:
    probs = torch.stack(probabilities).float()
    y = torch.tensor(labels, dtype=torch.long)
    # The shared AUROC helper applies softmax, so log-probabilities reproduce
    # the already-normalized MC-dropout probabilities.
    logits = probs.clamp_min(1e-12).log()
    return {
        "accuracy": float(accuracy_metric(logits, y).item()),
        "f1": float(f1_metric(logits, y).item()),
        "auroc": float(auroc_metric(logits, y).item()),
    }


def run_online_training(
    model: OriginalOnlineOL,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    sample_budget: float,
    n_episodes: int,
    p_phase_fraction: float,
    epsilon_start: float,
    epsilon_end: float,
    updates_per_sample: int,
    balanced_stream: bool,
    seed: int,
    log_every: int,
) -> dict:
    """Run a causal stream and return LPChain-compatible training metrics."""
    if not 0 <= p_phase_fraction <= 1:
        raise ValueError("p_phase_fraction must lie in [0, 1]")
    if not 0 <= epsilon_end <= epsilon_start <= 1:
        raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1")
    if updates_per_sample < 1:
        raise ValueError("updates_per_sample must be positive")

    rng = np.random.default_rng(seed)
    order = _balanced_stream_indices(
        labels,
        n_episodes,
        rng=rng,
        balanced=balanced_stream,
    )
    total_budget = float(sample_budget) * n_episodes
    remaining_budget = total_budget
    p_phase_episodes = int(n_episodes * p_phase_fraction)
    epsilon = float(epsilon_start)
    if epsilon_start > 0 and epsilon_end > 0:
        epsilon_decay = math.exp(
            math.log(epsilon_end / epsilon_start) / n_episodes
        )
    else:
        epsilon_decay = 1.0

    predictions: list[int] = []
    observed_labels: list[int] = []
    probabilities: list[torch.Tensor] = []
    cost_trace: list[float] = []
    acquisition_reward_trace: list[float] = []
    selected_modalities: list[list[int]] = []
    p_losses: list[float] = []
    q_losses: list[float] = []

    for episode, row in enumerate(order):
        train_q = episode >= p_phase_episodes
        episode_epsilon = epsilon if train_q else 1.0
        label = int(labels[row].item())
        result = model.collect_episode(
            features[row],
            label,
            sample_budget=sample_budget,
            global_remaining=remaining_budget,
            epsilon=episode_epsilon,
        )

        # Record prediction first. Learning from this encounter begins below.
        predictions.append(result["prediction"])
        observed_labels.append(label)
        probabilities.append(result["probabilities"])
        cost_trace.append(result["cost"])
        acquisition_reward_trace.append(
            float(sum(item.reward for item in result["transitions"]))
        )
        selected_modalities.append(result["selected_modalities"])
        remaining_budget = max(0.0, remaining_budget - result["cost"])

        model.replay.extend(result["transitions"])
        for _ in range(updates_per_sample):
            p_loss, q_loss = model.replay_update(train_q=train_q)
            if math.isfinite(p_loss):
                p_losses.append(p_loss)
            if math.isfinite(q_loss):
                q_losses.append(q_loss)

        if train_q:
            epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if log_every and (
            episode == 0
            or (episode + 1) % log_every == 0
            or episode + 1 == n_episodes
        ):
            correct_so_far = np.equal(predictions, observed_labels).mean()
            print(
                f"  OL episode {episode + 1}/{n_episodes}: "
                f"phase={'Q' if train_q else 'P'}, eps={episode_epsilon:.3f}, "
                f"online_error={1.0 - correct_so_far:.4f}, "
                f"replay={len(model.replay)}"
            )

    metrics = _classification_metrics(probabilities, observed_labels)
    correct = np.equal(predictions, observed_labels).astype(float)
    denominator = np.arange(1, len(correct) + 1, dtype=float)
    spent = float(sum(cost_trace))
    return {
        "train_reward": metrics["accuracy"],
        "training_error": 1.0 - metrics["accuracy"],
        "training_regret": 1.0 - metrics["accuracy"],
        "train_f1": metrics["f1"],
        "train_auroc": metrics["auroc"],
        "cum_train_reward": (np.cumsum(correct) / denominator).tolist(),
        "cumulative_mistakes": np.cumsum(1.0 - correct).tolist(),
        "train_predictions": predictions,
        "train_labels": observed_labels,
        "selected_modalities": selected_modalities,
        "train_cost_trace": cost_trace,
        "ol_acquisition_reward_trace": acquisition_reward_trace,
        "ol_mean_acquisition_reward": float(np.mean(acquisition_reward_trace)),
        "ol_cumulative_acquisition_reward": float(sum(acquisition_reward_trace)),
        "train_spent": spent,
        "train_total_budget": total_budget,
        "train_avg_cost_per_sample": spent / n_episodes,
        "train_budget_utilization": spent / total_budget if total_budget > 0 else 0.0,
        "episodes": n_episodes,
        "p_phase_episodes": p_phase_episodes,
        "replay_transitions": len(model.replay),
        "final_epsilon": epsilon,
        "mean_p_loss": float(np.mean(p_losses)) if p_losses else math.nan,
        "final_p_loss": p_losses[-1] if p_losses else math.nan,
        "mean_q_loss": float(np.mean(q_losses)) if q_losses else math.nan,
        "final_q_loss": q_losses[-1] if q_losses else math.nan,
    }


@torch.no_grad()
def evaluate_stream(
    model: OriginalOnlineOL,
    dataset,
    *,
    sample_budget: float,
) -> dict[str, float]:
    """Evaluate greedily without labels, replay insertion, or updates."""
    features, labels = dataset.tensors
    total_budget = float(sample_budget) * len(dataset)
    remaining_budget = total_budget
    observed_labels = []
    probabilities = []
    costs = []
    for row in range(len(dataset)):
        label = int(labels[row].item())
        result = model.collect_episode(
            features[row],
            label,
            sample_budget=sample_budget,
            global_remaining=remaining_budget,
            epsilon=0.0,
        )
        observed_labels.append(label)
        probabilities.append(result["probabilities"])
        costs.append(result["cost"])
        remaining_budget = max(0.0, remaining_budget - result["cost"])

    metrics = _classification_metrics(probabilities, observed_labels)
    spent = float(sum(costs))
    return {
        **metrics,
        "avg_cost_per_sample": spent / max(len(dataset), 1),
        "spent": spent,
        "total_budget": total_budget,
        "budget_utilization": spent / total_budget if total_budget > 0 else 0.0,
    }


def _parse_numbers(value, cast, label):
    try:
        parsed = [cast(item.strip()) for item in str(value).split(",") if item.strip()]
    except ValueError as error:
        raise ValueError(f"invalid {label}: {value!r}") from error
    if not parsed:
        raise ValueError(f"{label} cannot be empty")
    return parsed


def run_experiments(args) -> list[dict]:
    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    budget_fractions = _parse_numbers(
        args.budget_fractions, float, "budget fractions"
    )
    seeds = _parse_numbers(args.seeds, int, "seeds")
    hidden_sizes = tuple(_parse_numbers(args.hidden_sizes, int, "hidden sizes"))
    if any(not 0 <= value <= 1 for value in budget_fractions):
        raise ValueError("budget fractions must lie in [0, 1]")
    max_modalities = (
        None if str(args.max_modalities).lower() == "all" else int(args.max_modalities)
    )

    rows: list[dict] = []
    for seed in seeds:
        print(f"\n{'=' * 70}\nOriginal online OL seed {seed}\n{'=' * 70}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        ctx = build_experiment(
            args.dataset,
            device,
            data_path=args.data_path,
            max_modalities=max_modalities,
            split_seed=seed,
            synthetic_seed=(seed if args.synthetic_seed is None else args.synthetic_seed),
            nsamples=args.nsamples,
            n_views=args.n_views,
            max_samples=args.max_samples,
            sampling=args.sampling,
            image_pool_side=args.image_pool_side,
            image_data_home=args.image_cache_dir,
            num_classes=args.num_classes,
        )
        train_features, train_labels = ctx["train_dataset"].tensors
        n_episodes = len(train_features) if args.episodes is None else args.episodes
        total_paid_cost = float(sum(ctx["paid_costs"]))
        n_train = len(ctx["train_dataset"])
        n_test = len(ctx["test_dataset"])

        for budget_fraction in budget_fractions:
            torch.manual_seed(seed)
            np.random.seed(seed)
            sample_budget = float(budget_fraction) * total_paid_cost
            print(
                f"\nBudget fraction {budget_fraction:.3f} "
                f"(per-encounter cap {sample_budget:.6f})"
            )
            model = OriginalOnlineOL(
                ctx["mask_layer"],
                ctx["feature_costs"],
                ctx["d_out"],
                hidden_sizes=hidden_sizes,
                dropout=args.dropout,
                use_feature_mask=args.use_feature_mask,
                reward_method=args.reward_method,
                mcdrop_samples=args.mcdrop_samples,
                gamma=args.gamma,
                target_tau=args.target_tau,
                replay_size=args.replay_size,
                replay_batch_size=args.replay_batch_size,
                learning_rate=args.learning_rate,
                max_grad_norm=args.max_grad_norm,
                force_acquisition=not args.allow_early_stop,
                cost_normalized_actions=not args.disable_cost_normalization,
            ).to(device)

            train_started = time.time()
            training = run_online_training(
                model,
                train_features,
                train_labels,
                sample_budget=sample_budget,
                n_episodes=n_episodes,
                p_phase_fraction=args.p_phase_fraction,
                epsilon_start=args.epsilon_start,
                epsilon_end=args.epsilon_end,
                updates_per_sample=args.updates_per_sample,
                balanced_stream=not args.unbalanced_stream,
                seed=seed,
                log_every=args.log_every,
            )
            train_time = time.time() - train_started

            inference_started = time.time()
            validation = evaluate_stream(
                model, ctx["val_dataset"], sample_budget=sample_budget
            )
            test = evaluate_stream(
                model, ctx["test_dataset"], sample_budget=sample_budget
            )
            inference_time = time.time() - inference_started
            print(
                f"  train online error={training['training_error']:.4f}, "
                f"test accuracy={test['accuracy']:.4f}, "
                f"F1={test['f1']:.4f}, AUROC={test['auroc']:.4f}, "
                f"avg cost={test['avg_cost_per_sample']:.6f}"
            )

            rows.append(
                {
                    "dataset": ctx["dataset_name"],
                    "split_mode": ctx["split_mode"],
                    "method": "ol",
                    "ol_variant": "original_online_causal",
                    "online_protocol": "predict_then_replay_update",
                    "seed": seed,
                    "num_modalities": int(ctx["num_modalities"]),
                    "num_classes": int(ctx["d_out"]),
                    "budget_fraction": float(budget_fraction),
                    "total_paid_cost": total_paid_cost,
                    "budget_per_sample": sample_budget,
                    "total_budget": sample_budget * (n_train + n_test),
                    "train_total": sample_budget * n_train,
                    "test_total": sample_budget * n_test,
                    "train_budget_per_sample": sample_budget,
                    "test_budget_per_sample": sample_budget,
                    "train_reward": training["train_reward"],
                    "training_error": training["training_error"],
                    "training_regret": training["training_regret"],
                    "train_f1": training["train_f1"],
                    "train_auroc": training["train_auroc"],
                    "cum_train_reward": training["cum_train_reward"],
                    "cumulative_mistakes": training["cumulative_mistakes"],
                    "train_predictions": training["train_predictions"],
                    "selected_modalities": training["selected_modalities"],
                    "train_cost_trace": training["train_cost_trace"],
                    "ol_acquisition_reward_trace": training[
                        "ol_acquisition_reward_trace"
                    ],
                    "ol_mean_acquisition_reward": training[
                        "ol_mean_acquisition_reward"
                    ],
                    "ol_cumulative_acquisition_reward": training[
                        "ol_cumulative_acquisition_reward"
                    ],
                    "train_spent": training["train_spent"],
                    "train_total_budget": training["train_total_budget"],
                    "train_avg_cost_per_sample": training[
                        "train_avg_cost_per_sample"
                    ],
                    "train_budget_utilization": training[
                        "train_budget_utilization"
                    ],
                    "val_accuracy": validation["accuracy"],
                    "val_f1": validation["f1"],
                    "val_auroc": validation["auroc"],
                    "val_avg_cost_per_sample": validation[
                        "avg_cost_per_sample"
                    ],
                    "test_accuracy": test["accuracy"],
                    "test_f1": test["f1"],
                    "test_auroc": test["auroc"],
                    "test_avg_cost_per_sample": test[
                        "avg_cost_per_sample"
                    ],
                    "test_spent": test["spent"],
                    "test_total_budget": test["total_budget"],
                    "test_budget_utilization": test["budget_utilization"],
                    "train_time_sec": train_time,
                    "inference_time_sec": inference_time,
                    "trial_wall_time_sec": train_time + inference_time,
                    "episodes": training["episodes"],
                    "p_phase_episodes": training["p_phase_episodes"],
                    "replay_transitions": training["replay_transitions"],
                    "final_epsilon": training["final_epsilon"],
                    "mean_p_loss": training["mean_p_loss"],
                    "final_p_loss": training["final_p_loss"],
                    "mean_q_loss": training["mean_q_loss"],
                    "final_q_loss": training["final_q_loss"],
                    "hidden_sizes": list(hidden_sizes),
                    "dropout": args.dropout,
                    "use_feature_mask": args.use_feature_mask,
                    "reward_method": args.reward_method,
                    "mcdrop_samples": args.mcdrop_samples,
                    "gamma": args.gamma,
                    "target_tau": args.target_tau,
                    "replay_size": model.replay.capacity,
                    "replay_batch_size": args.replay_batch_size,
                    "learning_rate": args.learning_rate,
                    "updates_per_sample": args.updates_per_sample,
                    "p_phase_fraction": args.p_phase_fraction,
                    "epsilon_start": args.epsilon_start,
                    "epsilon_end": args.epsilon_end,
                    "balanced_stream": not args.unbalanced_stream,
                    "force_acquisition": not args.allow_early_stop,
                    "cost_normalized_actions": (
                        not args.disable_cost_normalization
                    ),
                }
            )
    return rows


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the authors' original causal online OL protocol under this "
            "repository's grouped modalities and budget fractions."
        )
    )
    parser.add_argument("--dataset", choices=ALL_DATASETS, default="synthetic")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--max-modalities", default="all")
    parser.add_argument("--n-views", type=int, default=SYNTHETIC_N_VIEWS)
    parser.add_argument(
        "--n-samples", "--nsamples", dest="nsamples", type=int, default=1000
    )
    parser.add_argument("--synthetic-seed", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--sampling", choices=SAMPLING_MODES, default=DEFAULT_SAMPLING_MODE
    )
    parser.add_argument("--num-classes", type=int, default=SYNTHETIC_N_CLASSES)
    parser.add_argument(
        "--image-pool-side", type=int, default=DEFAULT_IMAGE_POOL_SIDE
    )
    parser.add_argument("--image-cache-dir", default=None)
    parser.add_argument("--budget-fractions", default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--device", default=None)

    parser.add_argument("--hidden-sizes", default="64,32,16")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--use-feature-mask", action="store_true")
    parser.add_argument(
        "--reward-method",
        choices=("softmax", "Bayesian-L1", "Bayesian-L2"),
        default="Bayesian-L1",
    )
    parser.add_argument("--mcdrop-samples", type=int, default=100)
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help=(
            "Training encounters per budget. Default: one per training row. "
            "Use 40000 for the original diabetes-notebook length."
        ),
    )
    parser.add_argument("--p-phase-fraction", type=float, default=0.1)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--target-tau", type=float, default=0.001)
    parser.add_argument("--replay-size", type=int, default=None)
    parser.add_argument("--replay-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--updates-per-sample", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--allow-early-stop",
        action="store_true",
        help="Let greedy Q choose stop while affordable modalities remain.",
    )
    parser.add_argument(
        "--disable-cost-normalization",
        action="store_true",
        help="Select by raw Q rather than the original Q divided by cost.",
    )
    parser.add_argument(
        "--unbalanced-stream",
        action="store_true",
        help="Shuffle rows instead of alternating classes as the notebook does.",
    )
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    if args.episodes is not None and args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.mcdrop_samples < 1 or args.replay_batch_size < 1:
        parser.error("MC-dropout and replay batch sizes must be positive")
    if args.updates_per_sample < 1:
        parser.error("--updates-per-sample must be positive")
    if not 0 <= args.p_phase_fraction <= 1:
        parser.error("--p-phase-fraction must lie in [0, 1]")
    if not 0 <= args.epsilon_end <= args.epsilon_start <= 1:
        parser.error("epsilon values must satisfy 0 <= end <= start <= 1")

    rows = run_experiments(args)
    if args.output_csv:
        output_path = Path(args.output_csv)
    else:
        dimensions = {
            (int(row["num_modalities"]), int(row["num_classes"])) for row in rows
        }
        if len(dimensions) != 1:
            raise RuntimeError("result rows disagree on modality/class dimensions")
        n_modalities, n_classes = dimensions.pop()
        output_path = Path("results") / baseline_output_filename(
            "ol",
            args.dataset,
            n_modalities,
            len(_parse_numbers(args.seeds, int, "seeds")),
            n_classes,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = [
        {
            key: str(value) if isinstance(value, (list, dict)) else value
            for key, value in row.items()
        }
        for row in rows
    ]
    pd.DataFrame(serializable).to_csv(output_path, index=False)
    print(f"\nSaved {len(rows)} original-online OL rows to {output_path}")


# Compatibility names for code that imported the earlier standalone module.
SequentialOL = OriginalOnlineOL
EpisodeReplayMemory = ExperienceReplayMemory


if __name__ == "__main__":
    main()
