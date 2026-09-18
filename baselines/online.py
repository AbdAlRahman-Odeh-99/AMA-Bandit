"""Causal online implementations built from the repository's model components.

The implementations use the same predict-then-update convention as
``adaptive.run_phase1_training``:

1. acquire a subset using only the current model state;
2. predict and record correctness before revealing the label;
3. update the baseline models using that observation and its acquired mask.

The ``*_oneshot`` modules imported below provide reusable model and acquisition
components; this module owns the supported online training protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from baselines.aaco_oneshot import AACOOneShot
from baselines.cwcf_oneshot import CwCFOneShot
from baselines.dime_oneshot import DIMEOneShot
from baselines.eddi_oneshot import EDDI
from baselines.gdfs_oneshot import GDFSOneShot
from baselines.jafa_oneshot import JAFAOneShot, JAFAPredictor
from baselines.ol_oneshot import OLOneShot
from baselines.pt_oneshot import PTOneShot
from baselines import pvae
from core.budget_state import BudgetState, apply_global_budget_fallback
from core.utils import get_entropy, get_linear_network, get_mlp_network


TensorStep = Callable[[torch.Tensor, BudgetState], tuple[torch.Tensor, torch.Tensor, float]]
ScoreStep = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
UpdateStep = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None]


@dataclass
class OnlineModel:
    """Callbacks and concrete model objects needed by an online run."""

    acquire_and_predict: TensorStep
    score_with_mask: ScoreStep
    update: UpdateStep
    predictor: nn.Module
    acquisition_model: nn.Module


def run_prequential_stream(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    nclasses: int,
    feature_costs,
    budget_state: BudgetState,
    acquire_and_predict: TensorStep,
    score_with_mask: ScoreStep,
    update: UpdateStep,
    rng: np.random.Generator,
) -> dict:
    """Run one causal training pass and return adaptive-compatible metrics.

    The first ``nclasses`` rows mirror adaptive's initialization: request the
    full modality set, record a uniform-random class prediction, and only then
    update.  If the remaining global pool cannot afford the full set, the row
    receives the free-only mask, matching adaptive's hard fallback semantics.

    ``acquire_and_predict`` is responsible for charging non-initialization
    acquisitions to ``budget_state``.  This function charges initialization
    acquisitions itself.  ``update`` is always invoked after the prediction
    has been appended to the trace; this ordering is intentionally testable.
    """
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows")
    if nclasses < 1:
        raise ValueError("nclasses must be positive")

    device = X.device
    costs = torch.as_tensor(feature_costs, dtype=torch.float32, device=device)
    nviews = int(costs.numel())
    full_cost = float(costs.sum().item())
    free_mask = torch.zeros(1, nviews, dtype=X.dtype, device=device)
    free_mask[:, 0] = 1.0

    predictions: list[int] = []
    labels: list[int] = []
    logits_trace: list[torch.Tensor] = []
    masks: list[list[int]] = []
    costs_trace: list[float] = []

    for t in range(len(X)):
        x_t = X[t:t + 1]
        y_t = y[t:t + 1]

        if t < nclasses:
            mask_t = torch.ones(1, nviews, dtype=X.dtype, device=device)
            accepted_cost, used_fallback = apply_global_budget_fallback(
                full_cost, budget_state, fallback_cost=0.0
            )
            if used_fallback:
                mask_t = free_mask.clone()
            with torch.no_grad():
                logits_t = score_with_mask(x_t, mask_t)
            pred_t = int(rng.integers(nclasses))
            cost_t = float(accepted_cost)
        else:
            logits_t, mask_t, cost_t = acquire_and_predict(x_t, budget_state)
            pred_t = int(logits_t.argmax(dim=1).item())

        # Record the causal prediction before the current observation is used
        # by either the predictor or the acquisition model.
        predictions.append(pred_t)
        labels.append(int(y_t.item()))
        logits_trace.append(logits_t.detach().cpu())
        masks.append(
            torch.nonzero(mask_t[0], as_tuple=False).flatten().detach().cpu().tolist()
        )
        costs_trace.append(float(cost_t))

        # The shared online budget uses the same per-round dual update as the
        # proposed method.  apply_global_budget_fallback already charges the
        # hard pool; this updates only its Lagrange price.
        budget_state.update_lambda(cost_t)
        update(x_t, y_t, mask_t)

    pred_arr = np.asarray(predictions, dtype=int)
    label_arr = np.asarray(labels, dtype=int)
    correct = (pred_arr == label_arr).astype(float)
    denom = np.arange(1, len(correct) + 1, dtype=float)

    return {
        "train_reward": float(correct.mean()) if len(correct) else float("nan"),
        "training_error": float(1.0 - correct.mean()) if len(correct) else float("nan"),
        "cum_train_reward": (np.cumsum(correct) / denom).tolist(),
        "cumulative_mistakes": np.cumsum(1.0 - correct).tolist(),
        "train_predictions": predictions,
        "train_labels": labels,
        "train_logits": torch.cat(logits_trace, dim=0) if logits_trace else torch.empty(0, nclasses),
        "selected_modalities": masks,
        "train_cost_trace": costs_trace,
        "train_spent": float(sum(costs_trace)),
    }


class OnlineCAESelector(nn.Module):
    """Causal, hard-subset counterpart of CAE's global Concrete selector.

    The policy retains CAE's multiple categorical selection heads, but samples
    without replacement and exposes only a hard union mask.  Consequently a
    training prediction never reads an unacquired value.  Selector learning is
    performed with the log-probability returned by ``sample_training_mask``;
    using CAE's usual relaxed path here would leak unacquired values through
    the backward pass.
    """

    @staticmethod
    def _max_features_for_budget(paid_costs, budget):
        """Return how many cheapest paid modalities fit in ``budget``."""
        if budget is None:
            return len(paid_costs)
        spent = 0.0
        count = 0
        for cost in sorted(paid_costs):
            if spent + cost > budget:
                break
            spent += cost
            count += 1
        return count

    def __init__(self, mask_layer, feature_costs, train_budget, gamma=0.2):
        super().__init__()
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        self.mask_layer = mask_layer
        self.feature_costs = [float(c) for c in feature_costs]
        self.num_modalities = int(mask_layer.mask_size)
        self.num_paid = self.num_modalities - 1
        self.num_select = self._max_features_for_budget(
            self.feature_costs[1:], train_budget
        )
        self.num_select = max(0, min(self.num_select, self.num_paid))
        self.gamma = float(gamma)
        self.selector_logits = nn.Parameter(
            torch.randn(self.num_select, self.num_paid, dtype=torch.float32)
        )
        self.register_buffer(
            "paid_costs", torch.tensor(self.feature_costs[1:], dtype=torch.float32)
        )

    def _head_logits(self, head, omd_lambda, selected):
        logits = self.selector_logits[head] - float(omd_lambda) * self.paid_costs
        logits = logits / self.gamma
        if selected:
            logits = logits.clone()
            logits[list(selected)] = -torch.inf
        return logits

    def _hard_selection(self, omd_lambda=0.0):
        selected = []
        selected_set = set()
        for head in range(self.num_select):
            logits = self._head_logits(head, omd_lambda, selected_set)
            choice = int(torch.argmax(logits).item())
            selected.append(choice)
            selected_set.add(choice)
        return selected

    def sample_training_mask(self, x, budget_state):
        """Sample one hard one-shot subset without inspecting ``x`` values."""
        selected = []
        selected_set = set()
        log_probs = []
        entropies = []
        for head in range(self.num_select):
            dist = Categorical(
                logits=self._head_logits(
                    head, budget_state.omd_lambda, selected_set
                )
            )
            choice = dist.sample()
            selected.append(int(choice.item()))
            selected_set.add(int(choice.item()))
            log_probs.append(dist.log_prob(choice))
            entropies.append(dist.entropy())

        proposed_cost = float(sum(self.feature_costs[j + 1] for j in selected))
        accepted_cost, used_fallback = apply_global_budget_fallback(
            proposed_cost, budget_state, fallback_cost=0.0
        )
        mask = torch.zeros(
            len(x), self.num_modalities, dtype=x.dtype, device=x.device
        )
        mask[:, 0] = 1.0
        if not used_fallback:
            for j in selected:
                mask[:, j + 1] = 1.0

        log_prob = torch.stack(log_probs).sum() if log_probs else None
        entropy = torch.stack(entropies).sum() if entropies else None
        if used_fallback:
            # The sampled action was not executed and therefore must not be
            # reinforced using the free-only prediction's loss.
            log_prob = None
            entropy = None
        return mask, float(accepted_cost), log_prob, entropy

    def select_features(self, x, budget=None, verbose=False,
                        global_budget_state=None):
        """Apply the frozen, deterministic online-CAE subset at evaluation."""
        omd_lambda = (0.0 if global_budget_state is None
                      else global_budget_state.omd_lambda)
        selected = self._hard_selection(omd_lambda)
        if global_budget_state is None and budget is not None:
            kept = []
            spent = 0.0
            for j in selected:
                cost = self.feature_costs[j + 1]
                if spent + cost <= budget + 1e-12:
                    kept.append(j)
                    spent += cost
            selected = kept

        proposed_cost = float(sum(self.feature_costs[j + 1] for j in selected))
        mask = torch.zeros(
            len(x), self.num_modalities, dtype=x.dtype, device=x.device
        )
        mask[:, 0] = 1.0
        total_cost = 0.0
        for i in range(len(x)):
            accepted_cost = proposed_cost
            used_fallback = False
            if global_budget_state is not None:
                accepted_cost, used_fallback = apply_global_budget_fallback(
                    proposed_cost, global_budget_state, fallback_cost=0.0
                )
            if not used_fallback:
                for j in selected:
                    mask[i, j + 1] = 1.0
            total_cost += float(accepted_cost)

        if verbose:
            print(f"Online CAE fixed subset={selected}, total_cost={total_cost:.4f}")
        return self.mask_layer(x, mask), mask, total_cost


def build_online_cae(
    ctx,
    device,
    *,
    train_budget,
    learning_rate=1e-3,
    updates_per_sample=1,
    entropy_weight=1e-2,
    gamma=0.2,
    linear_classifier=True,
) -> OnlineModel:
    """Construct a strict predict-then-update online CAE adaptation."""
    if updates_per_sample < 1:
        raise ValueError("updates_per_sample must be at least 1")
    if entropy_weight < 0:
        raise ValueError("entropy_weight must be nonnegative")

    classifier_fn = get_linear_network if linear_classifier else get_mlp_network
    d_in = ctx["d_in"]
    d_out = ctx["d_out"]
    nviews = ctx["num_modalities"]
    mask_layer = ctx["mask_layer"]
    predictor = classifier_fn(d_in + nviews, d_out).to(device)
    selector = OnlineCAESelector(
        mask_layer, ctx["feature_costs"], train_budget, gamma=gamma
    ).to(device)
    predictor_opt = torch.optim.Adam(predictor.parameters(), lr=learning_rate)
    selector_opt = (torch.optim.Adam([selector.selector_logits], lr=learning_rate)
                    if selector.num_select > 0 else None)
    classification_loss = nn.CrossEntropyLoss()
    pending = {"log_prob": None, "entropy": None, "logits": None}
    loss_baseline = {"value": None}

    def score_with_mask(x_t, mask_t):
        pending.update(log_prob=None, entropy=None, logits=None)
        predictor.eval()
        return predictor(mask_layer(x_t, mask_t))

    def acquire_and_predict(x_t, budget_state):
        selector.train()
        mask_t, cost_t, log_prob, entropy = selector.sample_training_mask(
            x_t, budget_state
        )
        predictor.eval()
        with torch.no_grad():
            logits_t = predictor(mask_layer(x_t, mask_t))
        pending.update(
            log_prob=log_prob,
            entropy=entropy,
            logits=logits_t.detach(),
        )
        return logits_t, mask_t, cost_t

    def update(x_t, y_t, mask_t):
        pre_update_loss = None
        if pending["logits"] is not None:
            pre_update_loss = float(
                F.cross_entropy(pending["logits"], y_t).item()
            )

        # The predictor only receives the hard acquired mask.  Repeating this
        # update is safe; the selector receives one policy update per action.
        for _ in range(updates_per_sample):
            predictor.train()
            predictor_opt.zero_grad()
            logits = predictor(mask_layer(x_t, mask_t))
            loss = classification_loss(logits, y_t)
            loss.backward()
            predictor_opt.step()

        if selector_opt is not None and pending["log_prob"] is not None:
            baseline = (pre_update_loss if loss_baseline["value"] is None
                        else loss_baseline["value"])
            advantage = float(pre_update_loss - baseline)
            selector_opt.zero_grad()
            policy_loss = advantage * pending["log_prob"]
            if pending["entropy"] is not None:
                policy_loss = policy_loss - entropy_weight * pending["entropy"]
            policy_loss.backward()
            selector_opt.step()
            if loss_baseline["value"] is None:
                loss_baseline["value"] = pre_update_loss
            else:
                loss_baseline["value"] = (
                    0.9 * loss_baseline["value"] + 0.1 * pre_update_loss
                )

        pending.update(log_prob=None, entropy=None, logits=None)

    return OnlineModel(acquire_and_predict, score_with_mask, update,
                       predictor, selector)


def build_online_aaco(
    ctx,
    device,
    *,
    train_budget,
    learning_rate=1e-3,
    updates_per_sample=1,
    k_neighbors=5,
    acquisition_cost=0.05,
    max_candidates=100,
    exact_max_paid=12,
    linear_classifier=True,
    seed=42,
) -> OnlineModel:
    """Construct a causal AACO with a growing acquired-only KNN bank."""
    if updates_per_sample < 1:
        raise ValueError("updates_per_sample must be at least 1")
    classifier_fn = get_linear_network if linear_classifier else get_mlp_network
    d_in = ctx["d_in"]
    d_out = ctx["d_out"]
    nviews = ctx["num_modalities"]
    mask_layer = ctx["mask_layer"]
    predictor = classifier_fn(d_in + nviews, d_out).to(device)
    aaco = AACOOneShot(
        predictor,
        mask_layer,
        ctx["feature_costs"],
        k_neighbors=k_neighbors,
        acquisition_cost=acquisition_cost,
        max_candidates=max_candidates,
        exact_max_paid=exact_max_paid,
        seed=seed,
    ).to(device)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=learning_rate)
    classification_loss = nn.CrossEntropyLoss()

    def score_with_mask(x_t, mask_t):
        predictor.eval()
        return predictor(mask_layer(x_t, mask_t))

    def acquire_and_predict(x_t, budget_state):
        predictor.eval()
        with torch.no_grad():
            x_masked, mask_t, cost_t = aaco.select_features(
                x_t,
                budget=train_budget,
                verbose=False,
                global_budget_state=budget_state,
            )
            logits_t = predictor(x_masked)
        return logits_t, mask_t, float(cost_t)

    def update(x_t, y_t, mask_t):
        for _ in range(updates_per_sample):
            predictor.train()
            optimizer.zero_grad()
            logits = predictor(mask_layer(x_t, mask_t))
            loss = classification_loss(logits, y_t)
            loss.backward()
            optimizer.step()
        # Append only after prediction/update ordering has been respected, and
        # retain only values that were genuinely acquired for this row.
        aaco.add_reference(x_t, y_t, mask_t)

    return OnlineModel(acquire_and_predict, score_with_mask, update,
                       predictor, aaco)


def build_online_gdfs(
    ctx,
    device,
    *,
    train_budget,
    learning_rate=1e-3,
    updates_per_sample=1,
    entropy_weight=1e-3,
    linear_classifier=True,
) -> OnlineModel:
    """Construct acquired-only online GDFS with REINFORCE selector updates."""
    if updates_per_sample < 1:
        raise ValueError("updates_per_sample must be at least 1")
    classifier_fn = get_linear_network if linear_classifier else get_mlp_network
    d_in = ctx["d_in"]
    d_out = ctx["d_out"]
    nviews = ctx["num_modalities"]
    mask_layer = ctx["mask_layer"]
    predictor = classifier_fn(d_in + nviews, d_out).to(device)
    selector = get_mlp_network(d_in + nviews, nviews).to(device)
    gdfs = GDFSOneShot(
        selector, predictor, mask_layer, ctx["feature_costs"]
    ).to(device)
    predictor_opt = torch.optim.Adam(predictor.parameters(), lr=learning_rate)
    selector_opt = torch.optim.Adam(selector.parameters(), lr=learning_rate)
    classification_loss = nn.CrossEntropyLoss()
    pending = {"log_prob": None, "entropy": None, "logits": None}
    loss_baseline = {"value": None}

    def score_with_mask(x_t, mask_t):
        predictor.eval()
        return predictor(mask_layer(x_t, mask_t))

    def acquire_and_predict(x_t, budget_state):
        selector.train()
        x_masked, mask_t, cost_t, log_probs, entropies = (
            gdfs.sample_features(
                x_t,
                budget=train_budget,
                global_budget_state=budget_state,
            )
        )
        predictor.eval()
        with torch.no_grad():
            logits_t = predictor(x_masked)
        pending.update(
            log_prob=log_probs.mean(),
            entropy=entropies.mean(),
            logits=logits_t.detach(),
        )
        return logits_t, mask_t, float(cost_t)

    def update(x_t, y_t, mask_t):
        pre_update_loss = None
        if pending["logits"] is not None:
            pre_update_loss = float(
                classification_loss(pending["logits"], y_t).item()
            )
        for _ in range(updates_per_sample):
            predictor.train()
            predictor_opt.zero_grad()
            logits = predictor(mask_layer(x_t, mask_t))
            loss = classification_loss(logits, y_t)
            loss.backward()
            predictor_opt.step()

        if (
            pending["log_prob"] is not None
            and pending["log_prob"].requires_grad
            and pre_update_loss is not None
        ):
            baseline = (
                pre_update_loss if loss_baseline["value"] is None
                else loss_baseline["value"]
            )
            advantage = float(pre_update_loss - baseline)
            selector_opt.zero_grad()
            policy_loss = advantage * pending["log_prob"]
            if pending["entropy"] is not None:
                policy_loss = policy_loss - entropy_weight * pending["entropy"]
            policy_loss.backward()
            selector_opt.step()
            if loss_baseline["value"] is None:
                loss_baseline["value"] = pre_update_loss
            else:
                loss_baseline["value"] = (
                    0.9 * loss_baseline["value"] + 0.1 * pre_update_loss
                )
        pending.update(log_prob=None, entropy=None, logits=None)

    return OnlineModel(
        acquire_and_predict, score_with_mask, update, predictor, gdfs
    )


def build_online_jafa(
    ctx,
    device,
    *,
    train_budget,
    learning_rate=1e-3,
    updates_per_sample=1,
    embedding_size=16,
    hidden_size=32,
    memory_size=16,
    processing_steps=5,
    acquisition_cost_weight=0.05,
    epsilon_start=1.0,
    epsilon_end=0.1,
    max_candidates=512,
    exact_max_paid=12,
    linear_classifier=True,
    seed=42,
) -> OnlineModel:
    """Construct a causal one-step-Q JAFA with label-after-predict updates."""
    if updates_per_sample < 1:
        raise ValueError("updates_per_sample must be at least 1")
    if not 0 <= epsilon_end <= epsilon_start <= 1:
        raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1")

    predictor = JAFAPredictor(
        ctx["mask_layer"],
        ctx["d_out"],
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        memory_size=memory_size,
        processing_steps=processing_steps,
        linear_classifier=linear_classifier,
    ).to(device)
    jafa = JAFAOneShot(
        predictor,
        ctx["mask_layer"],
        ctx["feature_costs"],
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        acquisition_cost_weight=acquisition_cost_weight,
        max_candidates=max_candidates,
        exact_max_paid=exact_max_paid,
        seed=seed,
    ).to(device)
    predictor_optimizer = torch.optim.Adam(
        predictor.parameters(), lr=learning_rate
    )
    q_optimizer = torch.optim.Adam(
        jafa.q_network.parameters(), lr=learning_rate
    )
    n_train = max(len(ctx["train_dataset"]), 1)
    epsilon_state = {"step": 0}
    pending = {"logits": None, "mask": None}

    def current_epsilon():
        fraction = min(epsilon_state["step"] / max(n_train - 1, 1), 1.0)
        return epsilon_start + fraction * (epsilon_end - epsilon_start)

    def action_indices_for_masks(masks):
        indices = []
        for mask in masks.bool():
            matches = (jafa.candidate_masks == mask).all(dim=1)
            if not bool(matches.any()):
                raise RuntimeError("executed JAFA mask is not a candidate action")
            indices.append(torch.nonzero(matches, as_tuple=False)[0, 0])
        return torch.stack(indices)

    def score_with_mask(x_t, mask_t):
        predictor.eval()
        return predictor(ctx["mask_layer"](x_t, mask_t))

    def acquire_and_predict(x_t, budget_state):
        jafa.eval()
        with torch.no_grad():
            q_values = jafa.q_values(x_t)
            actions = jafa.choose_action_indices(
                q_values,
                budget=train_budget,
                omd_lambda=budget_state.omd_lambda,
                epsilon=current_epsilon(),
            )
            proposed = jafa.candidate_masks[actions].to(x_t.dtype)
            masks, _, cost = jafa._apply_global_budget(
                proposed, budget_state
            )
            logits = predictor(ctx["mask_layer"](x_t, masks))
        pending.update(logits=logits.detach(), mask=masks.detach())
        return logits, masks, float(cost)

    def update(x_t, y_t, mask_t):
        pre_update_loss = None
        if pending["logits"] is not None:
            pre_update_loss = F.cross_entropy(
                pending["logits"], y_t, reduction="none"
            ).detach()

        for _ in range(updates_per_sample):
            predictor.train()
            predictor_optimizer.zero_grad()
            logits = predictor(ctx["mask_layer"](x_t, mask_t))
            F.cross_entropy(logits, y_t).backward()
            predictor_optimizer.step()

        if pre_update_loss is not None:
            actions = action_indices_for_masks(mask_t)
            costs = jafa.candidate_costs[actions]
            target = -pre_update_loss - acquisition_cost_weight * costs
            for _ in range(updates_per_sample):
                q_optimizer.zero_grad()
                q_values = jafa.q_values(x_t)
                chosen_q = q_values.gather(
                    1, actions.unsqueeze(1)
                ).squeeze(1)
                F.mse_loss(chosen_q, target).backward()
                q_optimizer.step()

        epsilon_state["step"] += 1
        pending.update(logits=None, mask=None)

    return OnlineModel(
        acquire_and_predict, score_with_mask, update, predictor, jafa
    )


def build_online_ol(
    ctx,
    device,
    *,
    train_budget,
    learning_rate=1e-3,
    updates_per_sample=1,
    hidden_sizes=(64, 32, 16),
    dropout=0.5,
    use_feature_mask=False,
    reward_method="Bayesian-L1",
    mcdrop_samples=100,
    epsilon_start=1.0,
    epsilon_end=0.1,
    max_candidates=512,
    exact_max_paid=12,
    seed=42,
) -> OnlineModel:
    """Construct causal one-step OL with label-after-prediction updates."""
    if updates_per_sample < 1:
        raise ValueError("updates_per_sample must be at least 1")
    if not 0 <= epsilon_end <= epsilon_start <= 1:
        raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1")

    ol = OLOneShot(
        ctx["mask_layer"],
        ctx["feature_costs"],
        ctx["d_out"],
        hidden_sizes=hidden_sizes,
        dropout=dropout,
        use_feature_mask=use_feature_mask,
        reward_method=reward_method,
        mcdrop_samples=mcdrop_samples,
        max_candidates=max_candidates,
        exact_max_paid=exact_max_paid,
        seed=seed,
    ).to(device)
    predictor = ol.predictor
    p_optimizer = torch.optim.Adam(
        predictor.p_parameters(), lr=learning_rate
    )
    q_optimizer = torch.optim.Adam(
        predictor.q_parameters(), lr=learning_rate
    )
    n_train = max(len(ctx["train_dataset"]), 1)
    epsilon_state = {"step": 0}
    pending = {"action": None, "reward": None}

    def current_epsilon():
        fraction = min(epsilon_state["step"] / max(n_train - 1, 1), 1.0)
        return epsilon_start + fraction * (epsilon_end - epsilon_start)

    def score_with_mask(x_t, mask_t):
        predictor.eval()
        return predictor(ctx["mask_layer"](x_t, mask_t))

    def acquire_and_predict(x_t, budget_state):
        ol.eval()
        with torch.no_grad():
            q_values = ol.q_values(x_t)
            actions = ol.choose_action_indices(
                q_values,
                budget=train_budget,
                omd_lambda=budget_state.omd_lambda,
                epsilon=current_epsilon(),
            )
            proposed = ol.candidate_masks[actions].to(x_t.dtype)
            masks, actions, cost = ol.apply_global_budget(
                proposed, budget_state
            )
            logits = predictor(ctx["mask_layer"](x_t, masks))
            reward = ol.rewards_for_masks(x_t, masks)
        pending.update(action=actions.detach(), reward=reward.detach())
        return logits, masks, float(cost)

    def update(x_t, y_t, mask_t):
        actions = pending["action"]
        reward = pending["reward"]
        if actions is not None and reward is not None:
            for _ in range(updates_per_sample):
                q_optimizer.zero_grad()
                q_values = ol.q_values(x_t)
                chosen_q = q_values.gather(
                    1, actions.unsqueeze(1)
                ).squeeze(1)
                F.mse_loss(chosen_q, reward).backward()
                q_optimizer.step()

        for _ in range(updates_per_sample):
            predictor.train()
            p_optimizer.zero_grad()
            logits = predictor(ctx["mask_layer"](x_t, mask_t))
            F.cross_entropy(logits, y_t).backward()
            p_optimizer.step()

        epsilon_state["step"] += 1
        pending.update(action=None, reward=None)

    return OnlineModel(
        acquire_and_predict, score_with_mask, update, predictor, ol
    )


def build_online_cwcf(
    ctx,
    device,
    *,
    train_budget,
    learning_rate=5e-4,
    updates_per_sample=1,
    hidden_size=128,
    hidden_layers=3,
    cost_weight=1.0,
    epsilon_start=1.0,
    epsilon_end=0.1,
    max_candidates=512,
    exact_max_paid=12,
    grad_clip=1.0,
    seed=42,
) -> OnlineModel:
    """Construct causal CwCF: select, predict, reveal label, update Q."""
    if updates_per_sample < 1:
        raise ValueError("updates_per_sample must be at least 1")
    if not 0 <= epsilon_end <= epsilon_start <= 1:
        raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1")
    cwcf = CwCFOneShot(
        ctx["mask_layer"], ctx["feature_costs"], ctx["d_out"],
        hidden_size=hidden_size, hidden_layers=hidden_layers,
        cost_weight=cost_weight, max_candidates=max_candidates,
        exact_max_paid=exact_max_paid, seed=seed,
    ).to(device)
    optimizer = torch.optim.Adam(cwcf.parameters(), lr=learning_rate)
    n_train = max(len(ctx["train_dataset"]), 1)
    state = {"step": 0, "action": None, "prediction": None}

    def epsilon():
        fraction = min(state["step"] / max(n_train - 1, 1), 1.0)
        return epsilon_start + fraction * (epsilon_end - epsilon_start)

    def score_with_mask(x_t, mask_t):
        cwcf.eval()
        return cwcf.predictor(ctx["mask_layer"](x_t, mask_t))

    def acquire_and_predict(x_t, budget_state):
        cwcf.eval()
        with torch.no_grad():
            subset_q = cwcf.subset_q_values(x_t)
            proposed_action = cwcf.choose_action_indices(
                subset_q, budget=train_budget,
                omd_lambda=budget_state.omd_lambda, epsilon=epsilon(),
            )
            masks, actions, cost = cwcf.apply_global_budget(
                cwcf.candidate_masks[proposed_action].to(x_t.dtype),
                budget_state,
            )
            logits = cwcf.predictor(ctx["mask_layer"](x_t, masks))
        state.update(action=actions.detach(), prediction=logits.argmax(1).detach())
        return logits, masks, float(cost)

    def update(x_t, y_t, mask_t):
        # During the shared K-row initialization there is no subset action,
        # but terminal class actions can still be initialized causally.
        for _ in range(updates_per_sample):
            cwcf.train()
            optimizer.zero_grad()
            class_q = cwcf.predictor(ctx["mask_layer"](x_t, mask_t))
            target = cwcf.classification_targets(
                y_t, cwcf.num_classes, dtype=class_q.dtype
            )
            loss = F.mse_loss(class_q, target)
            if state["action"] is not None:
                subset_q = cwcf.subset_q_values(x_t)
                chosen_q = subset_q.gather(1, state["action"][:, None]).squeeze(1)
                cost = (mask_t * cwcf.feature_costs).sum(1)
                terminal_return = -(state["prediction"] != y_t).to(class_q.dtype)
                terminal_return = terminal_return - cwcf.cost_weight * cost
                loss = loss + F.mse_loss(chosen_q, terminal_return)
            loss.backward()
            nn.utils.clip_grad_norm_(cwcf.parameters(), grad_clip)
            optimizer.step()
        state.update(step=state["step"] + 1, action=None, prediction=None)

    return OnlineModel(
        acquire_and_predict, score_with_mask, update, cwcf.predictor, cwcf
    )


def build_online_pt(
    ctx,
    device,
    *,
    train_budget,
    learning_rate=1e-3,
    updates_per_sample=1,
    importance_decay=0.9,
    linear_classifier=True,
    seed=42,
) -> OnlineModel:
    """Construct causal PT with streaming past-value permutation scores."""
    if updates_per_sample < 1:
        raise ValueError("updates_per_sample must be at least 1")
    classifier_fn = get_linear_network if linear_classifier else get_mlp_network
    d_in = ctx["d_in"]
    d_out = ctx["d_out"]
    nviews = ctx["num_modalities"]
    mask_layer = ctx["mask_layer"]
    predictor = classifier_fn(d_in + nviews, d_out).to(device)
    pt = PTOneShot(
        predictor,
        mask_layer,
        ctx["feature_costs"],
        importance_decay=importance_decay,
        seed=seed,
    ).to(device)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=learning_rate)
    classification_loss = nn.CrossEntropyLoss()

    def score_with_mask(x_t, mask_t):
        predictor.eval()
        return predictor(mask_layer(x_t, mask_t))

    def acquire_and_predict(x_t, budget_state):
        predictor.eval()
        with torch.no_grad():
            x_masked, mask_t, cost_t = pt.select_features(
                x_t,
                budget=train_budget,
                global_budget_state=budget_state,
            )
            logits_t = predictor(x_masked)
        return logits_t, mask_t, float(cost_t)

    def update(x_t, y_t, mask_t):
        pt.update_causal_importance(x_t, y_t, mask_t)
        for _ in range(updates_per_sample):
            predictor.train()
            optimizer.zero_grad()
            logits = predictor(mask_layer(x_t, mask_t))
            loss = classification_loss(logits, y_t)
            loss.backward()
            optimizer.step()

    return OnlineModel(
        acquire_and_predict, score_with_mask, update, predictor, pt
    )


def build_online_eddi(
    ctx,
    device,
    *,
    learning_rate=1e-3,
    updates_per_sample=1,
    cost_normalized=True,
    pvae_samples=128,
    linear_classifier=True,
) -> OnlineModel:
    """Construct a one-pass EDDI model with per-observation SGD updates."""
    if updates_per_sample < 1:
        raise ValueError("updates_per_sample must be at least 1")

    classifier_fn = get_linear_network if linear_classifier else get_mlp_network
    d_in = ctx["d_in"]
    d_out = ctx["d_out"]
    nviews = ctx["num_modalities"]
    mask_layer = ctx["mask_layer"]

    encoder = get_mlp_network(d_in + nviews, 32)
    decoder = get_mlp_network(16, d_in)
    sampler = pvae.PVAE(
        encoder, decoder, mask_layer, num_samples=pvae_samples,
        decoder_distribution="gaussian",
    ).to(device)
    predictor = classifier_fn(d_in + nviews, d_out).to(device)
    eddi = EDDI(
        sampler, predictor, mask_layer, task="classification",
        feature_costs=ctx["feature_costs"], cost_normalized=cost_normalized,
    ).to(device)

    sampler_opt = torch.optim.Adam(sampler.parameters(), lr=learning_rate)
    predictor_opt = torch.optim.Adam(predictor.parameters(), lr=learning_rate)
    classification_loss = nn.CrossEntropyLoss()

    def score_with_mask(x_t, mask_t):
        predictor.eval()
        return predictor(mask_layer(x_t, mask_t))

    def acquire_and_predict(x_t, budget_state):
        sampler.eval()
        predictor.eval()
        with torch.no_grad():
            x_masked, mask, cost = eddi.select_features(
                x_t, budget=None, verbose=False,
                global_budget_state=budget_state,
            )
            logits = predictor(x_masked)
        return logits, mask, float(cost)

    def update(x_t, y_t, mask_t):
        for _ in range(updates_per_sample):
            sampler.train()
            sampler_opt.zero_grad()
            sampler_loss = sampler.loss(x_t, mask_t).mean()
            sampler_loss.backward()
            sampler_opt.step()

            predictor.train()
            predictor_opt.zero_grad()
            logits = predictor(mask_layer(x_t, mask_t))
            loss = classification_loss(logits, y_t)
            loss.backward()
            predictor_opt.step()

    return OnlineModel(acquire_and_predict, score_with_mask, update, predictor, eddi)


def build_online_dime(
    ctx,
    device,
    *,
    learning_rate=1e-3,
    updates_per_sample=1,
    cmi_scaling="bounded",
    linear_classifier=True,
) -> OnlineModel:
    """Construct a one-pass DIME model with causal predictor/CMI updates."""
    if updates_per_sample < 1:
        raise ValueError("updates_per_sample must be at least 1")
    if cmi_scaling not in ("bounded", "positive", "none"):
        raise ValueError("cmi_scaling must be bounded, positive, or none")

    classifier_fn = get_linear_network if linear_classifier else get_mlp_network
    d_in = ctx["d_in"]
    d_out = ctx["d_out"]
    nviews = ctx["num_modalities"]
    mask_layer = ctx["mask_layer"]
    predictor = classifier_fn(d_in + nviews, d_out).to(device)
    value_network = get_mlp_network(d_in + nviews, nviews).to(device)

    # DIMEOneShot intentionally uses a duck-typed estimator container.
    estimator = SimpleNamespace(
        mask_layer=mask_layer,
        value_network=value_network,
        predictor=predictor,
        cmi_scaling=cmi_scaling,
    )
    dime = DIMEOneShot(estimator, feature_costs=ctx["feature_costs"]).to(device)
    optimizer = torch.optim.Adam(
        list(predictor.parameters()) + list(value_network.parameters()),
        lr=learning_rate,
    )
    costs = torch.as_tensor(ctx["feature_costs"], dtype=torch.float32, device=device)

    def score_with_mask(x_t, mask_t):
        predictor.eval()
        return predictor(mask_layer(x_t, mask_t))

    def acquire_and_predict(x_t, budget_state):
        predictor.eval()
        value_network.eval()
        with torch.no_grad():
            x_masked, mask, cost = dime.select_features(
                x_t, budget=None, verbose=False,
                global_budget_state=budget_state,
            )
            logits = predictor(x_masked)
        return logits, mask, float(cost)

    def update(x_t, y_t, acquired_mask):
        for _ in range(updates_per_sample):
            predictor.train()
            value_network.train()
            optimizer.zero_grad()

            free_mask = torch.zeros_like(acquired_mask)
            free_mask[:, 0] = 1.0
            x_free = mask_layer(x_t, free_mask)
            pred0 = predictor(x_free)
            if cmi_scaling == "bounded":
                pred_cmi = value_network(x_free).sigmoid() * get_entropy(pred0).unsqueeze(1)
            elif cmi_scaling == "positive":
                pred_cmi = F.softplus(value_network(x_free))
            else:
                pred_cmi = value_network(x_free)

            scores = pred_cmi.detach()[0] / torch.clamp(costs, min=1e-12)
            scores[0] = -torch.inf
            selected = torch.nonzero(
                acquired_mask[0] > 0, as_tuple=False
            ).flatten().tolist()
            selected = [j for j in selected if j != 0]
            selected.sort(key=lambda j: float(scores[j]), reverse=True)

            current_mask = free_mask
            loss_prev = F.cross_entropy(pred0, y_t, reduction="none")
            predictor_losses = [loss_prev.mean()]
            value_losses = []
            for j in selected:
                current_mask = current_mask.clone()
                current_mask[:, j] = 1.0
                pred_after = predictor(mask_layer(x_t, current_mask))
                loss_after = F.cross_entropy(pred_after, y_t, reduction="none")
                target_delta = (loss_prev.detach() - loss_after.detach())
                value_losses.append(F.mse_loss(pred_cmi[:, j], target_delta))
                predictor_losses.append(loss_after.mean())
                loss_prev = loss_after

            total_loss = torch.stack(predictor_losses).mean()
            if value_losses:
                total_loss = total_loss + torch.stack(value_losses).mean()
            total_loss.backward()
            optimizer.step()

    return OnlineModel(acquire_and_predict, score_with_mask, update, predictor, dime)
