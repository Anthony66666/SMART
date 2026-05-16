from typing import Mapping, Optional

import torch
import torch.nn.functional as F


def _get_weight(config: Optional[Mapping], name: str, default: float = 0.0) -> float:
    if config is None:
        return default
    if hasattr(config, name):
        return float(getattr(config, name))
    if isinstance(config, Mapping) and name in config:
        return float(config[name])
    return default


def _masked_mean(values: torch.Tensor,
                 mask: torch.Tensor,
                 weights: Optional[torch.Tensor] = None,
                 eps: float = 1e-6) -> torch.Tensor:
    mask_f = mask.to(values.dtype)
    while mask_f.dim() < values.dim():
        mask_f = mask_f.unsqueeze(-1)
    if weights is not None:
        weight_f = weights.to(values.dtype)
        while weight_f.dim() < values.dim():
            weight_f = weight_f.unsqueeze(-1)
        mask_f = mask_f * weight_f
    denom = mask_f.sum().clamp_min(eps)
    return (values * mask_f).sum() / denom


def masked_kl_div(student_logits: torch.Tensor,
                  teacher_logits: torch.Tensor,
                  mask: torch.Tensor,
                  temperature: float,
                  weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """KL(p_teacher || p_student) over valid token positions."""
    if student_logits.numel() == 0 or not mask.any():
        return student_logits.new_zeros(())
    temp = max(float(temperature), 1e-6)
    student_log_prob = F.log_softmax(student_logits / temp, dim=-1)
    teacher_prob = F.softmax(teacher_logits.detach() / temp, dim=-1)
    kl = F.kl_div(student_log_prob, teacher_prob, reduction='none').sum(dim=-1)
    return _masked_mean(kl * (temp ** 2), mask, weights=weights)


def masked_entropy(logits: torch.Tensor,
                   mask: torch.Tensor,
                   weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mean categorical entropy over valid token positions."""
    if logits.numel() == 0 or not mask.any():
        return logits.new_zeros(())
    prob = F.softmax(logits, dim=-1)
    log_prob = F.log_softmax(logits, dim=-1)
    entropy = -(prob * log_prob).sum(dim=-1)
    return _masked_mean(entropy, mask, weights=weights)


def catk_recovery_targets(candidate_idx: torch.Tensor,
                          candidate_trajs: torch.Tensor,
                          gt_traj: torch.Tensor,
                          valid_mask: torch.Tensor):
    """Select the closest candidate token to GT for each rollout step.

    Args:
        candidate_idx: [R, A, S, K] token ids.
        candidate_trajs: [R, A, S, K, T, 2] candidate centers for each step.
        gt_traj: [A, F, 2] future ground truth centers.
        valid_mask: [A, F] future validity.
    """
    rollout_count, num_agent, num_steps, topk = candidate_idx.shape
    step_len = candidate_trajs.size(-2)
    device = candidate_idx.device
    targets = candidate_idx.new_zeros((rollout_count, num_agent, num_steps))
    masks = torch.zeros((rollout_count, num_agent, num_steps), dtype=torch.bool, device=device)
    distances = candidate_trajs.new_zeros((rollout_count, num_agent, num_steps, topk))

    for step in range(num_steps):
        start = step * step_len
        end = start + step_len
        if start >= gt_traj.size(1):
            continue
        gt_step = gt_traj[:, start:min(end, gt_traj.size(1))]
        valid_step = valid_mask[:, start:min(end, valid_mask.size(1))]
        if gt_step.size(1) < step_len:
            pad_len = step_len - gt_step.size(1)
            gt_step = F.pad(gt_step, (0, 0, 0, pad_len))
            valid_step = F.pad(valid_step, (0, pad_len), value=False)

        diff = candidate_trajs[:, :, step] - gt_step[None, :, None]
        dist = torch.norm(diff, p=2, dim=-1)
        valid_f = valid_step[None, :, None].to(dist.dtype)
        denom = valid_f.sum(dim=-1).clamp_min(1.0)
        dist = (dist * valid_f).sum(dim=-1) / denom
        best = dist.argmin(dim=-1)
        targets[:, :, step] = torch.gather(candidate_idx[:, :, step], -1, best.unsqueeze(-1)).squeeze(-1)
        masks[:, :, step] = valid_step.any(dim=-1)[None, :]
        distances[:, :, step] = dist

    return targets, masks, distances


def rollout_quality_weights(pred_traj: torch.Tensor,
                            pred_head: torch.Tensor,
                            gt: torch.Tensor,
                            valid_mask: torch.Tensor,
                            rewards_config: Optional[Mapping] = None,
                            agent_batch: Optional[torch.Tensor] = None):
    """Compute per-rollout quality rewards from rollout tensors.

    Returns:
        rewards: [R, A] larger is better.
        components: scalar diagnostics.
    """
    del agent_batch  # Batch-aware collision masking can be added when batch_size > 1 is needed.
    rollout_count, num_agent, horizon, _ = pred_traj.shape
    gt = gt[:, :horizon]
    valid_mask = valid_mask[:, :horizon]
    if gt.size(1) < horizon:
        pad_len = horizon - gt.size(1)
        gt = F.pad(gt, (0, 0, 0, pad_len))
        valid_mask = F.pad(valid_mask, (0, pad_len), value=False)

    valid_f = valid_mask[None].to(pred_traj.dtype)
    denom = valid_f.sum(dim=-1).clamp_min(1.0)
    ade = (torch.norm(pred_traj - gt[None], p=2, dim=-1) * valid_f).sum(dim=-1) / denom

    collision_penalty = pred_traj.new_zeros((rollout_count, num_agent))
    if num_agent > 1 and _get_weight(rewards_config, 'collision') != 0.0:
        pair_valid = valid_mask[None, :, None] & valid_mask[None, None, :]
        eye = torch.eye(num_agent, dtype=torch.bool, device=pred_traj.device)[None, None]
        pair_valid = pair_valid.permute(0, 3, 1, 2) & ~eye
        rel = pred_traj[:, :, None] - pred_traj[:, None]
        dist = torch.norm(rel, p=2, dim=-1).permute(0, 3, 1, 2)
        close = torch.relu(2.0 - dist) / 2.0
        close = close * pair_valid.to(close.dtype)
        collision_penalty = close.sum(dim=(1, 3)) / pair_valid.to(close.dtype).sum(dim=(1, 3)).clamp_min(1.0)

    kinematic_penalty = pred_traj.new_zeros((rollout_count, num_agent))
    if horizon > 2 and _get_weight(rewards_config, 'kinematic') != 0.0:
        speed = torch.norm(pred_traj[:, :, 1:] - pred_traj[:, :, :-1], p=2, dim=-1)
        accel = torch.abs(speed[:, :, 1:] - speed[:, :, :-1])
        yaw_delta = torch.atan2(
            torch.sin(pred_head[:, :, 1:] - pred_head[:, :, :-1]),
            torch.cos(pred_head[:, :, 1:] - pred_head[:, :, :-1])
        ).abs()
        kinematic_penalty = accel.mean(dim=-1) + 0.2 * yaw_delta.mean(dim=-1)

    diversity_reward = pred_traj.new_zeros((rollout_count, num_agent))
    if rollout_count > 1 and _get_weight(rewards_config, 'diversity') != 0.0:
        endpoints = pred_traj[:, :, -1]
        center = endpoints.mean(dim=0, keepdim=True)
        diversity_reward = torch.norm(endpoints - center, p=2, dim=-1)

    reward = (
        -_get_weight(rewards_config, 'gt_ade', 1.0) * ade
        -_get_weight(rewards_config, 'collision') * collision_penalty
        -_get_weight(rewards_config, 'kinematic') * kinematic_penalty
        +_get_weight(rewards_config, 'diversity') * diversity_reward
    )
    components = {
        'ade': ade.detach().mean(),
        'collision': collision_penalty.detach().mean(),
        'kinematic': kinematic_penalty.detach().mean(),
        'diversity': diversity_reward.detach().mean(),
        'reward': reward.detach().mean(),
    }
    return reward, components


def aggregate_rollout_weights(rewards: torch.Tensor,
                              mask: torch.Tensor,
                              temperature: float = 1.0) -> torch.Tensor:
    """Normalize rollout rewards across rollouts for each agent and step."""
    temp = max(float(temperature), 1e-6)
    while rewards.dim() < mask.dim():
        rewards = rewards.unsqueeze(-1)
    rewards = rewards.expand_as(mask).to(torch.float)
    masked_rewards = rewards.masked_fill(~mask, -1e9)
    weights = torch.softmax(masked_rewards / temp, dim=0)
    weights = weights * mask.to(weights.dtype)
    denom = weights.sum(dim=0, keepdim=True).clamp_min(1e-6)
    return weights / denom
