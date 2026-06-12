import argparse
import copy
import csv
import json
import math
import numbers
import random
import sys
import time
from pathlib import Path

import torch
from torch_geometric.data import Batch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from smart.datasets.scalable_dataset import MultiDataset
from smart.model import SMARTCausalDiffusion
from smart.transforms import WaymoTargetBuilder
from smart.utils.config import load_config_act
from smart.utils.torch_compat import register_checkpoint_safe_globals


def _as_list(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _load_dataset(config, raw_dir):
    data_config = config.Dataset
    raw_paths = _as_list(raw_dir) if raw_dir else data_config.val_raw_dir
    return MultiDataset(
        root=data_config.root,
        split="val",
        raw_dir=raw_paths,
        processed_dir=getattr(data_config, "val_processed_dir", None),
        transform=WaymoTargetBuilder(
            config.Model.num_historical_steps,
            config.Model.decoder.num_future_steps,
        ),
        token_size=int(
            getattr(
                data_config,
                "token_size",
                getattr(config.Model.decoder, "token_size", 512),
            )
        ),
    )


def _load_model(config, ckpt_path, device):
    register_checkpoint_safe_globals()
    model = SMARTCausalDiffusion(config.Model)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[load_model] missing keys: {len(missing)}")
    if unexpected:
        print(f"[load_model] unexpected keys: {len(unexpected)}")
    model.debug_validation_logging = False
    model.eval()
    model.to(device)
    return model, {"missing": len(missing), "unexpected": len(unexpected)}


def _make_batch(dataset, index, device):
    graph = dataset[int(index)]
    return Batch.from_data_list([graph]).to(device)


def _scenario_id(batch):
    scenario_id = getattr(batch, "scenario_id", None)
    if scenario_id is None and hasattr(batch, "get"):
        scenario_id = batch.get("scenario_id", None)
    if scenario_id is None:
        return "unknown"
    return str(scenario_id)


def _parse_ints(value):
    if value is None or value.strip() == "":
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_indices(index, indices, num_scenes, dataset_len):
    if indices is not None and str(indices).strip():
        requested = _parse_ints(str(indices))
    else:
        start = max(0, int(index))
        end = min(int(dataset_len), start + max(1, int(num_scenes)))
        requested = list(range(start, end))
    seen = set()
    bounded = []
    for value in requested:
        value = int(value)
        if value in seen or value < 0 or value >= int(dataset_len):
            continue
        seen.add(value)
        bounded.append(value)
    return bounded


def _ego_agent_indices(batch):
    agent = batch["agent"]
    value = None
    if "ego_agent_id" in agent:
        value = agent["ego_agent_id"]
    elif "av_index" in agent:
        value = agent["av_index"]
    if value is None:
        return set()
    if not torch.is_tensor(value):
        return {int(value)}
    return {
        int(item)
        for item in value.detach().cpu().reshape(-1).tolist()
        if int(item) >= 0
    }


def _select_target_agents(batch, model, explicit_agents, max_targets):
    current_step = int(model.num_historical_steps) - 1
    current_valid = batch["agent"]["valid_mask"][:, current_step].bool()
    agent_type = batch["agent"]["type"].long()
    candidates = current_valid & (agent_type != 3)
    ego_indices = _ego_agent_indices(batch)
    if ego_indices:
        ego_mask = torch.zeros_like(current_valid, dtype=torch.bool)
        for ego_idx in ego_indices:
            if 0 <= int(ego_idx) < int(ego_mask.shape[0]):
                ego_mask[int(ego_idx)] = True
        candidates = candidates & ~ego_mask
    if "category" in batch["agent"]:
        category3 = candidates & (batch["agent"]["category"].long() == 3)
        if category3.any():
            candidates = category3
    if explicit_agents:
        target_mask = torch.zeros_like(current_valid, dtype=torch.bool)
        for agent_idx in explicit_agents:
            if 0 <= int(agent_idx) < int(target_mask.shape[0]):
                target_mask[int(agent_idx)] = True
        if ego_indices:
            for ego_idx in ego_indices:
                if 0 <= int(ego_idx) < int(target_mask.shape[0]):
                    target_mask[int(ego_idx)] = False
        target_mask = target_mask & current_valid
    else:
        indices = torch.nonzero(candidates, as_tuple=False).squeeze(-1)
        if indices.numel() == 0:
            indices = torch.nonzero(current_valid, as_tuple=False).squeeze(-1)
        target_mask = torch.zeros_like(current_valid, dtype=torch.bool)
        if indices.numel() > 0:
            target_mask[indices[: int(max_targets)]] = True
    batch["agent"]["target_agents"] = target_mask
    return torch.nonzero(target_mask, as_tuple=False).squeeze(-1).tolist()


def _apply_guidance_overrides(model, args):
    mapping = {
        "target_spec": ("guidance_target_spec", str),
        "ego_interaction_alpha": ("guidance_ego_interaction_alpha", float),
        "target_event_eta": ("guidance_target_event_eta", float),
        "edit_gamma": ("guidance_edit_gamma", float),
        "invalid_beta": ("guidance_invalid_beta", float),
        "path_corridor_width": ("guidance_path_corridor_width", float),
        "conflict_tta_threshold": ("guidance_conflict_tta_threshold", float),
        "near_miss_distance": ("guidance_near_miss_distance", float),
        "ttc_threshold": ("guidance_ttc_threshold", float),
    }
    for arg_name, (attribute, caster) in mapping.items():
        if not hasattr(args, arg_name):
            continue
        value = getattr(args, arg_name)
        if value is None or value == "":
            continue
        setattr(model, attribute, caster(value))


def _attach_seed_controls(batch, model, target_agents, target_token_window):
    future_start = int(model.num_historical_steps)
    future_end = future_start + int(model.ar_total_rollout_steps)
    batch["agent"]["seed_token_ids"] = batch["agent"]["token_idx"].clone()
    batch["agent"]["seed_trajs"] = batch["agent"]["position"][
        :, future_start:future_end, :2
    ].float().clone()
    if target_token_window is not None:
        token_count = int(batch["agent"]["token_idx"].shape[1])
        edit_mask = torch.zeros(
            batch["agent"]["token_idx"].shape,
            dtype=torch.bool,
            device=batch["agent"]["token_idx"].device,
        )
        start, end = target_token_window
        start = int(model.ar_history_tokens) + max(0, int(start))
        end = int(model.ar_history_tokens) + max(0, int(end))
        end = min(token_count, end)
        if end > start and target_agents:
            edit_mask[
                torch.as_tensor(target_agents, device=edit_mask.device),
                start:end,
            ] = True
        batch["agent"]["edit_mask"] = edit_mask


def _scalar_dict(values):
    result = {}
    for key, value in values.items():
        if torch.is_tensor(value):
            result[key] = float(value.detach().cpu().item())
        else:
            result[key] = float(value)
    return result


def _token_change_rate(pred_tokens, gt_tokens, token_valid):
    token_valid = token_valid.bool()
    if not token_valid.any():
        return 0.0
    token_changed = (pred_tokens != gt_tokens) & token_valid
    return float(token_changed[token_valid].float().mean().detach().cpu().item())


def _trajectory_metrics(output, target_agents=None):
    pred = output["pred_traj"]
    target = output["gt"]
    valid = output.get("valid_mask", output.get("official_valid_mask")).bool()
    official_valid = output.get("official_valid_mask", valid).bool()
    pred_valid = output["pred_valid_mask"].bool()
    eval_valid = valid & pred_valid
    distances = torch.norm(pred - target, dim=-1)
    if eval_valid.any():
        ade = distances[eval_valid].mean()
        step_ids = torch.arange(
            pred.shape[1],
            device=pred.device,
        ).unsqueeze(0).expand_as(eval_valid)
        last_index = step_ids.masked_fill(~eval_valid, -1).max(dim=1).values
        has_valid = last_index >= 0
        rows = torch.nonzero(has_valid, as_tuple=False).squeeze(-1)
        fde = distances[rows, last_index[has_valid]].mean()
    else:
        ade = pred.sum() * 0.0
        fde = pred.sum() * 0.0
    non_target_mask = torch.ones(
        pred.shape[0],
        dtype=torch.bool,
        device=pred.device,
    )
    if target_agents:
        for agent_idx in target_agents:
            if 0 <= int(agent_idx) < int(non_target_mask.shape[0]):
                non_target_mask[int(agent_idx)] = False
    non_target_valid = official_valid & pred_valid & non_target_mask.unsqueeze(-1)
    if non_target_valid.any():
        non_target_preservation_ade = distances[non_target_valid].mean()
    else:
        non_target_preservation_ade = pred.sum() * 0.0
    pair_valid = eval_valid[:, 1:] & eval_valid[:, :-1]
    pred_speed = torch.norm(pred[:, 1:] - pred[:, :-1], dim=-1) / 0.1
    gt_speed = torch.norm(target[:, 1:] - target[:, :-1], dim=-1) / 0.1
    if pair_valid.any():
        pred_speed_mean = pred_speed[pair_valid].mean()
        gt_speed_mean = gt_speed[pair_valid].mean()
    else:
        pred_speed_mean = pred.sum() * 0.0
        gt_speed_mean = pred.sum() * 0.0
    moving_valid = pair_valid & (gt_speed > 1.0)
    if moving_valid.any():
        speed_ratio = pred_speed[moving_valid] / gt_speed[moving_valid].clamp_min(1e-3)
        moving_speed_ratio_mean = speed_ratio.mean()
        quantiles = torch.quantile(
            speed_ratio,
            speed_ratio.new_tensor([0.1, 0.5, 0.9]),
        )
    else:
        moving_speed_ratio_mean = pred.sum() * 0.0
        quantiles = pred.new_zeros(3)
    return {
        "ade": float(ade.detach().cpu().item()),
        "fde": float(fde.detach().cpu().item()),
        "non_target_preservation_ADE": float(
            non_target_preservation_ade.detach().cpu().item()
        ),
        "pred_valid_frames": int(pred_valid.sum().detach().cpu().item()),
        "eval_valid_frames": int(eval_valid.sum().detach().cpu().item()),
        "pred_speed": float(pred_speed_mean.detach().cpu().item()),
        "gt_speed": float(gt_speed_mean.detach().cpu().item()),
        "moving_pair_count": int(moving_valid.sum().detach().cpu().item()),
        "moving_speed_ratio": float(
            moving_speed_ratio_mean.detach().cpu().item()
        ),
        "moving_speed_ratio_p10": float(quantiles[0].detach().cpu().item()),
        "moving_speed_ratio_p50": float(quantiles[1].detach().cpu().item()),
        "moving_speed_ratio_p90": float(quantiles[2].detach().cpu().item()),
    }


def _zero_guidance_metrics():
    return {
        "collision": 0.0,
        "dynamics": 0.0,
        "dynamics_energy": 0.0,
        "edit_distance": 0.0,
        "ego_conflict_tta_error": 0.0,
        "ego_min_distance": 0.0,
        "ego_min_ttc": 0.0,
        "ego_risk_min_ttc": 0.0,
        "ego_risk_reward": 0.0,
        "ego_risk_success_rate": 0.0,
        "ego_near_miss_success_rate": 0.0,
        "ego_path_intrusion_rate": 0.0,
        "ego_required_decel": 0.0,
        "hard_collision_rate": 0.0,
        "lane_distance": 0.0,
        "lane_heading": 0.0,
        "min_ttc": 0.0,
        "near_miss_rate": 0.0,
        "offroad_rate": 0.0,
        "target_event_success_rate": 0.0,
        "target_success_rate": 0.0,
    }


def _flatten_record(record):
    flat = {}
    for key, value in record.items():
        if key == "guidance_metrics":
            for metric_key, metric_value in value.items():
                flat[f"guidance_{metric_key}"] = metric_value
        else:
            flat[key] = value
    return flat


def _is_number(value):
    return isinstance(value, numbers.Number) and not isinstance(value, bool)


def _mean(values):
    finite = [
        float(value)
        for value in values
        if _is_number(value) and math.isfinite(float(value))
    ]
    if not finite:
        return 0.0
    return sum(finite) / len(finite)


def _aggregate_results(records):
    flattened = [_flatten_record(record) for record in records]
    by_mode_records = {}
    for record in flattened:
        by_mode_records.setdefault(record.get("mode", "unknown"), []).append(record)
    by_mode = {}
    for mode, mode_records in by_mode_records.items():
        numeric_keys = sorted({
            key
            for record in mode_records
            for key, value in record.items()
            if _is_number(value)
            and key not in {"scene_index"}
        })
        summary = {"count": len(mode_records)}
        for key in numeric_keys:
            summary[f"{key}_mean"] = _mean(
                record.get(key)
                for record in mode_records
            )
        by_mode[mode] = summary
    return {
        "count": len(records),
        "by_mode": by_mode,
    }


def _pareto_objectives(mode_summary):
    return {
        "realism_ade": float(mode_summary.get("ade_mean", 0.0)),
        "invalidity": (
            float(mode_summary.get("guidance_hard_collision_rate_mean", 0.0))
            + float(mode_summary.get("guidance_offroad_rate_mean", 0.0))
            + float(mode_summary.get("guidance_dynamics_energy_mean", 0.0))
        ),
        "criticality": (
            float(mode_summary.get("guidance_ego_risk_success_rate_mean", 0.0))
            + 0.5 * float(mode_summary.get("guidance_ego_near_miss_success_rate_mean", 0.0))
        ),
        "minimality": (
            float(mode_summary.get("guidance_edit_distance_mean", 0.0))
            + float(mode_summary.get("token_change_rate_vs_gt_mean", 0.0))
            + float(mode_summary.get("non_target_preservation_ADE_mean", 0.0))
        ),
    }


def _dominates(left, right):
    left_obj = left["objectives"]
    right_obj = right["objectives"]
    minimize_keys = ("realism_ade", "invalidity", "minimality")
    maximize_keys = ("criticality",)
    no_worse = all(
        left_obj[key] <= right_obj[key]
        for key in minimize_keys
    ) and all(
        left_obj[key] >= right_obj[key]
        for key in maximize_keys
    )
    strictly_better = any(
        left_obj[key] < right_obj[key]
        for key in minimize_keys
    ) or any(
        left_obj[key] > right_obj[key]
        for key in maximize_keys
    )
    return no_worse and strictly_better


def _pareto_modes(summary, include_seed=False):
    entries = [
        {
            "mode": mode,
            "objectives": _pareto_objectives(mode_summary),
        }
        for mode, mode_summary in summary.get("by_mode", {}).items()
        if include_seed or mode != "seed"
    ]
    pareto = []
    for entry in entries:
        if any(
            _dominates(other, entry)
            for other in entries
            if other["mode"] != entry["mode"]
        ):
            continue
        pareto.append(entry)
    return sorted(pareto, key=lambda item: item["mode"])


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat_rows = [_flatten_record(row) for row in rows]
    if not flat_rows:
        path.write_text("", encoding="utf-8")
        return
    preferred = [
        "mode",
        "scene_index",
        "scenario_id",
        "target_agents",
        "target_time_window",
        "ade",
        "fde",
        "token_change_rate_vs_gt",
        "non_target_preservation_ADE",
        "guidance_ego_min_ttc",
        "guidance_ego_risk_min_ttc",
        "guidance_ego_min_distance",
        "guidance_ego_required_decel",
        "guidance_ego_risk_reward",
        "guidance_ego_risk_success_rate",
        "guidance_ego_path_intrusion_rate",
        "guidance_ego_conflict_tta_error",
        "guidance_ego_near_miss_success_rate",
        "guidance_target_event_success_rate",
        "guidance_hard_collision_rate",
        "guidance_offroad_rate",
        "guidance_dynamics_energy",
        "guidance_edit_distance",
        "elapsed_sec",
    ]
    all_keys = sorted({key for row in flat_rows for key in row})
    fieldnames = [
        key for key in preferred if key in all_keys
    ] + [
        key for key in all_keys if key not in preferred
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)


def _summary_rows(summary):
    rows = []
    for mode, values in sorted(summary.get("by_mode", {}).items()):
        row = {"mode": mode}
        row.update(values)
        rows.append(row)
    return rows


def _seed_baseline(model, dataset, index, device, args):
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))
    batch = _make_batch(dataset, index, device)
    target_agents = _select_target_agents(
        batch,
        model,
        _parse_ints(args.target_agents),
        args.max_target_agents,
    )
    future_start = int(model.num_historical_steps)
    future_end = future_start + int(model.ar_total_rollout_steps)
    gt = batch["agent"]["position"][:, future_start:future_end, :2].float()
    official_valid = batch["agent"]["valid_mask"][
        :, future_start:future_end
    ].bool()
    eval_valid = official_valid.clone()
    if "category" in batch["agent"]:
        eval_valid[batch["agent"]["category"].long() != 3] = False
    output = {
        "pred_traj": gt,
        "gt": gt,
        "valid_mask": eval_valid,
        "official_valid_mask": official_valid,
        "pred_valid_mask": official_valid,
    }
    metrics = _trajectory_metrics(output, target_agents=target_agents)
    metrics.update(
        {
            "mode": "seed",
            "scene_index": int(index),
            "scenario_id": _scenario_id(batch),
            "target_agents": target_agents,
            "target_time_window": args.target_time_window,
            "elapsed_sec": 0.0,
            "empty_output": False,
            "guidance_metrics": _zero_guidance_metrics(),
            "token_change_rate_vs_gt": 0.0,
        }
    )
    return metrics


def _run_mode(model, dataset, index, device, mode, args):
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))
    batch = _make_batch(dataset, index, device)
    target_agents = _select_target_agents(
        batch,
        model,
        _parse_ints(args.target_agents),
        args.max_target_agents,
    )
    target_window = None
    if args.target_time_window:
        parsed = _parse_ints(args.target_time_window)
        if len(parsed) != 2:
            raise ValueError("--target-time-window must contain exactly start,end.")
        target_window = tuple(parsed)
    model.guidance_mode = mode
    model.guidance_target_agents = ()
    model.guidance_target_time_window = target_window
    _apply_guidance_overrides(model, args)
    if mode in {"ego_stress", "ego_edit"}:
        _attach_seed_controls(batch, model, target_agents, target_window)
    start = time.perf_counter()
    with torch.no_grad():
        output = model.inference(batch)
    elapsed = time.perf_counter() - start
    if output is None:
        return {
            "mode": mode,
            "scene_index": int(index),
            "scenario_id": _scenario_id(batch),
            "target_agents": target_agents,
            "target_time_window": args.target_time_window,
            "elapsed_sec": elapsed,
            "empty_output": True,
        }
    metrics = _trajectory_metrics(output, target_agents=target_agents)
    metrics.update(
        {
            "mode": mode,
            "scene_index": int(index),
            "scenario_id": _scenario_id(batch),
            "target_agents": target_agents,
            "target_time_window": args.target_time_window,
            "elapsed_sec": elapsed,
            "empty_output": False,
            "guidance_metrics": _scalar_dict(output.get("guidance_metrics", {})),
        }
    )
    if "next_token_idx_gt" in output:
        pred_tokens = output["next_token_idx"]
        gt_tokens = output["next_token_idx_gt"]
        token_valid = output["next_token_eval_mask"].bool()
        metrics["token_change_rate_vs_gt"] = _token_change_rate(
            pred_tokens,
            gt_tokens,
            token_valid,
        )
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Run a small checkpoint smoke for causal guidance modes."
    )
    parser.add_argument(
        "--config",
        default="configs/train/train_scalable_causal_diffusion_local.yaml",
    )
    parser.add_argument(
        "--ckpt",
        default="checkpoints/causal_diffusion/epoch=00.ckpt",
    )
    parser.add_argument("--raw-dir", default="data/valid_demo")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--indices", default="")
    parser.add_argument("--num-scenes", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--modes", default="seed,none,safe,ego_stress,ego_edit")
    parser.add_argument("--target-agents", default="")
    parser.add_argument("--max-target-agents", type=int, default=1)
    parser.add_argument("--target-time-window", default="2,4")
    parser.add_argument("--target-spec", default="ego_risk")
    parser.add_argument("--ego-interaction-alpha", type=float, default=None)
    parser.add_argument("--target-event-eta", type=float, default=None)
    parser.add_argument("--edit-gamma", type=float, default=None)
    parser.add_argument("--invalid-beta", type=float, default=None)
    parser.add_argument("--path-corridor-width", type=float, default=None)
    parser.add_argument("--conflict-tta-threshold", type=float, default=None)
    parser.add_argument("--near-miss-distance", type=float, default=None)
    parser.add_argument("--ttc-threshold", type=float, default=None)
    parser.add_argument(
        "--output",
        default="outputs/causal_guidance_smoke/metrics.json",
    )
    parser.add_argument("--records-csv", default="")
    parser.add_argument("--summary-csv", default="")
    args = parser.parse_args()

    config = load_config_act(args.config)
    model_config = copy.deepcopy(config)
    model, load_info = _load_model(model_config, args.ckpt, torch.device(args.device))
    dataset = _load_dataset(config, args.raw_dir)
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty.")
    modes = [mode.strip().lower() for mode in args.modes.split(",") if mode.strip()]
    scene_indices = _parse_indices(
        args.index,
        args.indices,
        args.num_scenes,
        len(dataset),
    )
    if not scene_indices:
        raise RuntimeError("No valid scene indices selected.")
    results = []
    for scene_index in scene_indices:
        for mode in modes:
            if mode not in {"seed", "none", "safe", "ego_stress", "ego_edit"}:
                raise ValueError(f"Unsupported mode: {mode}")
            print(f"[smoke] running index={scene_index} mode={mode}")
            if mode == "seed":
                results.append(
                    _seed_baseline(
                        model,
                        dataset,
                        scene_index,
                        torch.device(args.device),
                        args,
                    )
                )
            else:
                results.append(
                    _run_mode(
                        model,
                        dataset,
                        scene_index,
                        torch.device(args.device),
                        mode,
                        args,
                    )
                )

    summary = _aggregate_results(results)
    summary["pareto_modes"] = _pareto_modes(summary)

    output = {
        "config": args.config,
        "ckpt": args.ckpt,
        "raw_dir": args.raw_dir,
        "index": args.index,
        "indices": scene_indices,
        "device": args.device,
        "seed": args.seed,
        "load_info": load_info,
        "summary": summary,
        "results": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    records_csv = (
        Path(args.records_csv)
        if args.records_csv
        else output_path.with_name("records.csv")
    )
    summary_csv = (
        Path(args.summary_csv)
        if args.summary_csv
        else output_path.with_name("summary.csv")
    )
    _write_csv(records_csv, results)
    _write_csv(summary_csv, _summary_rows(summary))
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
