import argparse
import ast
import copy
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from smart.utils.config import load_config_act
from scripts.smoke_causal_guidance_modes import (
    _aggregate_results,
    _apply_guidance_overrides,
    _attach_seed_controls,
    _load_dataset,
    _load_model,
    _make_batch,
    _parse_indices,
    _parse_ints,
    _pareto_modes,
    _scalar_dict,
    _scenario_id,
    _select_target_agents,
    _summary_rows,
    _token_change_rate,
    _trajectory_metrics,
    _write_csv,
    _zero_guidance_metrics,
)


SUPPORTED_MODES = ("seed", "none", "safe", "ego_stress", "ego_edit")


def _clean_scenario_id(value) -> str:
    text = str(value)
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = None
    if isinstance(parsed, (list, tuple)) and len(parsed) == 1:
        text = str(parsed[0])
    elif parsed is not None and not isinstance(parsed, (list, tuple, dict)):
        text = str(parsed)
    replacements = {
        "/": "_",
        "\\": "_",
        "[": "",
        "]": "",
        "'": "",
        '"': "",
        " ": "_",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _mode_list(value: str) -> List[str]:
    modes = []
    seen = set()
    for raw_mode in str(value).split(","):
        mode = raw_mode.strip().lower()
        if not mode or mode in seen:
            continue
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"Unsupported mode: {mode}")
        modes.append(mode)
        seen.add(mode)
    if not modes:
        raise ValueError("At least one mode must be selected.")
    return modes


def _target_window(value: str) -> Optional[Tuple[int, int]]:
    if value is None or str(value).strip() == "":
        return None
    parsed = _parse_ints(str(value))
    if len(parsed) != 2:
        raise ValueError("--target-time-window must contain exactly start,end.")
    return int(parsed[0]), int(parsed[1])


def _cpu_tensor_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tensor_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tensor_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tensor_tree(item) for item in value)
    return value


def _finite_xy(points: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(points) or points.numel() == 0:
        return torch.zeros(0, 2)
    points = points.detach().cpu().float().reshape(-1, points.shape[-1])[:, :2]
    finite = torch.isfinite(points).all(dim=-1)
    return points[finite]


def _ego_agent_index(data) -> Optional[int]:
    agent = data["agent"]
    agent_count = None
    if "position" in agent:
        agent_count = int(agent["position"].shape[0])
    for key in ("ego_agent_id", "av_index"):
        if key not in agent:
            continue
        value = agent[key]
        try:
            if torch.is_tensor(value):
                flat = value.detach().cpu().reshape(-1)
                if flat.numel() == 0:
                    continue
                index = int(flat[0].item())
            elif isinstance(value, (list, tuple)):
                if len(value) == 0:
                    continue
                index = int(value[0])
            else:
                index = int(value)
        except (TypeError, ValueError):
            continue
        if agent_count is None or 0 <= index < agent_count:
            return index
    return None


def _closest_time_aligned_pair(
    target_path: torch.Tensor,
    ego_path: torch.Tensor,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, float]]:
    target_xy = _finite_xy(target_path)
    ego_xy = _finite_xy(ego_path)
    steps = min(int(target_xy.shape[0]), int(ego_xy.shape[0]))
    if steps <= 0:
        return None
    deltas = target_xy[:steps] - ego_xy[:steps]
    distances = torch.norm(deltas, dim=-1)
    best_index = int(torch.argmin(distances).item())
    return (
        target_xy[best_index],
        ego_xy[best_index],
        float(distances[best_index].item()),
    )


def _first_corridor_entry(
    target_path: torch.Tensor,
    ego_path: torch.Tensor,
    corridor_width: float,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, int, float]]:
    target_xy = _finite_xy(target_path)
    ego_xy = _finite_xy(ego_path)
    if target_xy.numel() == 0 or ego_xy.numel() == 0:
        return None
    distances = torch.cdist(target_xy, ego_xy).min(dim=1).values
    if distances[0] <= float(corridor_width):
        return None
    entry_indices = torch.nonzero(
        distances <= float(corridor_width),
        as_tuple=False,
    ).reshape(-1)
    if entry_indices.numel() == 0:
        return None
    entry_index = int(entry_indices[0].item())
    return (
        target_xy[0],
        target_xy[entry_index],
        entry_index,
        float(distances[entry_index].item()),
    )


def _uses_corridor_entry_marker(target_spec: str) -> bool:
    spec = str(target_spec or "").lower().replace("-", "_")
    return spec in {"cut_in", "cutin", "lane_cut_in"}


def _trajectory_extent(
    history: torch.Tensor,
    gt: torch.Tensor,
    predictions: Dict[str, torch.Tensor],
    min_window_m: float = 60.0,
    extra_paths: Optional[Iterable[torch.Tensor]] = None,
) -> Tuple[float, float, float, float]:
    point_sets = [_finite_xy(history), _finite_xy(gt)]
    point_sets.extend(_finite_xy(points) for points in predictions.values())
    if extra_paths is not None:
        point_sets.extend(_finite_xy(points) for points in extra_paths)
    valid_sets = [points for points in point_sets if points.numel() > 0]
    if not valid_sets:
        half = max(float(min_window_m), 1.0) * 0.5
        return -half, half, -half, half
    all_points = torch.cat(valid_sets, dim=0)
    mins = all_points.min(dim=0).values
    maxs = all_points.max(dim=0).values
    span = (maxs - mins).clamp_min(0.0)
    pad = max(float(span.max().item()) * 0.05, 0.5)
    center = (mins + maxs) * 0.5
    width = max(float(span[0].item()) + 2.0 * pad, float(min_window_m))
    height = max(float(span[1].item()) + 2.0 * pad, float(min_window_m))
    return (
        float(center[0].item() - width * 0.5),
        float(center[0].item() + width * 0.5),
        float(center[1].item() - height * 0.5),
        float(center[1].item() + height * 0.5),
    )


def _panel_title(mode: str, record: Dict) -> str:
    guidance = record.get("guidance_metrics", {}) or {}
    risk_ttc = guidance.get("ego_risk_min_ttc", guidance.get("ego_min_ttc", 0.0))
    return (
        f"{mode}\n"
        f"ADE {float(record.get('ade', 0.0)):.2f} | "
        f"FDE {float(record.get('fde', 0.0)):.2f} | "
        f"ego_d {float(guidance.get('ego_min_distance', 0.0)):.2f} | "
        f"risk_ttc {float(risk_ttc):.2f}\n"
        f"coll {float(guidance.get('hard_collision_rate', 0.0)):.2f} | "
        f"risk {float(guidance.get('ego_risk_success_rate', 0.0)):.2f} | "
        f"edit {float(guidance.get('edit_distance', 0.0)):.2f}"
    )


def _draw_base_map(ax, data) -> None:
    if ("map_point", "to", "map_polygon") not in data.edge_types:
        return
    point_to_polygon = data[("map_point", "to", "map_polygon")]["edge_index"]
    if point_to_polygon.numel() == 0:
        return
    polygon_ids = point_to_polygon[1].long()
    point_position = data["map_point"]["position"][:, :2]
    for polygon_id in torch.unique(polygon_ids).tolist():
        indices = point_to_polygon[0, polygon_ids == polygon_id]
        if indices.numel() < 2:
            continue
        points = point_position.index_select(0, indices).detach().cpu()
        ax.plot(
            points[:, 0],
            points[:, 1],
            color="#bfc7ce",
            linewidth=0.8,
            alpha=0.72,
            zorder=1,
        )


def _agent_future(output: Dict, agent_index: int, key: str) -> torch.Tensor:
    if output is None or key not in output:
        return torch.zeros(0, 2)
    values = output[key][agent_index]
    if key == "gt":
        valid = output.get("official_valid_mask", output.get("valid_mask"))
    else:
        valid = output.get("pred_valid_mask")
    if valid is None:
        return values
    return values[valid[agent_index].bool()]


def _default_target_agents(data, hist_steps: int, max_agents: int) -> List[int]:
    current_valid = data["agent"]["valid_mask"][:, hist_steps - 1].bool()
    agent_type = data["agent"]["type"].long()
    mask = current_valid & (agent_type != 3)
    if "category" in data["agent"]:
        category3 = mask & (data["agent"]["category"].long() == 3)
        if category3.any():
            mask = category3
    indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    if indices.numel() == 0:
        indices = torch.nonzero(current_valid, as_tuple=False).squeeze(-1)
    return indices[: max(1, int(max_agents))].tolist()


def _plot_guidance_comparison(
    output_path: Path,
    data_cpu,
    mode_results: Dict[str, Dict],
    target_agents: Sequence[int],
    scenario_id: str,
    min_window_m: float,
    path_corridor_width: float = 3.0,
    target_spec: str = "ego_risk",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    first_output = next(
        (item.get("output") for item in mode_results.values() if item.get("output") is not None),
        None,
    )
    if first_output is None:
        return
    hist_steps = int(data_cpu["agent"]["valid_mask"].shape[1] - first_output["gt"].shape[1])
    if not target_agents:
        target_agents = _default_target_agents(data_cpu, hist_steps, max_agents=1)
    modes = list(mode_results.keys())
    rows = max(1, len(target_agents))
    cols = max(1, len(modes))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4.8 * cols, 5.2 * rows + 0.8),
        squeeze=False,
    )
    current_step = hist_steps - 1
    current_valid = data_cpu["agent"]["valid_mask"][:, current_step].bool()
    current_xy_all = data_cpu["agent"]["position"][:, current_step, :2].detach().cpu()
    ego_index = _ego_agent_index(data_cpu)
    show_corridor_entry = _uses_corridor_entry_marker(target_spec)
    ego_history = torch.zeros(0, 2)
    ego_gt = torch.zeros(0, 2)
    if ego_index is not None:
        ego_history_mask = data_cpu["agent"]["valid_mask"][ego_index, :hist_steps].bool()
        ego_history = data_cpu["agent"]["position"][ego_index, :hist_steps, :2][
            ego_history_mask
        ].detach().cpu()
        ego_gt = _agent_future(first_output, int(ego_index), "gt")
    styles = {
        "seed": {"color": "#2ca02c", "linewidth": 2.4, "label": "seed/GT"},
        "none": {"color": "#e15759", "linewidth": 2.0, "label": "none"},
        "safe": {"color": "#1f77b4", "linewidth": 2.0, "label": "safe"},
        "ego_stress": {"color": "#d62728", "linewidth": 2.4, "label": "ego stress"},
        "ego_edit": {"color": "#7b2cbf", "linewidth": 2.2, "label": "ego edit"},
    }

    for row_idx, agent_index in enumerate(target_agents):
        history_mask = data_cpu["agent"]["valid_mask"][agent_index, :hist_steps].bool()
        history = data_cpu["agent"]["position"][agent_index, :hist_steps, :2][history_mask].detach().cpu()
        gt = _agent_future(first_output, int(agent_index), "gt")
        predictions_for_extent = {}
        extra_paths = []
        if ego_history.numel() > 0:
            extra_paths.append(ego_history)
        if ego_gt.numel() > 0:
            extra_paths.append(ego_gt)
        for mode, result in mode_results.items():
            output = result.get("output")
            if output is not None:
                predictions_for_extent[mode] = _agent_future(output, int(agent_index), "pred_traj")
                if ego_index is not None:
                    extra_paths.append(_agent_future(output, int(ego_index), "pred_traj"))
        x_min, x_max, y_min, y_max = _trajectory_extent(
            history,
            gt,
            predictions_for_extent,
            min_window_m=float(min_window_m),
            extra_paths=extra_paths,
        )
        for col_idx, mode in enumerate(modes):
            ax = axes[row_idx, col_idx]
            result = mode_results[mode]
            record = result.get("record", {})
            output = result.get("output")
            _draw_base_map(ax, data_cpu)
            context_mask = current_valid.detach().cpu().clone()
            if 0 <= int(agent_index) < int(context_mask.shape[0]):
                context_mask[int(agent_index)] = False
            if ego_index is not None and 0 <= int(ego_index) < int(context_mask.shape[0]):
                context_mask[int(ego_index)] = False
            context_xy = current_xy_all[context_mask]
            if context_xy.numel() > 0:
                ax.scatter(
                    context_xy[:, 0],
                    context_xy[:, 1],
                    s=12,
                    color="#6b7280",
                    alpha=0.45,
                    zorder=3,
                    label="current agents",
                )
            ego_future = ego_gt
            if output is not None and ego_index is not None:
                ego_future = _agent_future(output, int(ego_index), "pred_traj")
            if ego_future.numel() > 0:
                ax.plot(
                    ego_future[:, 0],
                    ego_future[:, 1],
                    color="#0284c7",
                    linewidth=11.0,
                    alpha=0.12,
                    zorder=4,
                    solid_capstyle="round",
                    label="ego corridor",
                )
                ax.plot(
                    ego_future[:, 0],
                    ego_future[:, 1],
                    color="#0369a1",
                    linestyle="-",
                    linewidth=2.5,
                    alpha=0.95,
                    zorder=7,
                    label="ego/SDC future",
                )
            if ego_history.numel() > 0:
                ax.plot(
                    ego_history[:, 0],
                    ego_history[:, 1],
                    color="#075985",
                    linestyle=":",
                    linewidth=2.0,
                    alpha=0.85,
                    zorder=6,
                    label="ego history",
                )
            if ego_index is not None:
                ego_xy = current_xy_all[int(ego_index)]
                ax.scatter(
                    [ego_xy[0]],
                    [ego_xy[1]],
                    s=86,
                    facecolors="#ffffff",
                    edgecolors="#0369a1",
                    linewidths=1.8,
                    marker="D",
                    zorder=12,
                    label="ego/SDC",
                )
                ax.annotate(
                    "ego",
                    xy=(float(ego_xy[0]), float(ego_xy[1])),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    color="#075985",
                    zorder=13,
                )
            if history.numel() > 0:
                ax.plot(
                    history[:, 0],
                    history[:, 1],
                    color="#111827",
                    linestyle=":",
                    linewidth=1.8,
                    alpha=0.9,
                    zorder=5,
                    label="history",
                )
            if gt.numel() > 0:
                ax.plot(
                    gt[:, 0],
                    gt[:, 1],
                    color="#2ca02c",
                    linestyle="-",
                    linewidth=1.9,
                    alpha=0.72,
                    zorder=6,
                    label="GT future",
                )
            if output is not None:
                pred = _agent_future(output, int(agent_index), "pred_traj")
                if pred.numel() > 0:
                    style = styles.get(mode, styles["none"])
                    linestyle = "--" if mode == "seed" else "-"
                    ax.plot(
                        pred[:, 0],
                        pred[:, 1],
                        color=style["color"],
                        linestyle=linestyle,
                        linewidth=style["linewidth"],
                        alpha=0.96,
                        zorder=8,
                        label=style["label"],
                    )
                    if pred.shape[0] > 2:
                        marker_step = max(1, int(pred.shape[0]) // 8)
                        marker_xy = pred[::marker_step]
                        ax.scatter(
                            marker_xy[:, 0],
                            marker_xy[:, 1],
                            s=13,
                            color=style["color"],
                            alpha=0.72,
                            zorder=9,
                        )
                    pair = _closest_time_aligned_pair(pred, ego_future)
                    if pair is not None:
                        target_near, ego_near, _distance = pair
                        ax.plot(
                            [target_near[0], ego_near[0]],
                            [target_near[1], ego_near[1]],
                            color="#f97316",
                            linestyle="--",
                            linewidth=1.4,
                            alpha=0.88,
                            zorder=9,
                            label="target-ego closest",
                        )
                        ax.scatter(
                            [target_near[0], ego_near[0]],
                            [target_near[1], ego_near[1]],
                            s=22,
                            color="#f97316",
                            alpha=0.9,
                            zorder=10,
                        )
                    if show_corridor_entry:
                        entry = _first_corridor_entry(
                            pred,
                            ego_future,
                            corridor_width=float(path_corridor_width),
                        )
                        if entry is not None:
                            start_xy, entry_xy, _entry_index, _entry_distance = entry
                            ax.annotate(
                                "",
                                xy=(float(entry_xy[0]), float(entry_xy[1])),
                                xytext=(float(start_xy[0]), float(start_xy[1])),
                                arrowprops={
                                    "arrowstyle": "->",
                                    "color": "#facc15",
                                    "linewidth": 1.8,
                                    "alpha": 0.95,
                                },
                                zorder=12,
                            )
                            ax.scatter(
                                [entry_xy[0]],
                                [entry_xy[1]],
                                s=54,
                                color="#facc15",
                                edgecolors="#7c2d12",
                                linewidths=0.8,
                                zorder=13,
                                label="corridor entry",
                            )
            target_xy = current_xy_all[int(agent_index)]
            ax.scatter(
                [target_xy[0]],
                [target_xy[1]],
                s=78,
                color="#f97316",
                marker="*",
                edgecolors="#111827",
                linewidths=0.8,
                zorder=10,
                label="controlled target",
            )
            ax.annotate(
                "controlled",
                xy=(float(target_xy[0]), float(target_xy[1])),
                xytext=(5, -11),
                textcoords="offset points",
                fontsize=8,
                color="#7c2d12",
                zorder=13,
            )
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, linestyle=":", alpha=0.25)
            ax.set_title(_panel_title(mode, record), fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f"controlled agent {int(agent_index)}")
    handles = [
        Line2D([0], [0], color="#111827", linestyle=":", linewidth=1.8, label="controlled history"),
        Line2D([0], [0], color="#2ca02c", linewidth=1.9, label="controlled GT future"),
        Line2D([0], [0], color="#0369a1", linewidth=2.5, label="ego/SDC future"),
        Line2D([0], [0], color="#0284c7", linewidth=7.0, alpha=0.18, label="ego corridor"),
        Line2D([0], [0], color="#f97316", linestyle="--", linewidth=1.4, label="target-ego closest"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#6b7280", markersize=5, label="current agents"),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markeredgecolor="#0369a1",
            markerfacecolor="#ffffff",
            markersize=7,
            label="ego/SDC",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markeredgecolor="#111827",
            markerfacecolor="#f97316",
            markersize=9,
            label="controlled target",
        ),
    ]
    if show_corridor_entry:
        handles.insert(
            5,
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markeredgecolor="#7c2d12",
                markerfacecolor="#facc15",
                markersize=7,
                label="corridor entry",
            ),
        )
    for mode in modes:
        style = styles.get(mode, styles["none"])
        handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                linewidth=style["linewidth"],
                linestyle="--" if mode == "seed" else "-",
                label=style["label"],
            )
        )
    fig.legend(handles=handles, loc="lower center", ncol=min(8, len(handles)), fontsize=9)
    fig.suptitle(
        f"Causal guidance comparison | scenario={scenario_id}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.93))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=155)
    plt.close(fig)


def _seed_visual_output(batch, model, target_agents: Sequence[int]) -> Tuple[Dict, Dict]:
    future_start = int(model.num_historical_steps)
    future_end = future_start + int(model.ar_total_rollout_steps)
    gt = batch["agent"]["position"][:, future_start:future_end, :2].float()
    official_valid = batch["agent"]["valid_mask"][:, future_start:future_end].bool()
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
            "guidance_metrics": _zero_guidance_metrics(),
            "token_change_rate_vs_gt": 0.0,
            "elapsed_sec": 0.0,
            "empty_output": False,
        }
    )
    return metrics, output


def _run_visual_mode(
    model,
    dataset,
    index: int,
    device: torch.device,
    mode: str,
    args,
) -> Tuple[Dict, Optional[Dict], List[int], str]:
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
    scenario = _clean_scenario_id(_scenario_id(batch))
    target_window = _target_window(args.target_time_window)
    if mode == "seed":
        record, output = _seed_visual_output(batch, model, target_agents)
    else:
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
            record = {
                "mode": mode,
                "elapsed_sec": elapsed,
                "empty_output": True,
                "guidance_metrics": _zero_guidance_metrics(),
            }
        else:
            record = _trajectory_metrics(output, target_agents=target_agents)
            record.update(
                {
                    "mode": mode,
                    "elapsed_sec": elapsed,
                    "empty_output": False,
                    "guidance_metrics": _scalar_dict(output.get("guidance_metrics", {})),
                }
            )
            if "next_token_idx_gt" in output:
                pred_tokens = output["next_token_idx"]
                gt_tokens = output["next_token_idx_gt"]
                token_valid = output["next_token_eval_mask"].bool()
                record["token_change_rate_vs_gt"] = _token_change_rate(
                    pred_tokens,
                    gt_tokens,
                    token_valid,
                )
    record.update(
        {
            "scene_index": int(index),
            "scenario_id": scenario,
            "target_agents": list(target_agents),
            "target_time_window": args.target_time_window,
        }
    )
    return record, _cpu_tensor_tree(output) if output is not None else None, target_agents, scenario


def _json_ready(value):
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_manifest(
    path: Path,
    config: str,
    ckpt: str,
    raw_dir: str,
    indices: Sequence[int],
    seed: int,
    load_info: Dict,
    figures: Sequence[Path],
    records: Sequence[Dict],
) -> None:
    payload = {
        "config": config,
        "ckpt": ckpt,
        "raw_dir": raw_dir,
        "indices": [int(index) for index in indices],
        "seed": int(seed),
        "load_info": _json_ready(load_info),
        "figures": [str(path) for path in figures],
        "records": _json_ready(list(records)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _scene_output_path(output_dir: Path, index: int, scenario_id: str) -> Path:
    safe_scenario = _clean_scenario_id(scenario_id)
    return output_dir / f"idx_{int(index):05d}_{safe_scenario}_guidance_modes.png"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize causal guidance modes on selected validation scenes."
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
    parser.add_argument("--num-scenes", type=int, default=3)
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
    parser.add_argument("--min-window-m", type=float, default=70.0)
    parser.add_argument(
        "--output-dir",
        default="outputs/causal_guidance_visualizations",
    )
    parser.add_argument("--manifest", default="")
    parser.add_argument("--records-csv", default="")
    parser.add_argument("--summary-csv", default="")
    args = parser.parse_args()

    device = torch.device(args.device)
    config = load_config_act(args.config)
    model_config = copy.deepcopy(config)
    model, load_info = _load_model(model_config, args.ckpt, device)
    dataset = _load_dataset(config, args.raw_dir)
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty.")
    modes = _mode_list(args.modes)
    scene_indices = _parse_indices(
        args.index,
        args.indices,
        args.num_scenes,
        len(dataset),
    )
    if not scene_indices:
        raise RuntimeError("No valid scene indices selected.")

    output_dir = Path(args.output_dir)
    records = []
    figures = []
    for scene_index in scene_indices:
        mode_results = {}
        target_agents = []
        scenario = "unknown"
        for mode in modes:
            print(f"[visualize] running index={scene_index} mode={mode}")
            record, output, selected_agents, scenario = _run_visual_mode(
                model,
                dataset,
                scene_index,
                device,
                mode,
                args,
            )
            if not target_agents:
                target_agents = selected_agents
            records.append(record)
            mode_results[mode] = {"record": record, "output": output}
        data_cpu = _make_batch(dataset, scene_index, torch.device("cpu"))
        figure_path = _scene_output_path(output_dir, scene_index, scenario)
        _plot_guidance_comparison(
            figure_path,
            data_cpu,
            mode_results,
            target_agents,
            scenario,
            args.min_window_m,
            float(args.path_corridor_width if args.path_corridor_width is not None else 3.0),
            args.target_spec,
        )
        figures.append(figure_path)
        print(f"[visualize] wrote {figure_path}")

    summary = _aggregate_results(records)
    summary["pareto_modes"] = _pareto_modes(summary)
    manifest_path = Path(args.manifest) if args.manifest else output_dir / "manifest.json"
    _write_manifest(
        manifest_path,
        config=args.config,
        ckpt=args.ckpt,
        raw_dir=args.raw_dir,
        indices=scene_indices,
        seed=args.seed,
        load_info=load_info,
        figures=figures,
        records=records,
    )
    records_csv = Path(args.records_csv) if args.records_csv else output_dir / "records.csv"
    summary_csv = Path(args.summary_csv) if args.summary_csv else output_dir / "summary.csv"
    _write_csv(records_csv, records)
    _write_csv(summary_csv, _summary_rows(summary))
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(_json_ready(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "records_csv": str(records_csv),
                "summary_csv": str(summary_csv),
                "summary_json": str(summary_path),
                "figures": [str(path) for path in figures],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
