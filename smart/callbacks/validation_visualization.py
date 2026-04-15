from pathlib import Path
from typing import Dict, Iterable, Optional

import pytorch_lightning as pl
import torch
from torch_geometric.data import Batch


class ValidationVisualizationCallback(pl.Callback):
    def __init__(
        self,
        enabled: bool = False,
        interval_epochs: int = 1,
        sample_indices: Optional[Iterable[int]] = None,
        output_dir: str = "outputs/val_visualizations",
        max_agents: int = 0,
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.interval_epochs = max(1, int(interval_epochs))
        self.sample_indices = [int(index) for index in (sample_indices or [0])]
        self.output_dir = output_dir
        self.max_agents = int(max_agents)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not self.enabled or not trainer.is_global_zero:
            return
        if getattr(pl_module, "training_stage", "joint") == "pretrain":
            return
        epoch = trainer.current_epoch + 1
        if epoch % self.interval_epochs != 0:
            return
        datamodule = trainer.datamodule
        if datamodule is None or not hasattr(datamodule, "val_dataset"):
            return
        dataset = datamodule.val_dataset
        output_root = Path(self.output_dir) / pl_module.model_config.predictor / f"epoch_{epoch:03d}"
        output_root.mkdir(parents=True, exist_ok=True)

        was_training = pl_module.training
        pl_module.eval()
        try:
            with torch.no_grad():
                for sample_index in self.sample_indices:
                    if sample_index < 0 or sample_index >= len(dataset):
                        continue
                    graph = dataset[sample_index]
                    batch = Batch.from_data_list([graph]).to(pl_module.device)
                    prepared = self._prepare_batch(pl_module, batch)
                    prediction = pl_module.inference(prepared)
                    jepa_overlay = self._build_jepa_overlay(pl_module, prepared)
                    scenario_id = self._scenario_id(graph)
                    filename = f"idx_{sample_index:05d}_{scenario_id}.png"
                    save_validation_visualization(
                        data=prepared.cpu(),
                        prediction={key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in prediction.items()},
                        jepa_overlay=jepa_overlay,
                        output_path=output_root / filename,
                        title=f"{pl_module.model_config.predictor} epoch={epoch} idx={sample_index} scenario={scenario_id}",
                        max_agents=self.max_agents,
                    )
        finally:
            if was_training:
                pl_module.train()

    @staticmethod
    def _prepare_batch(pl_module, batch: Batch) -> Batch:
        data = pl_module.match_token_map(batch)
        data = pl_module.sample_pt_pred(data)
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        return data

    @staticmethod
    def _scenario_id(graph) -> str:
        scenario_id = getattr(graph, "scenario_id", None)
        if scenario_id is None and hasattr(graph, "get"):
            scenario_id = graph.get("scenario_id", None)
        if scenario_id is None and hasattr(graph, "__contains__") and "scenario_id" in graph:
            scenario_id = graph["scenario_id"]
        if scenario_id is None and isinstance(graph, dict):
            scenario_id = graph.get("scenario_id")
        if scenario_id is None:
            return "unknown"
        return str(scenario_id).replace("/", "_")

    @staticmethod
    def _build_jepa_overlay(pl_module, data: Batch) -> Optional[Dict[str, object]]:
        required_attrs = [
            "_build_future_targets",
            "_build_agent_chunk_mask",
            "_get_node_batch",
            "_build_random_chunk_mask",
            "_build_masked_agent_history_context_mask",
        ]
        if not all(hasattr(pl_module, attr) for attr in required_attrs):
            return None

        future_states, future_mask, valid_agents = pl_module._build_future_targets(data)
        del future_states
        agent_chunk_mask = pl_module._build_agent_chunk_mask(valid_agents, future_mask)
        agent_graph_index = pl_module._get_node_batch(data, "agent")

        if getattr(pl_module, "mask_strategy", "random_chunk") == "interaction_multiblock" and hasattr(
            pl_module, "_build_interaction_joint_masks"
        ):
            agent_prediction_mask, map_prediction_mask, _map_visible_mask = pl_module._build_interaction_joint_masks(
                data,
                agent_chunk_mask,
                agent_graph_index,
            )
        else:
            agent_prediction_mask = pl_module._build_random_chunk_mask(agent_chunk_mask, agent_graph_index)
            map_prediction_mask = torch.zeros(
                int(data["map_polygon"]["num_nodes"]),
                dtype=torch.bool,
                device=agent_prediction_mask.device,
            )

        if not getattr(pl_module, "predict_map_latent", False) or not getattr(pl_module, "joint_predictor", False):
            map_prediction_mask = torch.zeros_like(map_prediction_mask)

        _agent_history_mask, history_stats = pl_module._build_masked_agent_history_context_mask(
            data,
            agent_prediction_mask,
        )

        return {
            "future_mask": future_mask.cpu(),
            "agent_prediction_mask": agent_prediction_mask.cpu(),
            "map_prediction_mask": map_prediction_mask.cpu(),
            "future_chunk_steps": int(getattr(pl_module, "future_chunk_steps", 5)),
            "history_mode": str(getattr(pl_module, "masked_agent_history_mode", "visible")),
            "history_visible_fraction": float(history_stats["masked_agent_history_visible_fraction"]),
        }


def save_validation_visualization(
    data,
    prediction,
    output_path: Path,
    title: str,
    max_agents: int = 0,
    jepa_overlay: Optional[Dict[str, object]] = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Polygon

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hist_steps = data["agent"]["valid_mask"].shape[1] - prediction["gt"].shape[1]
    current_step = hist_steps - 1
    valid_agents = data["agent"]["valid_mask"][:, current_step] & (data["agent"]["type"] != 3)
    if valid_agents.any():
        agent_indices = torch.nonzero(valid_agents, as_tuple=False).squeeze(-1)
    else:
        agent_indices = torch.arange(data["agent"]["num_nodes"])

    av_index = int(data["agent"]["av_index"])
    if max_agents > 0 and agent_indices.numel() > max_agents:
        anchor = data["agent"]["position"][av_index, current_step, :2]
        distance = torch.norm(data["agent"]["position"][agent_indices, current_step, :2] - anchor, dim=-1)
        keep = torch.argsort(distance)[:max_agents]
        agent_indices = agent_indices[keep]
        if av_index not in agent_indices.tolist():
            agent_indices = torch.cat([torch.tensor([av_index]), agent_indices[:-1]])

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_title(title)

    _draw_map(ax, data, jepa_overlay)
    _draw_agents(ax, data, prediction, agent_indices, current_step, hist_steps, av_index, Polygon, Circle, jepa_overlay)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if jepa_overlay is not None:
        ax.text(
            0.02,
            0.98,
            " | ".join(
                [
                    f"masked agents={int(jepa_overlay['agent_prediction_mask'].any(dim=-1).sum())}",
                    f"masked polygons={int(jepa_overlay['map_prediction_mask'].sum())}",
                    f"history={jepa_overlay['history_mode']}",
                    f"hist vis={float(jepa_overlay['history_visible_fraction']):.2f}",
                ]
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 2.0},
        )
        ax.legend(handles=_legend_handles(Line2D, show_jepa_overlay=True), loc="lower left", fontsize=8, framealpha=0.92)
    else:
        ax.legend(handles=_legend_handles(Line2D, show_jepa_overlay=False), loc="lower left", fontsize=8, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _draw_map(ax, data, jepa_overlay: Optional[Dict[str, object]] = None) -> None:
    point_to_polygon = data[("map_point", "to", "map_polygon")]["edge_index"]
    polygon_ids = point_to_polygon[1].long()
    point_position = data["map_point"]["position"][:, :2]
    masked_polygon_ids = set()
    if jepa_overlay is not None:
        masked_polygon_ids = set(int(i) for i in torch.nonzero(jepa_overlay["map_prediction_mask"], as_tuple=False).squeeze(-1).tolist())
    for polygon_id in torch.unique(polygon_ids).tolist():
        indices = point_to_polygon[0, polygon_ids == polygon_id]
        if indices.numel() < 2:
            continue
        points = point_position.index_select(0, indices)
        is_masked = polygon_id in masked_polygon_ids
        ax.plot(
            points[:, 0],
            points[:, 1],
            color="#f28e2b" if is_masked else "#c7c7c7",
            linewidth=2.8 if is_masked else 1.0,
            alpha=0.95 if is_masked else 0.8,
            zorder=2 if is_masked else 1,
        )


def _draw_agents(
    ax,
    data,
    prediction,
    agent_indices,
    current_step,
    hist_steps,
    av_index,
    polygon_cls,
    circle_cls,
    jepa_overlay: Optional[Dict[str, object]] = None,
) -> None:
    step_prediction_mask = None
    masked_agent_mask = None
    if jepa_overlay is not None:
        step_prediction_mask = _chunk_mask_to_step_mask(
            jepa_overlay["agent_prediction_mask"],
            jepa_overlay["future_mask"],
            int(jepa_overlay["future_chunk_steps"]),
        )
        masked_agent_mask = jepa_overlay["agent_prediction_mask"].any(dim=-1)

    for agent_index in agent_indices.tolist():
        is_ego = agent_index == av_index
        is_masked = bool(masked_agent_mask[agent_index]) if masked_agent_mask is not None else False
        color = "#8f63d2" if is_ego else "#a9d2ff"
        edge_color = "#8f63d2" if is_ego else ("#e15759" if is_masked else "#000000")
        linewidth = 2.4 if is_ego else 1.8

        history_mask = data["agent"]["valid_mask"][agent_index, :hist_steps]
        history = data["agent"]["position"][agent_index, :hist_steps, :2][history_mask]
        gt_mask = data["agent"]["valid_mask"][agent_index, hist_steps:]
        gt = prediction["gt"][agent_index][gt_mask]
        pred = prediction["pred_traj"][agent_index]
        pred_valid = gt_mask[:pred.shape[0]] if gt_mask.numel() >= pred.shape[0] else torch.ones(pred.shape[0], dtype=torch.bool)
        pred = pred[pred_valid]
        masked_gt = None
        visible_gt = gt
        if step_prediction_mask is not None:
            agent_masked_gt_mask = step_prediction_mask[agent_index] & gt_mask
            masked_gt = prediction["gt"][agent_index][agent_masked_gt_mask]
            visible_gt = prediction["gt"][agent_index][gt_mask & ~step_prediction_mask[agent_index]]

        if history.numel() > 0:
            ax.plot(history[:, 0], history[:, 1], color=color, linestyle=":", linewidth=linewidth, alpha=0.95, zorder=3)
        if visible_gt.numel() > 0:
            ax.plot(visible_gt[:, 0], visible_gt[:, 1], color=color, linestyle="-", linewidth=linewidth, alpha=0.75, zorder=4)
        if masked_gt is not None and masked_gt.numel() > 0:
            ax.plot(masked_gt[:, 0], masked_gt[:, 1], color="#2ca02c", linestyle="-", linewidth=linewidth + 0.9, alpha=0.95, zorder=5)
        if pred.numel() > 0:
            ax.plot(pred[:, 0], pred[:, 1], color="#e15759", linestyle="-", linewidth=linewidth, alpha=0.95, zorder=6)

        current_xy = data["agent"]["position"][agent_index, current_step, :2]
        current_heading = float(data["agent"]["heading"][agent_index, current_step].item())
        shape = data["agent"]["shape"][agent_index, current_step, :2]
        agent_type = int(data["agent"]["type"][agent_index].item())
        face_color = "#8f63d2" if is_ego else ("#f6b5ae" if is_masked else "#a9d2ff")
        _draw_agent_shape(ax, current_xy, current_heading, shape, agent_type, face_color, edge_color, polygon_cls, circle_cls)


def _draw_agent_shape(ax, center_xy, heading, shape, agent_type, face_color, edge_color, polygon_cls, circle_cls) -> None:
    length = max(float(shape[0].item()), 0.6)
    width = max(float(shape[1].item()), 0.4)

    if agent_type == 1:
        patch = circle_cls(
            (center_xy[0].item(), center_xy[1].item()),
            radius=0.3 * max(length, width),
            facecolor=face_color,
            edgecolor=edge_color,
            linewidth=1.2,
            alpha=0.28,
            zorder=6,
        )
        ax.add_patch(patch)
    else:
        corners = _oriented_box(center_xy, heading, length, width)
        patch = polygon_cls(
            corners,
            closed=True,
            facecolor=face_color,
            edgecolor=edge_color,
            linewidth=1.2,
            alpha=0.28,
            zorder=6,
        )
        ax.add_patch(patch)

    arrow = polygon_cls(
        _heading_triangle(center_xy, heading, length, width),
        closed=True,
        facecolor="none",
        edgecolor=edge_color,
        linewidth=1.1,
        alpha=0.95,
        zorder=7,
    )
    ax.add_patch(arrow)


def _oriented_box(center_xy, heading, length, width):
    cos = torch.cos(torch.tensor(heading, dtype=torch.float32))
    sin = torch.sin(torch.tensor(heading, dtype=torch.float32))
    forward = torch.tensor([cos, sin]) * (length * 0.5)
    lateral = torch.tensor([-sin, cos]) * (width * 0.5)
    corners = [
        center_xy + forward + lateral,
        center_xy + forward - lateral,
        center_xy - forward - lateral,
        center_xy - forward + lateral,
    ]
    return [(float(point[0].item()), float(point[1].item())) for point in corners]


def _heading_triangle(center_xy, heading, length, width):
    direction = torch.tensor(
        [torch.cos(torch.tensor(heading, dtype=torch.float32)), torch.sin(torch.tensor(heading, dtype=torch.float32))]
    )
    lateral = torch.tensor([-direction[1], direction[0]])
    triangle_length = 0.5 * length
    tip = center_xy + direction * (0.5 * triangle_length)
    base_center = center_xy - direction * (0.5 * triangle_length)
    base_left = base_center + lateral * (0.5 * width)
    base_right = base_center - lateral * (0.5 * width)
    points = [tip, base_right, base_left]
    return [(float(point[0].item()), float(point[1].item())) for point in points]


def _chunk_mask_to_step_mask(chunk_mask: torch.Tensor, future_mask: torch.Tensor, future_chunk_steps: int) -> torch.Tensor:
    step_mask = chunk_mask.repeat_interleave(future_chunk_steps, dim=-1)
    if step_mask.shape[-1] > future_mask.shape[-1]:
        step_mask = step_mask[:, : future_mask.shape[-1]]
    return step_mask & future_mask


def _legend_handles(line_cls, show_jepa_overlay: bool):
    handles = [
        line_cls([0], [0], color="#c7c7c7", lw=1.2, label="Map"),
        line_cls([0], [0], color="#8f63d2", lw=1.8, linestyle=":", label="Ego History"),
        line_cls([0], [0], color="#a9d2ff", lw=1.8, linestyle=":", label="Other History"),
        line_cls([0], [0], color="#8f63d2", lw=1.8, linestyle="-", label="GT Future"),
        line_cls([0], [0], color="#e15759", lw=1.8, linestyle="-", label="Pred Future"),
    ]
    if show_jepa_overlay:
        handles = [
            line_cls([0], [0], color="#c7c7c7", lw=1.2, label="Visible Map"),
            line_cls([0], [0], color="#f28e2b", lw=2.8, label="Masked Map"),
            line_cls([0], [0], color="#8f63d2", lw=1.8, linestyle=":", label="Ego History"),
            line_cls([0], [0], color="#a9d2ff", lw=1.8, linestyle=":", label="Other History"),
            line_cls([0], [0], color="#8f63d2", lw=1.8, linestyle="-", label="Visible GT Future"),
            line_cls([0], [0], color="#2ca02c", lw=2.4, linestyle="-", label="Masked GT Future"),
            line_cls([0], [0], color="#e15759", lw=1.8, linestyle="-", label="Pred Future"),
        ]
    return handles
