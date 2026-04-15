from argparse import ArgumentParser
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon
import torch
from torch_geometric.data import Batch

from smart.datasets.scalable_dataset import MultiDataset
from smart.model.smart_jepa import SMARTJEPA
from smart.transforms import WaymoTargetBuilder
from smart.utils.config import load_config_act


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train/train_scalable_jepa.yaml")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--indices", type=int, nargs="*", default=[0, 1, 2, 3])
    parser.add_argument("--output-dir", type=str, default="outputs/jepa_mask_debug")
    parser.add_argument("--max-agents", type=int, default=40)
    parser.add_argument("--radius", type=float, default=70.0)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def load_dataset(config, split):
    if split == "train":
        raw_dir = config.Dataset.train_raw_dir
        processed_dir = config.Dataset.train_processed_dir
    else:
        raw_dir = config.Dataset.val_raw_dir
        processed_dir = config.Dataset.val_processed_dir
    return MultiDataset(
        root=config.Dataset.root,
        split=split,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        transform=WaymoTargetBuilder(config.Model.num_historical_steps, config.Model.decoder.num_future_steps),
    )


def scenario_id_from_graph(graph):
    scenario_id = getattr(graph, "scenario_id", None)
    if scenario_id is None and hasattr(graph, "get"):
        scenario_id = graph.get("scenario_id", None)
    if scenario_id is None and hasattr(graph, "__contains__") and "scenario_id" in graph:
        scenario_id = graph["scenario_id"]
    if scenario_id is None:
        return "unknown"
    return str(scenario_id).replace("/", "_")


def prepare_sample(model, graph):
    batch = Batch.from_data_list([graph])
    data = model._prepare_batch(batch)
    future_states, future_mask, valid_agents = model._build_future_targets(data)
    agent_chunk_mask = model._build_agent_chunk_mask(valid_agents, future_mask)
    agent_graph_index = model._get_node_batch(data, "agent")
    if model.mask_strategy == "interaction_multiblock":
        agent_prediction_mask, map_prediction_mask, map_visible_mask = model._build_interaction_joint_masks(
            data,
            agent_chunk_mask,
            agent_graph_index,
        )
    else:
        agent_prediction_mask = model._build_random_chunk_mask(agent_chunk_mask, agent_graph_index)
        map_prediction_mask = torch.zeros(int(data["map_polygon"]["num_nodes"]), dtype=torch.bool, device=agent_chunk_mask.device)
        map_visible_mask = None
    _agent_history_mask, history_stats = model._build_masked_agent_history_context_mask(data, agent_prediction_mask)
    polygon_centers, polygon_valid_mask, _ = model._compute_polygon_centers(data)
    return {
        "data": data.cpu(),
        "agent_prediction_mask": agent_prediction_mask.cpu(),
        "agent_chunk_mask": agent_chunk_mask.cpu(),
        "future_mask": future_mask.cpu(),
        "map_prediction_mask": map_prediction_mask.cpu(),
        "map_visible_mask": None if map_visible_mask is None else map_visible_mask.cpu(),
        "polygon_centers": polygon_centers.cpu(),
        "polygon_valid_mask": polygon_valid_mask.cpu(),
        "history_mode": model.masked_agent_history_mode,
        "history_visible_fraction": float(history_stats["masked_agent_history_visible_fraction"]),
    }


def pick_agent_indices(data, max_agents):
    hist_index = 10
    valid_agents = data["agent"]["valid_mask"][:, hist_index] & (data["agent"]["type"] != 3)
    if valid_agents.any():
        agent_indices = torch.nonzero(valid_agents, as_tuple=False).squeeze(-1)
    else:
        agent_indices = torch.arange(data["agent"]["num_nodes"])

    av_index = int(data["agent"]["av_index"])
    if max_agents > 0 and agent_indices.numel() > max_agents:
        anchor = data["agent"]["position"][av_index, hist_index, :2]
        distance = torch.norm(data["agent"]["position"][agent_indices, hist_index, :2] - anchor, dim=-1)
        keep = torch.argsort(distance)[:max_agents]
        agent_indices = agent_indices[keep]
        if av_index not in agent_indices.tolist():
            agent_indices = torch.cat([torch.tensor([av_index]), agent_indices[:-1]])
    return agent_indices, av_index


def chunk_mask_to_step_mask(chunk_mask, future_mask, future_chunk_steps):
    step_mask = chunk_mask.repeat_interleave(future_chunk_steps, dim=-1)
    if step_mask.shape[-1] > future_mask.shape[-1]:
        step_mask = step_mask[:, : future_mask.shape[-1]]
    return step_mask & future_mask


def draw_map(ax, data, masked_polygon_ids):
    point_to_polygon = data[("map_point", "to", "map_polygon")]["edge_index"]
    polygon_ids = point_to_polygon[1].long()
    point_position = data["map_point"]["position"][:, :2]
    masked_polygon_ids = set(int(i) for i in masked_polygon_ids.tolist())

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
            alpha=0.95 if is_masked else 0.7,
            zorder=2 if is_masked else 1,
        )


def oriented_box(center_xy, heading, length, width):
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


def heading_triangle(center_xy, heading, length, width):
    direction = torch.tensor(
        [torch.cos(torch.tensor(heading, dtype=torch.float32)), torch.sin(torch.tensor(heading, dtype=torch.float32))]
    )
    lateral = torch.tensor([-direction[1], direction[0]])
    tip = center_xy + direction * (0.5 * length)
    base_center = center_xy
    base_left = base_center + lateral * (0.35 * width)
    base_right = base_center - lateral * (0.35 * width)
    points = [tip, base_right, base_left]
    return [(float(point[0].item()), float(point[1].item())) for point in points]


def draw_agent_shape(ax, center_xy, heading, shape, agent_type, face_color, edge_color, alpha=0.28):
    length = max(float(shape[0].item()), 0.6)
    width = max(float(shape[1].item()), 0.4)
    face_rgba = matplotlib.colors.to_rgba(face_color, alpha=alpha)

    if agent_type == 1:
        patch = Circle(
            (center_xy[0].item(), center_xy[1].item()),
            radius=0.3 * max(length, width),
            facecolor=face_rgba,
            edgecolor=edge_color,
            linewidth=1.3,
            zorder=7,
        )
        ax.add_patch(patch)
    else:
        patch = Polygon(
            oriented_box(center_xy, heading, length, width),
            closed=True,
            facecolor=face_rgba,
            edgecolor=edge_color,
            linewidth=1.3,
            zorder=7,
        )
        ax.add_patch(patch)

    arrow = Polygon(
        heading_triangle(center_xy, heading, length, width),
        closed=True,
        facecolor="none",
        edgecolor=edge_color,
        linewidth=1.0,
        alpha=0.95,
        zorder=8,
    )
    ax.add_patch(arrow)


def draw_agents(ax, sample, agent_indices, av_index, radius):
    data = sample["data"]
    hist_steps = data["agent"]["valid_mask"].shape[1] - sample["future_mask"].shape[1]
    current_step = hist_steps - 1
    step_prediction_mask = chunk_mask_to_step_mask(
        sample["agent_prediction_mask"],
        sample["future_mask"],
        future_chunk_steps=5,
    )

    anchor = data["agent"]["position"][av_index, current_step, :2]
    for agent_index in agent_indices.tolist():
        is_ego = agent_index == av_index
        is_masked = bool(sample["agent_prediction_mask"][agent_index].any())
        history_color = "#7a4bc2" if is_ego else "#9ec5fe"
        future_color = "#9e9e9e"
        masked_color = "#e15759"
        edge_color = "#7a4bc2" if is_ego else ("#e15759" if is_masked else "#000000")

        history_mask = data["agent"]["valid_mask"][agent_index, :hist_steps]
        history = data["agent"]["position"][agent_index, :hist_steps, :2][history_mask]
        future = data["agent"]["position"][agent_index, hist_steps:, :2]
        future_valid = sample["future_mask"][agent_index]
        future = future[future_valid]
        masked_future = data["agent"]["position"][agent_index, hist_steps:, :2][step_prediction_mask[agent_index]]

        if history.numel() > 0:
            ax.plot(history[:, 0], history[:, 1], color=history_color, linestyle=":", linewidth=1.7, alpha=0.95, zorder=3)
        if future.numel() > 0:
            ax.plot(future[:, 0], future[:, 1], color=future_color, linestyle="-", linewidth=1.1, alpha=0.55, zorder=3)
        if masked_future.numel() > 0:
            ax.plot(masked_future[:, 0], masked_future[:, 1], color=masked_color, linestyle="-", linewidth=2.6, alpha=0.95, zorder=5)

        current_xy = data["agent"]["position"][agent_index, current_step, :2]
        current_heading = float(data["agent"]["heading"][agent_index, current_step].item())
        shape = data["agent"]["shape"][agent_index, current_step, :2]
        agent_type = int(data["agent"]["type"][agent_index].item())
        face_color = "#8f63d2" if is_ego else ("#f6b5ae" if is_masked else "#dcecff")
        draw_agent_shape(ax, current_xy, current_heading, shape, agent_type, face_color, edge_color)

    ax.set_xlim(float(anchor[0].item() - radius), float(anchor[0].item() + radius))
    ax.set_ylim(float(anchor[1].item() - radius), float(anchor[1].item() + radius))


def render_single(sample, title, output_path, max_agents, radius, dpi):
    data = sample["data"]
    agent_indices, av_index = pick_agent_indices(data, max_agents)
    masked_polygon_ids = torch.nonzero(sample["map_prediction_mask"], as_tuple=False).squeeze(-1)

    fig, ax = plt.subplots(figsize=(9, 9))
    draw_map(ax, data, masked_polygon_ids)
    draw_agents(ax, sample, agent_indices, av_index, radius)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11, pad=10)
    ax.text(
        0.02,
        0.98,
        " | ".join([
            f"masked agents={int(sample['agent_prediction_mask'].any(dim=-1).sum())}",
            f"masked polygons={int(sample['map_prediction_mask'].sum())}",
            f"history={sample['history_mode']}",
            f"hist vis={sample['history_visible_fraction']:.2f}",
        ]),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2.0},
    )
    ax.legend(
        handles=legend_handles(),
        loc="lower left",
        fontsize=8,
        framealpha=0.92,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def render_gallery(samples, titles, output_path, max_agents, radius, dpi):
    cols = 2
    rows = (len(samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(9 * cols, 9 * rows), squeeze=False)
    for ax, sample, title in zip(axes.flat, samples, titles):
        data = sample["data"]
        agent_indices, av_index = pick_agent_indices(data, max_agents)
        masked_polygon_ids = torch.nonzero(sample["map_prediction_mask"], as_tuple=False).squeeze(-1)
        draw_map(ax, data, masked_polygon_ids)
        draw_agents(ax, sample, agent_indices, av_index, radius)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=10, pad=8)
        ax.text(
            0.02,
            0.98,
            " | ".join([
                f"agents={int(sample['agent_prediction_mask'].any(dim=-1).sum())}",
                f"polygons={int(sample['map_prediction_mask'].sum())}",
                f"history={sample['history_mode']}",
                f"hist vis={sample['history_visible_fraction']:.2f}",
            ]),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 1.8},
        )
    for ax in axes.flat[len(samples):]:
        ax.axis("off")
    if samples:
        axes.flat[0].legend(
            handles=legend_handles(),
            loc="lower left",
            fontsize=8,
            framealpha=0.92,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def legend_handles():
    return [
        Line2D([0], [0], color="#c7c7c7", lw=1.2, label="Visible Map"),
        Line2D([0], [0], color="#f28e2b", lw=2.8, label="Masked Map"),
        Line2D([0], [0], color="#9e9e9e", lw=1.2, label="Visible Future"),
        Line2D([0], [0], color="#e15759", lw=2.6, label="Masked Agent Future"),
        Line2D([0], [0], color="#7a4bc2", lw=1.8, linestyle=":", label="Ego History"),
        Line2D([0], [0], color="#9ec5fe", lw=1.8, linestyle=":", label="Other History"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#8f63d2", markeredgecolor="#7a4bc2", markersize=9, label="Ego"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#f6b5ae", markeredgecolor="#e15759", markersize=9, label="Masked Agent"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#dcecff", markeredgecolor="#000000", markersize=9, label="Other Agent"),
    ]


def main():
    args = parse_args()
    config = load_config_act(args.config)
    model = SMARTJEPA(config.Model)
    dataset = load_dataset(config, args.split)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    titles = []
    for index in args.indices:
        if index < 0 or index >= len(dataset):
            continue
        graph = dataset[index]
        scenario_id = scenario_id_from_graph(graph)
        sample = prepare_sample(model, graph)
        title = f"{args.split}[{index}] | {scenario_id}"
        render_single(
            sample,
            title=title,
            output_path=output_dir / f"{args.split}_{index:05d}_{scenario_id}.png",
            max_agents=args.max_agents,
            radius=args.radius,
            dpi=args.dpi,
        )
        samples.append(sample)
        titles.append(title)

    if samples:
        render_gallery(
            samples,
            titles,
            output_path=output_dir / f"gallery_{args.split}.png",
            max_agents=args.max_agents,
            radius=args.radius,
            dpi=args.dpi,
        )
        print(f"Saved {len(samples)} mask visualizations to {output_dir}")
    else:
        print("No valid indices to render.")


if __name__ == "__main__":
    main()
