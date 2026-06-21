import argparse
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch_geometric.data import Batch

from smart.datasets.scalable_dataset import MultiDataset
from smart.model import SMARTDiffusion
from smart.transforms import WaymoTargetBuilder
from smart.utils.config import load_config_act


def _parse_chunk_ids(value: str, num_chunks: int) -> List[int]:
    if value.strip().lower() == "all":
        return list(range(num_chunks))
    chunks = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        chunk = int(raw)
        if 0 <= chunk < num_chunks:
            chunks.append(chunk)
    seen = set()
    return [chunk for chunk in chunks if not (chunk in seen or seen.add(chunk))]


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


def _load_dataset(config, split: str) -> MultiDataset:
    data_config = config.Dataset
    return MultiDataset(
        root=data_config.root,
        split=split,
        raw_dir=getattr(data_config, f"{split}_raw_dir"),
        processed_dir=getattr(data_config, f"{split}_processed_dir", None),
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


def _load_model(config, ckpt_path: str) -> SMARTDiffusion:
    model = SMARTDiffusion(config.Model)
    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def _draw_base_map(ax, data) -> None:
    point_to_polygon = data[("map_point", "to", "map_polygon")]["edge_index"]
    polygon_ids = point_to_polygon[1].long()
    point_position = data["map_point"]["position"][:, :2]
    for polygon_id in torch.unique(polygon_ids).tolist():
        indices = point_to_polygon[0, polygon_ids == polygon_id]
        if indices.numel() < 2:
            continue
        points = point_position.index_select(0, indices)
        ax.plot(
            points[:, 0],
            points[:, 1],
            color="#c8c8c8",
            linewidth=0.8,
            alpha=0.75,
            zorder=1,
        )


def _agent_seq_lookup(packed) -> Dict[int, Tuple[int, int]]:
    lookup = {}
    for _scene_idx, seq_idx, agent_indices in packed["agent_maps"]:
        for local_idx, agent_idx in enumerate(agent_indices.tolist()):
            lookup[int(agent_idx)] = (int(seq_idx), int(local_idx))
    return lookup


def _map_edges_for_geometry(model, packed, token_ids) -> Tuple[torch.Tensor, torch.Tensor]:
    token_positions, token_headings, _geometry_confidence = model._refresh_token_geometry(token_ids, packed)
    edge_index, _ = model.diffusion_decoder._build_map2token_edges(
        token_positions,
        token_headings,
        packed["valid_mask"],
        packed.get("map_positions"),
        packed.get("map_orientations"),
        packed.get("map_batch"),
        packed.get("map_valid_mask"),
    )
    return edge_index, token_positions.detach().cpu()


def _connected_map_indices(
    edge_index: torch.Tensor,
    seq_idx: int,
    node_idx: int,
    packed_len: int,
) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    flat_node_idx = seq_idx * packed_len + node_idx
    return edge_index[0, edge_index[1] == flat_node_idx].detach().cpu()


def _cpu_packed(packed) -> Dict:
    return {
        key: value.detach().cpu() if torch.is_tensor(value) else value
        for key, value in packed.items()
    }


def _plot_agent_map_tokens(
    output_path: Path,
    data,
    packed,
    model: SMARTDiffusion,
    agent_index: int,
    scenario_id: str,
    chunk_ids: Sequence[int],
    window_m: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seq_lookup = _agent_seq_lookup(packed)
    if agent_index not in seq_lookup:
        return
    seq_idx, local_agent_idx = seq_lookup[agent_index]
    num_chunks = model.num_future_chunks
    packed_len = packed["valid_mask"].shape[1]
    all_mask_tokens = torch.full_like(packed["token_ids"], model.mask_token_id)
    gt_tokens = packed["token_ids"]
    all_mask_edges, all_mask_positions = _map_edges_for_geometry(model, packed, all_mask_tokens)
    gt_edges, gt_positions = _map_edges_for_geometry(model, packed, gt_tokens)

    current_step = model.num_historical_steps - 1
    hist_mask = data["agent"]["valid_mask"][agent_index, : model.num_historical_steps]
    history = data["agent"]["position"][agent_index, : model.num_historical_steps, :2][hist_mask]
    future_mask = data["agent"]["valid_mask"][agent_index, model.num_historical_steps :]
    future = data["agent"]["position"][agent_index, model.num_historical_steps :, :2][future_mask]
    current_xy = data["agent"]["position"][agent_index, current_step, :2]

    map_positions = packed.get("map_positions")
    if map_positions is None:
        return
    map_positions_cpu = map_positions.detach().cpu()

    fig, axes = plt.subplots(
        1,
        len(chunk_ids),
        figsize=(5.2 * len(chunk_ids), 5.2),
        squeeze=False,
    )
    colors = {
        "all_mask": "#ff9f1c",
        "gt": "#2ec4b6",
        "history": "#1f77b4",
        "future": "#2ca02c",
    }
    for ax, chunk_id in zip(axes[0], chunk_ids):
        node_idx = local_agent_idx * num_chunks + int(chunk_id)
        if not bool(packed["valid_mask"][seq_idx, node_idx].item()):
            ax.set_axis_off()
            continue
        all_mask_map = _connected_map_indices(all_mask_edges, seq_idx, node_idx, packed_len)
        gt_map = _connected_map_indices(gt_edges, seq_idx, node_idx, packed_len)

        _draw_base_map(ax, data)
        ax.scatter(
            map_positions_cpu[:, 0],
            map_positions_cpu[:, 1],
            s=5,
            color="#b0b0b0",
            alpha=0.18,
            zorder=2,
            label="packed map tokens",
        )
        if all_mask_map.numel() > 0:
            pts = map_positions_cpu[all_mask_map]
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                s=18,
                color=colors["all_mask"],
                alpha=0.72,
                zorder=4,
                label="all-mask edges",
            )
        if gt_map.numel() > 0:
            pts = map_positions_cpu[gt_map]
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                s=10,
                color=colors["gt"],
                alpha=0.78,
                zorder=5,
                label="GT-geometry edges",
            )
        if history.numel() > 0:
            ax.plot(
                history[:, 0],
                history[:, 1],
                color=colors["history"],
                linewidth=2.0,
                linestyle=":",
                zorder=7,
                label="history",
            )
        if future.numel() > 0:
            ax.plot(
                future[:, 0],
                future[:, 1],
                color=colors["future"],
                linewidth=1.8,
                alpha=0.82,
                zorder=6,
                label="GT future",
            )
        all_mask_xy = all_mask_positions[seq_idx, node_idx]
        gt_xy = gt_positions[seq_idx, node_idx]
        ax.scatter(
            [float(all_mask_xy[0])],
            [float(all_mask_xy[1])],
            s=78,
            marker="P",
            color=colors["all_mask"],
            edgecolors="#000000",
            linewidths=0.7,
            zorder=8,
            label="all-mask query pose",
        )
        ax.scatter(
            [float(gt_xy[0])],
            [float(gt_xy[1])],
            s=80,
            marker="X",
            color="#d62728",
            linewidths=2.2,
            zorder=9,
            label="GT-geometry query pose",
        )
        ax.scatter(
            [float(current_xy[0])],
            [float(current_xy[1])],
            s=48,
            marker="o",
            color="#000000",
            zorder=10,
            label="current pose",
        )
        ax.set_title(
            f"agent={agent_index} chunk={chunk_id}\n"
            f"all-mask={int(all_mask_map.numel())}, gt={int(gt_map.numel())}"
        )
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle=":", alpha=0.25)
        ax.set_xlim(float(current_xy[0] - window_m), float(current_xy[0] + window_m))
        ax.set_ylim(float(current_xy[1] - window_m), float(current_xy[1] + window_m))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="lower center", ncol=5, fontsize=8)
    fig.suptitle(f"SMART-Diffusion map-token context | scenario={scenario_id}", fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _choose_agents(packed, rng: random.Random, agents_per_sample: int) -> List[int]:
    candidates = []
    for _scene_idx, _seq_idx, agent_indices in packed["agent_maps"]:
        candidates.extend(int(agent_idx) for agent_idx in agent_indices.tolist())
    rng.shuffle(candidates)
    return candidates[:agents_per_sample]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize map tokens connected to randomly selected SMART-Diffusion agents."
    )
    parser.add_argument("--config", default="configs/validation/validation_scalable_diffusion.yaml")
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--agents-per-sample", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--chunk-ids", default="0,1,4,8,12,15")
    parser.add_argument("--window-m", type=float, default=70.0)
    parser.add_argument("--output-dir", default="outputs/diffusion_map_token_debug")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    config = load_config_act(args.config)
    dataset = _load_dataset(config, args.split)
    model = _load_model(config, args.ckpt).cpu()
    chunk_ids = _parse_chunk_ids(args.chunk_ids, model.num_future_chunks)
    output_dir = Path(args.output_dir)

    sample_indices = rng.sample(range(len(dataset)), k=min(args.num_samples, len(dataset)))
    written = []
    with torch.no_grad():
        for sample_index in sample_indices:
            graph = dataset[sample_index]
            scenario = _scenario_id(graph)
            batch = Batch.from_data_list([graph])
            prepared = model._prepare_batch(batch)
            packed, _summary, *_ = model._build_diffusion_inputs(prepared)
            if packed is None or packed.get("map_positions") is None:
                continue
            prepared_cpu = prepared.cpu()
            packed_cpu = _cpu_packed(packed)
            for agent_index in _choose_agents(packed_cpu, rng, args.agents_per_sample):
                output_path = output_dir / f"idx_{sample_index:05d}_{scenario}_agent_{agent_index:03d}.png"
                _plot_agent_map_tokens(
                    output_path=output_path,
                    data=prepared_cpu,
                    packed=packed_cpu,
                    model=model,
                    agent_index=agent_index,
                    scenario_id=scenario,
                    chunk_ids=chunk_ids,
                    window_m=args.window_m,
                )
                written.append(output_path)
    for path in written:
        print(path)
    print(f"Wrote {len(written)} map-token diagnostic images to {output_dir}")


if __name__ == "__main__":
    main()
