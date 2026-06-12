import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch_geometric.data import Batch

from smart.datasets.scalable_dataset import MultiDataset
from smart.model import SMARTAutoregressiveDiffusion
from smart.transforms import WaymoTargetBuilder
from smart.utils.config import load_config_act


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


def _load_model(config, ckpt_path: str, device: torch.device) -> SMARTAutoregressiveDiffusion:
    model = SMARTAutoregressiveDiffusion(config.Model)
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
    return model


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


def _agent_seq_lookup(packed) -> Dict[int, Tuple[int, int]]:
    lookup = {}
    for _scene_idx, seq_idx, agent_indices in packed["agent_maps"]:
        for local_idx, agent_idx in enumerate(agent_indices.tolist()):
            lookup[int(agent_idx)] = (int(seq_idx), int(local_idx))
    return lookup


def _connected_map_indices(
    edge_index: torch.Tensor,
    seq_idx: int,
    node_idx: int,
    packed_len: int,
) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=edge_index.device)
    flat_node_idx = seq_idx * packed_len + node_idx
    return edge_index[0, edge_index[1] == flat_node_idx]


def _agent_chunk_indices(
    model: SMARTAutoregressiveDiffusion,
    packed,
    agent_index: int,
) -> Optional[Tuple[int, int, int]]:
    lookup = _agent_seq_lookup(packed)
    if agent_index not in lookup:
        return None
    seq_idx, local_agent_idx = lookup[agent_index]
    start = local_agent_idx * int(model.num_future_chunks)
    end = start + int(model.num_future_chunks)
    return int(seq_idx), int(start), int(end)


def _empty_positions_like(packed, cols: int = 2) -> torch.Tensor:
    token_positions = packed.get("token_positions")
    if token_positions is not None:
        return token_positions.new_zeros((0, cols))
    return torch.zeros(0, cols)


def _map_indices_for_token_geometry(
    model: SMARTAutoregressiveDiffusion,
    packed,
    agent_index: int,
    chunk_idx: int,
    token_positions: torch.Tensor,
    token_headings: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    chunk_info = _agent_chunk_indices(model, packed, agent_index)
    if chunk_info is None:
        device = packed["valid_mask"].device
        return (
            torch.empty(0, dtype=torch.long, device=device),
            _empty_positions_like(packed),
            _empty_positions_like(packed).new_zeros(2),
        )
    seq_idx, start, _end = chunk_info
    node_idx = start + int(chunk_idx)
    query_xy = token_positions[seq_idx, node_idx]
    map_positions = packed.get("map_positions")
    if map_positions is None or map_positions.numel() == 0:
        return (
            torch.empty(0, dtype=torch.long, device=packed["valid_mask"].device),
            _empty_positions_like(packed),
            query_xy.detach(),
        )
    edge_index, _ = model.diffusion_decoder._build_map2token_edges(
        token_positions,
        token_headings,
        packed["valid_mask"],
        map_positions,
        packed.get("map_orientations"),
        packed.get("map_batch"),
        packed.get("map_valid_mask"),
    )
    connected = _connected_map_indices(edge_index, seq_idx, node_idx, packed["valid_mask"].shape[1])
    connected_positions = map_positions[connected] if connected.numel() > 0 else map_positions.new_zeros((0, 2))
    return connected.detach(), connected_positions.detach(), query_xy.detach()


def _map_indices_for_packed_geometry(
    model: SMARTAutoregressiveDiffusion,
    packed,
    agent_index: int,
    chunk_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _map_indices_for_token_geometry(
        model,
        packed,
        agent_index,
        chunk_idx,
        packed["token_positions"],
        packed["token_headings"],
    )


def _proposal_refreshed_geometry(
    model: SMARTAutoregressiveDiffusion,
    packed,
    initial_proposal_ids: Optional[torch.Tensor],
    initial_proposal_confidence: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not hasattr(model, "_refresh_token_geometry"):
        return (
            packed["token_positions"].clone(),
            packed["token_headings"].clone(),
            packed["valid_mask"].new_zeros(packed["valid_mask"].shape, dtype=torch.float),
        )
    mask_token_id = int(getattr(model, "mask_token_id", 0))
    masked_ids = torch.full_like(packed["valid_mask"].long(), mask_token_id)
    geometry_known = packed["valid_mask"] & (masked_ids != mask_token_id)
    use_proposal_geometry = bool(getattr(model, "use_proposal_geometry", True))
    proposal_ids = initial_proposal_ids if use_proposal_geometry else None
    proposal_confidence = initial_proposal_confidence if use_proposal_geometry else None
    return model._refresh_token_geometry(
        masked_ids,
        packed,
        geometry_known_mask=geometry_known,
        proposal_token_ids=proposal_ids,
        proposal_confidence=proposal_confidence,
    )


def _proposal_values_for_agent(
    values: Optional[torch.Tensor],
    model: SMARTAutoregressiveDiffusion,
    packed,
    agent_index: int,
) -> List[float]:
    if values is None:
        return []
    chunk_info = _agent_chunk_indices(model, packed, agent_index)
    if chunk_info is None:
        return []
    seq_idx, start, end = chunk_info
    return values[seq_idx, start:end].detach().cpu().tolist()


def _rollout_input_snapshot(
    model: SMARTAutoregressiveDiffusion,
    packed,
    agent_index: int,
    history_frame_pos: torch.Tensor,
    history_frame_valid: torch.Tensor,
    generation_agents: torch.Tensor,
    initial_proposal_ids: Optional[torch.Tensor],
    initial_proposal_confidence: Optional[torch.Tensor],
) -> Dict:
    current_agent_mask = history_frame_valid[:, -1].bool() & generation_agents.bool()
    input_agent_indices = torch.nonzero(current_agent_mask, as_tuple=False).squeeze(-1)
    input_agent_positions = history_frame_pos[input_agent_indices, -1, :2]
    selected_history_valid = history_frame_valid[agent_index].bool()
    selected_history = history_frame_pos[agent_index, :, :2][selected_history_valid]
    chunk_info = _agent_chunk_indices(model, packed, agent_index)
    if chunk_info is None:
        input_query_positions = _empty_positions_like(packed)
        input_query_valid = torch.empty(0, dtype=torch.bool, device=history_frame_pos.device)
    else:
        seq_idx, start, end = chunk_info
        input_query_positions = packed["token_positions"][seq_idx, start:end]
        input_query_valid = packed["valid_mask"][seq_idx, start:end]
    input_connected_ids, input_connected_positions, input_query_xy = _map_indices_for_packed_geometry(
        model,
        packed,
        agent_index,
        chunk_idx=0,
    )
    proposal_positions, proposal_headings, proposal_confidence = _proposal_refreshed_geometry(
        model,
        packed,
        initial_proposal_ids,
        initial_proposal_confidence,
    )
    if chunk_info is None:
        proposal_query_positions = _empty_positions_like(packed)
        proposal_query_valid = torch.empty(0, dtype=torch.bool, device=history_frame_pos.device)
        proposal_geometry_confidence = torch.empty(0, dtype=torch.float32, device=history_frame_pos.device)
    else:
        seq_idx, start, end = chunk_info
        proposal_query_positions = proposal_positions[seq_idx, start:end]
        proposal_query_valid = packed["valid_mask"][seq_idx, start:end]
        proposal_geometry_confidence = proposal_confidence[seq_idx, start:end]
    proposal_connected_ids, proposal_connected_positions, proposal_query_xy = _map_indices_for_token_geometry(
        model,
        packed,
        agent_index,
        chunk_idx=0,
        token_positions=proposal_positions,
        token_headings=proposal_headings,
    )
    map_positions = packed.get("map_positions")
    if map_positions is None:
        map_positions = _empty_positions_like(packed)
    return {
        "input_current_xy": history_frame_pos[agent_index, -1, :2].detach().cpu(),
        "input_query_xy": input_query_xy.detach().cpu(),
        "input_query_positions": input_query_positions.detach().cpu(),
        "input_query_valid": input_query_valid.detach().cpu(),
        "proposal_query_xy": proposal_query_xy.detach().cpu(),
        "proposal_query_positions": proposal_query_positions.detach().cpu(),
        "proposal_query_valid": proposal_query_valid.detach().cpu(),
        "proposal_geometry_confidence": proposal_geometry_confidence.detach().cpu(),
        "input_agent_indices": input_agent_indices.detach().cpu().tolist(),
        "input_agent_positions": input_agent_positions.detach().cpu(),
        "num_input_agents": int(input_agent_indices.numel()),
        "selected_history": selected_history.detach().cpu(),
        "input_map_positions": map_positions.detach().cpu(),
        "input_connected_map_positions": input_connected_positions.detach().cpu(),
        "input_connected_map_indices": input_connected_ids.detach().cpu().tolist(),
        "input_num_connected": int(input_connected_ids.numel()),
        "proposal_connected_map_positions": proposal_connected_positions.detach().cpu(),
        "proposal_connected_map_indices": proposal_connected_ids.detach().cpu().tolist(),
        "proposal_num_connected": int(proposal_connected_ids.numel()),
        "num_map_tokens": int(map_positions.shape[0]),
        "input_proposal_token_ids": _proposal_values_for_agent(initial_proposal_ids, model, packed, agent_index),
        "input_proposal_confidence": _proposal_values_for_agent(initial_proposal_confidence, model, packed, agent_index),
    }


def _map_indices_for_agent_chunk(
    model: SMARTAutoregressiveDiffusion,
    packed,
    token_ids: torch.Tensor,
    agent_index: int,
    chunk_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    chunk_info = _agent_chunk_indices(model, packed, agent_index)
    if chunk_info is None:
        return (
            torch.empty(0, dtype=torch.long, device=token_ids.device),
            _empty_positions_like(packed),
            _empty_positions_like(packed).new_zeros(2),
        )
    seq_idx, start, _end = chunk_info
    node_idx = start + int(chunk_idx)
    geometry_known = packed["valid_mask"] & (token_ids != model.mask_token_id)
    token_positions, token_headings, _ = model._refresh_token_geometry(
        token_ids,
        packed,
        geometry_known_mask=geometry_known,
    )
    map_positions = packed.get("map_positions")
    if map_positions is None or map_positions.numel() == 0:
        return (
            torch.empty(0, dtype=torch.long, device=token_ids.device),
            _empty_positions_like(packed),
            token_positions[seq_idx, node_idx].detach(),
        )
    edge_index, _ = model.diffusion_decoder._build_map2token_edges(
        token_positions,
        token_headings,
        packed["valid_mask"],
        map_positions,
        packed.get("map_orientations"),
        packed.get("map_batch"),
        packed.get("map_valid_mask"),
    )
    connected = _connected_map_indices(edge_index, seq_idx, node_idx, packed["valid_mask"].shape[1])
    connected_positions = map_positions[connected] if connected.numel() > 0 else map_positions.new_zeros((0, 2))
    return connected.detach(), connected_positions.detach(), token_positions[seq_idx, node_idx].detach()


def _pick_vehicle_agent(data, generation_agents: torch.Tensor) -> Optional[int]:
    agent = data["agent"]
    current_valid = agent["valid_mask"][:, data["agent"]["position"].shape[1] - agent["position"].shape[1]].bool()
    del current_valid  # the generation mask already encodes current validity after preparation.
    agent_type = agent["type"].long()
    category = agent["category"].long() if "category" in agent else torch.zeros_like(agent_type)
    candidates = torch.nonzero(generation_agents & (agent_type == 0) & (category == 3), as_tuple=False).squeeze(-1)
    if candidates.numel() == 0:
        candidates = torch.nonzero(generation_agents & (agent_type == 0), as_tuple=False).squeeze(-1)
    if candidates.numel() == 0:
        candidates = torch.nonzero(generation_agents, as_tuple=False).squeeze(-1)
    if candidates.numel() == 0:
        return None
    return int(candidates[0].item())


def _draw_base_map(ax, data) -> None:
    if ("map_point", "to", "map_polygon") not in data.edge_types:
        return
    point_to_polygon = data[("map_point", "to", "map_polygon")]["edge_index"]
    polygon_ids = point_to_polygon[1].long()
    point_position = data["map_point"]["position"][:, :2]
    for polygon_id in torch.unique(polygon_ids).tolist():
        indices = point_to_polygon[0, polygon_ids == polygon_id]
        if indices.numel() < 2:
            continue
        points = point_position.index_select(0, indices)
        ax.plot(points[:, 0], points[:, 1], color="#d0d0d0", linewidth=0.6, alpha=0.65, zorder=1)


def _plot_round_grid(
    output_path: Path,
    data_cpu,
    records: Sequence[Dict],
    agent_index: int,
    scenario_id: str,
    window_m: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = 4
    rows = max(1, (len(records) + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 4.6 * rows), squeeze=False)
    agent_pos_all = data_cpu["agent"]["position"]
    agent_valid_all = data_cpu["agent"]["valid_mask"]
    current_step = 10
    gt_future = agent_pos_all[agent_index, current_step + 1 :, :2][agent_valid_all[agent_index, current_step + 1 :]]

    for flat_idx, ax in enumerate(axes.reshape(-1)):
        if flat_idx >= len(records):
            ax.set_axis_off()
            continue
        rec = records[flat_idx]
        current_xy = rec["input_current_xy"]
        map_positions = rec["input_map_positions"]
        input_connected_positions = rec["input_connected_map_positions"]
        proposal_connected_positions = rec["proposal_connected_map_positions"]
        sampled_connected_positions = rec["sampled_connected_map_positions"]
        input_agent_positions = rec["input_agent_positions"]
        selected_history = rec["selected_history"]
        input_query_positions = rec["input_query_positions"]
        input_query_valid = rec["input_query_valid"]
        proposal_query_positions = rec["proposal_query_positions"]
        proposal_query_valid = rec["proposal_query_valid"]
        proposal_geometry_confidence = rec["proposal_geometry_confidence"]
        sampled_query_positions = rec["sampled_query_positions"]
        sampled_query_valid = rec["sampled_query_valid"]
        pred_so_far = rec["pred_so_far"]
        commit_traj = rec["commit_traj"]

        _draw_base_map(ax, data_cpu)
        if map_positions.numel() > 0:
            ax.scatter(map_positions[:, 0], map_positions[:, 1], s=3, color="#b7b7b7", alpha=0.16, zorder=2)
        if input_connected_positions.numel() > 0:
            ax.scatter(
                input_connected_positions[:, 0],
                input_connected_positions[:, 1],
                s=18,
                color="#ff9f1c",
                alpha=0.82,
                edgecolors="none",
                zorder=5,
                label="input map edges",
            )
        if proposal_connected_positions.numel() > 0:
            ax.scatter(
                proposal_connected_positions[:, 0],
                proposal_connected_positions[:, 1],
                s=22,
                color="#0f766e",
                alpha=0.72,
                marker="D",
                edgecolors="none",
                zorder=6,
                label="proposal map edges",
            )
        if sampled_connected_positions.numel() > 0:
            ax.scatter(
                sampled_connected_positions[:, 0],
                sampled_connected_positions[:, 1],
                s=24,
                color="#7b2cbf",
                alpha=0.7,
                marker="x",
                linewidths=0.8,
                zorder=7,
                label="sampled map edges",
            )
        if input_agent_positions.numel() > 0:
            ax.scatter(
                input_agent_positions[:, 0],
                input_agent_positions[:, 1],
                s=14,
                color="#5f6c72",
                alpha=0.55,
                zorder=4,
                label="rolled input agents",
            )
        if selected_history.numel() > 0:
            ax.plot(
                selected_history[:, 0],
                selected_history[:, 1],
                color="#1f77b4",
                linewidth=1.8,
                linestyle=":",
                zorder=7,
                label="rolled selected history",
            )
        if gt_future.numel() > 0:
            ax.plot(gt_future[:, 0], gt_future[:, 1], color="#2ca02c", linewidth=1.4, alpha=0.75, zorder=7, label="GT future")
        if pred_so_far.numel() > 0:
            ax.plot(pred_so_far[:, 0], pred_so_far[:, 1], color="#d62728", linewidth=1.8, alpha=0.85, zorder=8, label="pred so far")
        if commit_traj.numel() > 0:
            ax.plot(commit_traj[:, 0], commit_traj[:, 1], color="#d62728", linewidth=3.0, alpha=0.95, zorder=9)
        if input_query_positions.numel() > 0:
            valid_query = input_query_positions[input_query_valid.bool()]
            if valid_query.numel() > 0:
                ax.scatter(
                    valid_query[:, 0],
                    valid_query[:, 1],
                    s=30,
                    color="none",
                    marker="o",
                    edgecolors="#64748b",
                    linewidths=1.1,
                    zorder=10,
                    label="raw packed q",
                )
                for query_idx, xy in enumerate(input_query_positions):
                    if bool(input_query_valid[query_idx]):
                        ax.text(float(xy[0]), float(xy[1]), f"r{query_idx}", fontsize=7, color="#475569", zorder=11)
        if proposal_query_positions.numel() > 0:
            valid_query = proposal_query_positions[proposal_query_valid.bool()]
            if valid_query.numel() > 0:
                ax.scatter(valid_query[:, 0], valid_query[:, 1], s=42, color="#0f766e", marker="^", zorder=12, label="proposal q")
                for query_idx, xy in enumerate(proposal_query_positions):
                    if bool(proposal_query_valid[query_idx]):
                        conf = float(proposal_geometry_confidence[query_idx]) if proposal_geometry_confidence.numel() > query_idx else 0.0
                        ax.text(float(xy[0]), float(xy[1]), f"p{query_idx}:{conf:.1f}", fontsize=7, color="#064e3b", zorder=13)
        if sampled_query_positions.numel() > 0:
            valid_query = sampled_query_positions[sampled_query_valid.bool()]
            if valid_query.numel() > 0:
                ax.scatter(valid_query[:, 0], valid_query[:, 1], s=34, color="#7b2cbf", marker="x", linewidths=1.0, zorder=14, label="sampled q")
                for query_idx, xy in enumerate(sampled_query_positions):
                    if bool(sampled_query_valid[query_idx]):
                        ax.text(float(xy[0]), float(xy[1]), f"s{query_idx}", fontsize=7, color="#4c1d95", zorder=15)
        ax.scatter([current_xy[0]], [current_xy[1]], s=90, color="#000000", marker="*", zorder=10, label="selected vehicle")
        ax.set_xlim(float(current_xy[0] - window_m), float(current_xy[0] + window_m))
        ax.set_ylim(float(current_xy[1] - window_m), float(current_xy[1] + window_m))
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle=":", alpha=0.25)
        ax.set_title(
            f"round {rec['round'] + 1:02d} | token={rec['token_id']} | conf={rec['confidence']:.2f}\n"
            f"raw={rec['input_num_connected']} prop={rec['proposal_num_connected']} "
            f"samp={rec['sampled_num_connected']} agents={rec['num_input_agents']} map={rec['num_map_tokens']}"
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="lower center", ncol=8, fontsize=9)
    fig.suptitle(f"AR raw/proposal/sampled query debug | scenario={scenario_id} | agent={agent_index}", fontsize=14)
    fig.tight_layout(rect=(0, 0.04, 1, 0.965))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def _trace_one_scene(
    model: SMARTAutoregressiveDiffusion,
    graph,
    device: torch.device,
    max_rounds: Optional[int],
) -> Optional[Tuple[object, int, List[Dict], Dict]]:
    batch = Batch.from_data_list([graph]).to(device)
    data = model._prepare_batch(batch)
    generation_agents = model._generation_agent_mask(data)
    agent_index = _pick_vehicle_agent(data, generation_agents)
    if agent_index is None:
        return None

    num_agents = int(data["agent"]["position"].shape[0])
    rounds = model._num_ar_rollout_rounds()
    if max_rounds is not None:
        rounds = min(rounds, int(max_rounds))
    history_token_ids = data["agent"]["token_idx"][:, : model.ar_history_tokens].long().clone()
    history_token_valid = data["agent"]["agent_valid_mask"][:, : model.ar_history_tokens].bool().clone()
    history_token_pos = data["agent"]["token_pos"][:, : model.ar_history_tokens, :2].float().clone()
    history_token_heading = data["agent"]["token_heading"][:, : model.ar_history_tokens].float().clone()
    history_frame_pos = data["agent"]["position"][:, : model.num_historical_steps, :2].float().clone()
    history_frame_heading = data["agent"]["heading"][:, : model.num_historical_steps].float().clone()
    history_frame_valid = data["agent"]["valid_mask"][:, : model.num_historical_steps].bool().clone()
    current_pos = history_frame_pos[:, -1].clone()
    current_heading = history_frame_heading[:, -1].clone()
    carried_proposal_ids = None
    carried_proposal_confidence = None
    selected_pred_chunks: List[torch.Tensor] = []
    records: List[Dict] = []

    for round_idx in range(rounds):
        rollout_view = model._build_ar_rollout_view(
            data,
            history_token_ids,
            history_token_pos,
            history_token_heading,
            history_frame_pos,
            history_frame_heading,
            history_frame_valid,
            generation_agents,
            history_token_valid=history_token_valid,
        )
        packed, summary, _ft, fv, _generation_agents, _supervision_agents, _agent_batch = model._build_diffusion_inputs(rollout_view)
        if packed is None:
            break
        initial_proposal_ids = None
        initial_proposal_confidence = None
        if model.ar_carry_tail_proposal and carried_proposal_ids is not None and carried_proposal_confidence is not None:
            initial_proposal_ids = model._pack_agent_window_values(carried_proposal_ids, packed, fill_value=0)
            initial_proposal_confidence = model._pack_agent_window_values(carried_proposal_confidence, packed, fill_value=0.0)

        input_snapshot = _rollout_input_snapshot(
            model=model,
            packed=packed,
            agent_index=agent_index,
            history_frame_pos=history_frame_pos,
            history_frame_valid=history_frame_valid,
            generation_agents=generation_agents,
            initial_proposal_ids=initial_proposal_ids,
            initial_proposal_confidence=initial_proposal_confidence,
        )

        sampled_ids, sampled_confidence = model._diffusion_sample(
            summary=summary,
            token_positions=packed["token_positions"],
            token_headings=packed["token_headings"],
            token_agent_ids=packed["token_agent_ids"],
            chunk_ids=packed["chunk_ids"],
            valid_mask=packed["valid_mask"],
            agent_context=packed["agent_context"],
            agent_type_ids=packed["agent_type_ids"],
            agent_shape_embeddings=packed["agent_shape_embeddings"],
            map_context=packed.get("map_context"),
            map_positions=packed.get("map_positions"),
            map_orientations=packed.get("map_orientations"),
            map_batch=packed.get("map_batch"),
            map_valid_mask=packed.get("map_valid_mask"),
            packed=packed,
            initial_proposal_token_ids=initial_proposal_ids,
            initial_proposal_confidence=initial_proposal_confidence,
        )

        sampled_connected_ids, sampled_connected_map_positions, sampled_query_xy = _map_indices_for_agent_chunk(
            model,
            packed,
            sampled_ids,
            agent_index,
            chunk_idx=0,
        )
        sampled_geometry_known = packed["valid_mask"] & (sampled_ids != model.mask_token_id)
        sampled_token_positions, _sampled_token_headings, sampled_geometry_confidence = model._refresh_token_geometry(
            sampled_ids,
            packed,
            geometry_known_mask=sampled_geometry_known,
        )
        chunk_info = _agent_chunk_indices(model, packed, agent_index)
        if chunk_info is None:
            sampled_query_positions = _empty_positions_like(packed)
            sampled_query_valid = torch.empty(0, dtype=torch.bool, device=sampled_ids.device)
            sampled_query_confidence = torch.empty(0, dtype=torch.float32, device=sampled_ids.device)
        else:
            seq_idx, start, end = chunk_info
            sampled_query_positions = sampled_token_positions[seq_idx, start:end]
            sampled_query_valid = packed["valid_mask"][seq_idx, start:end]
            sampled_query_confidence = sampled_geometry_confidence[seq_idx, start:end]
        per_agent_tokens, per_agent_confidence = model._unpack_sampled_tokens(
            sampled_ids,
            sampled_confidence,
            packed,
            num_agents,
        )
        committed_tokens = per_agent_tokens[:, : model.ar_commit_tokens]
        committed_confidence = per_agent_confidence[:, : model.ar_commit_tokens]
        committed_valid = fv[:, : model.ar_commit_tokens].bool() & generation_agents[:, None]
        commit_traj, commit_head, commit_valid_frames, commit_token_pos, commit_token_heading, current_pos, current_heading = model._decode_token_sequence(
            committed_tokens,
            committed_valid,
            data["agent"]["type"],
            current_pos,
            current_heading,
        )
        selected_commit = commit_traj[agent_index].detach().cpu()
        selected_pred_chunks.append(selected_commit)
        pred_so_far = torch.cat(selected_pred_chunks, dim=0)
        record = dict(input_snapshot)
        record.update(
            {
                "round": round_idx,
                "token_id": int(committed_tokens[agent_index, 0].item()),
                "confidence": float(committed_confidence[agent_index, 0].item()),
                "sampled_query_xy": sampled_query_xy.detach().cpu(),
                "sampled_query_positions": sampled_query_positions.detach().cpu(),
                "sampled_query_valid": sampled_query_valid.detach().cpu(),
                "sampled_geometry_confidence": sampled_query_confidence.detach().cpu(),
                "sampled_connected_map_positions": sampled_connected_map_positions.detach().cpu(),
                "sampled_connected_map_indices": sampled_connected_ids.detach().cpu().tolist(),
                "sampled_num_connected": int(sampled_connected_ids.numel()),
                "commit_traj": selected_commit,
                "pred_so_far": pred_so_far,
            }
        )
        records.append(record)

        if model.ar_carry_tail_proposal:
            carried_proposal_ids, carried_proposal_confidence = model._next_tail_proposal(
                per_agent_tokens,
                per_agent_confidence,
                fv,
                generation_agents,
            )
        history_token_ids = model._roll_history_token_ids(history_token_ids, committed_tokens)
        history_token_valid = model._roll_history_token_valid(history_token_valid, committed_valid)
        history_token_pos = model._roll_history_token_state(history_token_pos, commit_token_pos)
        history_token_heading = model._roll_history_token_state(history_token_heading, commit_token_heading)
        history_frame_pos = torch.cat([history_frame_pos, commit_traj], dim=1)[:, -model.num_historical_steps :]
        history_frame_heading = torch.cat([history_frame_heading, commit_head], dim=1)[:, -model.num_historical_steps :]
        history_frame_valid = torch.cat([history_frame_valid, commit_valid_frames], dim=1)[:, -model.num_historical_steps :]

    metadata = {
        "agent_index": agent_index,
        "agent_type": int(data["agent"]["type"][agent_index].item()),
        "agent_category": int(data["agent"]["category"][agent_index].item()),
        "rounds": len(records),
        "tokens": [rec["token_id"] for rec in records],
        "input_map_edges_per_round": [rec["input_num_connected"] for rec in records],
        "proposal_map_edges_per_round": [rec["proposal_num_connected"] for rec in records],
        "sampled_map_edges_per_round": [rec["sampled_num_connected"] for rec in records],
        "map_edges_per_round": [rec["sampled_num_connected"] for rec in records],
        "map_tokens_per_round": [rec["num_map_tokens"] for rec in records],
        "input_agents_per_round": [rec["num_input_agents"] for rec in records],
        "proposal_tokens_per_round": [rec["input_proposal_token_ids"] for rec in records],
        "proposal_geometry_confidence_per_round": [rec["proposal_geometry_confidence"].tolist() for rec in records],
        "sampled_geometry_confidence_per_round": [rec["sampled_geometry_confidence"].tolist() for rec in records],
    }
    return data.cpu(), agent_index, records, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize AR rollout inputs and sampled map tokens for one vehicle per scene.")
    parser.add_argument("--config", default="configs/validation/validation_scalable_ar_diffusion.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--num-scenes", type=int, default=3)
    parser.add_argument("--sample-indices", default="")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-rounds", type=int, default=16)
    parser.add_argument("--window-m", type=float, default=45.0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", default="outputs/ar_map_rollout_debug")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    config = load_config_act(args.config)
    dataset = _load_dataset(config, args.split)
    model = _load_model(config, args.ckpt, device)
    output_dir = Path(args.output_dir)
    rng = random.Random(args.seed)
    if args.sample_indices:
        candidate_indices = [int(x.strip()) for x in args.sample_indices.split(",") if x.strip()]
    else:
        candidate_indices = list(range(len(dataset)))
        rng.shuffle(candidate_indices)

    written = []
    metadata = {
        "checkpoint": args.ckpt,
        "config": args.config,
        "device": str(device),
        "scenes": [],
    }
    for sample_index in candidate_indices:
        if len(written) >= args.num_scenes:
            break
        graph = dataset[sample_index]
        scenario = _scenario_id(graph)
        traced = _trace_one_scene(model, graph, device=device, max_rounds=args.max_rounds)
        if traced is None:
            continue
        data_cpu, agent_index, records, scene_meta = traced
        if not records:
            continue
        output_path = output_dir / f"idx_{sample_index:05d}_{scenario}_agent_{agent_index:03d}_ar_query_debug.png"
        _plot_round_grid(
            output_path=output_path,
            data_cpu=data_cpu,
            records=records,
            agent_index=agent_index,
            scenario_id=scenario,
            window_m=args.window_m,
        )
        scene_meta.update({
            "sample_index": sample_index,
            "scenario_id": scenario,
            "image": str(output_path),
        })
        metadata["scenes"].append(scene_meta)
        written.append(output_path)
        print(f"[written] {output_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Wrote {len(written)} AR map rollout visualizations to {output_dir}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
