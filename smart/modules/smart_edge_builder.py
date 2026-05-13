from typing import Optional

import torch
from torch_cluster import radius, radius_graph
from torch_geometric.utils import subgraph

from smart.utils import angle_between_2d_vectors, wrap_angle


def empty_edge_index(device: torch.device) -> torch.Tensor:
    return torch.empty(2, 0, dtype=torch.long, device=device)


def build_temporal_edges_from_flat(
    pos_s: torch.Tensor,
    head_s: torch.Tensor,
    head_vector_s: torch.Tensor,
    agent_ids: torch.Tensor,
    step_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    max_step_delta: Optional[int],
    causal: bool,
    target_mask: Optional[torch.Tensor] = None,
):
    device = pos_s.device
    if target_mask is None:
        target_mask = valid_mask
    valid_agents = torch.unique(agent_ids[valid_mask | target_mask])
    src_parts = []
    dst_parts = []
    for agent_id in valid_agents:
        src_nodes = torch.nonzero(valid_mask & (agent_ids == agent_id), as_tuple=False).squeeze(-1)
        dst_nodes = torch.nonzero(target_mask & (agent_ids == agent_id), as_tuple=False).squeeze(-1)
        if src_nodes.numel() == 0 or dst_nodes.numel() == 0:
            continue
        src = src_nodes.repeat_interleave(dst_nodes.numel())
        dst = dst_nodes.repeat(src_nodes.numel())
        delta = step_ids[dst] - step_ids[src]
        keep = src != dst
        if causal:
            keep = keep & (delta > 0)
        if max_step_delta is not None:
            keep = keep & (delta.abs() <= max_step_delta)
        src_parts.append(src[keep])
        dst_parts.append(dst[keep])

    if not src_parts:
        return empty_edge_index(device), pos_s.new_zeros((0, 4))

    edge_index = torch.stack([torch.cat(src_parts), torch.cat(dst_parts)], dim=0)
    src, dst = edge_index
    rel_pos = pos_s[src] - pos_s[dst]
    rel_head = wrap_angle(head_s[src] - head_s[dst])
    rel_step = (step_ids[src] - step_ids[dst]).to(dtype=pos_s.dtype)
    r = torch.stack(
        [
            torch.norm(rel_pos[:, :2], p=2, dim=-1),
            angle_between_2d_vectors(ctr_vector=head_vector_s[dst], nbr_vector=rel_pos[:, :2]),
            rel_head,
            rel_step,
        ],
        dim=-1,
    )
    return edge_index, r


def build_temporal_edges(
    pos_a: torch.Tensor,
    head_a: torch.Tensor,
    head_vector_a: torch.Tensor,
    mask: torch.Tensor,
    max_step_delta: Optional[int],
    causal: bool = True,
    target_mask: Optional[torch.Tensor] = None,
):
    num_agent, num_step, _ = pos_a.shape
    device = pos_a.device
    pos_s = pos_a.reshape(-1, pos_a.shape[-1])
    head_s = head_a.reshape(-1)
    head_vector_s = head_vector_a.reshape(-1, 2)
    agent_ids = torch.arange(num_agent, device=device).unsqueeze(1).expand(num_agent, num_step).reshape(-1)
    step_ids = torch.arange(num_step, device=device).unsqueeze(0).expand(num_agent, num_step).reshape(-1)
    return build_temporal_edges_from_flat(
        pos_s=pos_s,
        head_s=head_s,
        head_vector_s=head_vector_s,
        agent_ids=agent_ids,
        step_ids=step_ids,
        valid_mask=mask.reshape(-1),
        max_step_delta=max_step_delta,
        causal=causal,
        target_mask=None if target_mask is None else target_mask.reshape(-1),
    )


def build_radius_interaction_edges(
    pos_s: torch.Tensor,
    head_s: torch.Tensor,
    head_vector_s: torch.Tensor,
    batch_s: torch.Tensor,
    mask_s: torch.Tensor,
    radius_m: float,
    max_num_neighbors: int = 300,
):
    device = pos_s.device
    edge_index = radius_graph(
        x=pos_s[:, :2],
        r=radius_m,
        batch=batch_s,
        loop=False,
        max_num_neighbors=max_num_neighbors,
    )
    if mask_s is not None:
        edge_index = subgraph(subset=mask_s, edge_index=edge_index)[0]
    if edge_index.numel() == 0:
        return empty_edge_index(device), pos_s.new_zeros((0, 3))

    src, dst = edge_index
    rel_pos = pos_s[src] - pos_s[dst]
    rel_head = wrap_angle(head_s[src] - head_s[dst])
    r = torch.stack(
        [
            torch.norm(rel_pos[:, :2], p=2, dim=-1),
            angle_between_2d_vectors(ctr_vector=head_vector_s[dst], nbr_vector=rel_pos[:, :2]),
            rel_head,
        ],
        dim=-1,
    )
    return edge_index, r


def build_interaction_edges(
    pos_a: torch.Tensor,
    head_a: torch.Tensor,
    head_vector_a: torch.Tensor,
    batch_s: torch.Tensor,
    mask_s: torch.Tensor,
    radius_m: float,
    max_num_neighbors: int = 300,
):
    pos_s = pos_a.transpose(0, 1).reshape(-1, pos_a.shape[-1])
    head_s = head_a.transpose(0, 1).reshape(-1)
    head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
    return build_radius_interaction_edges(
        pos_s=pos_s,
        head_s=head_s,
        head_vector_s=head_vector_s,
        batch_s=batch_s,
        mask_s=mask_s,
        radius_m=radius_m,
        max_num_neighbors=max_num_neighbors,
    )


def build_radius_map2agent_edges(
    pos_s: torch.Tensor,
    head_s: torch.Tensor,
    head_vector_s: torch.Tensor,
    pos_pl: torch.Tensor,
    orient_pl: torch.Tensor,
    batch_s: torch.Tensor,
    batch_pl: torch.Tensor,
    mask_s: torch.Tensor,
    radius_m: float,
    map_token_visible_mask: Optional[torch.Tensor] = None,
    max_num_neighbors: int = 300,
):
    device = pos_s.device
    edge_index = radius(
        x=pos_s[:, :2],
        y=pos_pl[:, :2],
        r=radius_m,
        batch_x=batch_s,
        batch_y=batch_pl,
        max_num_neighbors=max_num_neighbors,
    )
    if edge_index.numel() == 0:
        return empty_edge_index(device), pos_s.new_zeros((0, 3))

    edge_keep_mask = mask_s[edge_index[1]]
    if map_token_visible_mask is not None:
        edge_keep_mask = edge_keep_mask & map_token_visible_mask.bool()[edge_index[0]]
    edge_index = edge_index[:, edge_keep_mask]
    if edge_index.numel() == 0:
        return empty_edge_index(device), pos_s.new_zeros((0, 3))

    src, dst = edge_index
    rel_pos = pos_pl[src] - pos_s[dst]
    rel_orient = wrap_angle(orient_pl[src] - head_s[dst])
    r = torch.stack(
        [
            torch.norm(rel_pos[:, :2], p=2, dim=-1),
            angle_between_2d_vectors(ctr_vector=head_vector_s[dst], nbr_vector=rel_pos[:, :2]),
            rel_orient,
        ],
        dim=-1,
    )
    return edge_index, r


def build_map2agent_edges(
    pos_a: torch.Tensor,
    head_a: torch.Tensor,
    head_vector_a: torch.Tensor,
    pos_pl: torch.Tensor,
    orient_pl: torch.Tensor,
    batch_s: torch.Tensor,
    batch_pl: torch.Tensor,
    mask: torch.Tensor,
    radius_m: float,
    map_token_visible_mask: Optional[torch.Tensor] = None,
    max_num_neighbors: int = 300,
):
    num_step = pos_a.shape[1]
    mask_s = mask.transpose(0, 1).reshape(-1)
    pos_s = pos_a.transpose(0, 1).reshape(-1, pos_a.shape[-1])
    head_s = head_a.transpose(0, 1).reshape(-1)
    head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
    pos_pl = pos_pl.repeat(num_step, 1)
    orient_pl = orient_pl.repeat(num_step)
    if map_token_visible_mask is not None:
        map_token_visible_mask = map_token_visible_mask.bool().repeat(num_step)
    return build_radius_map2agent_edges(
        pos_s=pos_s,
        head_s=head_s,
        head_vector_s=head_vector_s,
        pos_pl=pos_pl,
        orient_pl=orient_pl,
        batch_s=batch_s,
        batch_pl=batch_pl,
        mask_s=mask_s,
        radius_m=radius_m,
        map_token_visible_mask=map_token_visible_mask,
        max_num_neighbors=max_num_neighbors,
    )
