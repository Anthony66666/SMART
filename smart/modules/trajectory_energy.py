from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from smart.utils import wrap_angle


class TrajectoryEnergy(nn.Module):
    """Tensor-only safety energies for reranking discrete trajectory tokens."""

    def __init__(
        self,
        dt: float = 0.1,
        max_acceleration: float = 6.0,
        max_yaw_rate: float = 1.2,
        collision_distance: float = 2.0,
    ) -> None:
        super().__init__()
        self.dt = max(float(dt), 1e-3)
        self.max_acceleration = max(float(max_acceleration), 0.0)
        self.max_yaw_rate = max(float(max_yaw_rate), 0.0)
        self.collision_distance = max(float(collision_distance), 0.0)

    def lane_energy(
        self,
        candidate_positions: torch.Tensor,
        candidate_headings: torch.Tensor,
        map_positions: Optional[torch.Tensor],
        map_orientations: Optional[torch.Tensor],
        candidate_batch: Optional[torch.Tensor] = None,
        map_batch: Optional[torch.Tensor] = None,
        map_valid_mask: Optional[torch.Tensor] = None,
    ):
        num_candidates, topk, num_steps, _ = candidate_positions.shape
        lane_distance = candidate_positions.new_zeros(num_candidates, topk)
        lane_heading = candidate_positions.new_zeros(num_candidates, topk)
        if (
            map_positions is None
            or map_orientations is None
            or map_positions.numel() == 0
        ):
            return lane_distance, lane_heading

        if candidate_batch is None:
            candidate_batch = torch.zeros(
                num_candidates,
                dtype=torch.long,
                device=candidate_positions.device,
            )
        if map_batch is None:
            map_batch = torch.zeros(
                map_positions.shape[0],
                dtype=torch.long,
                device=map_positions.device,
            )
        if map_valid_mask is None:
            map_valid_mask = torch.ones(
                map_positions.shape[0],
                dtype=torch.bool,
                device=map_positions.device,
            )

        for candidate_idx in range(num_candidates):
            available_map = (
                map_valid_mask.bool()
                & (map_batch == candidate_batch[candidate_idx])
            )
            if not available_map.any():
                continue
            scene_positions = map_positions[available_map, :2]
            scene_orientations = map_orientations[available_map]
            points = candidate_positions[candidate_idx].reshape(-1, 2)
            distances = torch.cdist(points, scene_positions)
            nearest_distance, nearest_index = distances.min(dim=-1)
            nearest_orientation = scene_orientations[nearest_index]
            headings = candidate_headings[candidate_idx].reshape(-1)
            lane_distance[candidate_idx] = nearest_distance.reshape(
                topk,
                num_steps,
            ).mean(dim=-1)
            lane_heading[candidate_idx] = wrap_angle(
                headings - nearest_orientation
            ).abs().reshape(topk, num_steps).mean(dim=-1)
        return lane_distance, lane_heading

    def dynamics_energy(
        self,
        candidate_positions: torch.Tensor,
        candidate_headings: torch.Tensor,
        current_positions: Optional[torch.Tensor] = None,
        current_velocities: Optional[torch.Tensor] = None,
        current_headings: Optional[torch.Tensor] = None,
    ):
        num_candidates, topk, num_steps, _ = candidate_positions.shape
        if current_positions is not None:
            current_positions = current_positions.to(
                device=candidate_positions.device,
                dtype=candidate_positions.dtype,
            )
            positions = torch.cat(
                [
                    current_positions[:, None, None, :].expand(-1, topk, 1, -1),
                    candidate_positions,
                ],
                dim=2,
            )
        else:
            positions = candidate_positions

        if positions.shape[2] < 2:
            return candidate_positions.new_zeros(candidate_positions.shape[:2])
        velocity = (positions[:, :, 1:] - positions[:, :, :-1]) / self.dt

        if current_velocities is not None:
            current_velocities = current_velocities.to(
                device=candidate_positions.device,
                dtype=candidate_positions.dtype,
            )
            velocity_for_acceleration = torch.cat(
                [
                    current_velocities[:, None, None, :].expand(-1, topk, 1, -1),
                    velocity,
                ],
                dim=2,
            )
        else:
            velocity_for_acceleration = velocity
        if velocity_for_acceleration.shape[2] >= 2:
            acceleration = (
                velocity_for_acceleration[:, :, 1:]
                - velocity_for_acceleration[:, :, :-1]
            ) / self.dt
            acceleration_excess = F.relu(
                torch.norm(acceleration, dim=-1) - self.max_acceleration
            ).square().mean(dim=-1)
        else:
            acceleration_excess = candidate_positions.new_zeros(
                candidate_positions.shape[:2]
            )

        if current_headings is not None:
            current_headings = current_headings.to(
                device=candidate_headings.device,
                dtype=candidate_headings.dtype,
            )
            headings = torch.cat(
                [
                    current_headings[:, None, None].expand(-1, topk, 1),
                    candidate_headings,
                ],
                dim=2,
            )
        else:
            headings = candidate_headings
        if headings.shape[2] < 2:
            yaw_excess = candidate_positions.new_zeros((num_candidates, topk))
            return acceleration_excess + yaw_excess
        yaw_rate = wrap_angle(
            headings[:, :, 1:] - headings[:, :, :-1]
        ).abs() / self.dt
        yaw_excess = F.relu(
            yaw_rate - self.max_yaw_rate
        ).square().mean(dim=-1)
        return acceleration_excess + yaw_excess

    def collision_energy(
        self,
        candidate_positions: torch.Tensor,
        other_positions: Optional[torch.Tensor],
        candidate_batch: Optional[torch.Tensor] = None,
        other_batch: Optional[torch.Tensor] = None,
        candidate_agent_ids: Optional[torch.Tensor] = None,
        other_agent_ids: Optional[torch.Tensor] = None,
    ):
        num_candidates, topk, _num_steps, _ = candidate_positions.shape
        result = candidate_positions.new_zeros(num_candidates, topk)
        if other_positions is None or other_positions.numel() == 0:
            return result
        if candidate_batch is None:
            candidate_batch = torch.zeros(
                num_candidates,
                dtype=torch.long,
                device=candidate_positions.device,
            )
        if other_batch is None:
            other_batch = torch.zeros(
                other_positions.shape[0],
                dtype=torch.long,
                device=other_positions.device,
            )

        for candidate_idx in range(num_candidates):
            other_mask = other_batch == candidate_batch[candidate_idx]
            if candidate_agent_ids is not None and other_agent_ids is not None:
                other_mask = other_mask & (
                    other_agent_ids != candidate_agent_ids[candidate_idx]
                )
            if not other_mask.any():
                continue
            scene_other = other_positions[other_mask]
            step_count = min(
                candidate_positions.shape[2],
                scene_other.shape[1],
            )
            distances = torch.norm(
                candidate_positions[candidate_idx, :, None, :step_count]
                - scene_other[None, :, :step_count],
                dim=-1,
            )
            overlap = F.relu(self.collision_distance - distances).square()
            result[candidate_idx] = overlap.amax(dim=1).mean(dim=-1)
        return result
