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

    def criticality_metrics(
        self,
        candidate_positions: torch.Tensor,
        other_positions: Optional[torch.Tensor],
        near_miss_distance: float = 2.0,
        ttc_threshold: float = 3.0,
    ):
        """Measure near-miss criticality against one reference trajectory per row.

        Returns per-candidate metrics with shape [num_candidates, topk]. Hard
        overlaps receive zero critical reward so collisions are not counted as a
        successful stress/edit event.
        """
        num_candidates, topk, num_steps, _ = candidate_positions.shape
        device = candidate_positions.device
        dtype = candidate_positions.dtype
        inf = torch.tensor(float('inf'), device=device, dtype=dtype)
        zeros = candidate_positions.new_zeros(num_candidates, topk)
        result = {
            'min_distance': zeros.clone(),
            'min_ttc': torch.full((num_candidates, topk), inf, device=device, dtype=dtype),
            'required_deceleration': zeros.clone(),
            'critical_reward': zeros.clone(),
            'near_miss': torch.zeros(num_candidates, topk, dtype=torch.bool, device=device),
            'hard_collision': torch.zeros(num_candidates, topk, dtype=torch.bool, device=device),
        }
        if other_positions is None or other_positions.numel() == 0 or num_steps == 0:
            return result
        if other_positions.dim() == 2:
            other_positions = other_positions.unsqueeze(0).unsqueeze(0)
        elif other_positions.dim() == 3:
            if other_positions.shape[0] == num_candidates:
                other_positions = other_positions.unsqueeze(1)
            elif num_candidates == 1:
                other_positions = other_positions.unsqueeze(0)
            elif other_positions.shape[0] == 1:
                other_positions = other_positions.expand(
                    num_candidates,
                    -1,
                    -1,
                ).unsqueeze(1)
            else:
                raise ValueError("other_positions batch must match candidate rows.")
        elif other_positions.dim() == 4:
            if other_positions.shape[0] == 1 and num_candidates > 1:
                other_positions = other_positions.expand(
                    num_candidates,
                    -1,
                    -1,
                    -1,
                )
            elif other_positions.shape[0] != num_candidates:
                raise ValueError("other_positions batch must match candidate rows.")
        else:
            raise ValueError(
                "other_positions must have shape [T,2], [N,T,2], "
                "or [N,O,T,2]."
            )

        step_count = min(num_steps, int(other_positions.shape[-2]))
        candidates = candidate_positions[:, :, :step_count]
        others = other_positions[:, :, :step_count].to(device=device, dtype=dtype)
        distances = torch.norm(
            candidates[:, :, None, :step_count] - others[:, None],
            dim=-1,
        ).clamp_min(1e-6)
        min_distance = distances.amin(dim=(-1, -2))

        min_ttc = torch.full_like(min_distance, float('inf'))
        required_deceleration = torch.zeros_like(min_distance)
        if step_count >= 2:
            candidate_velocity = (candidates[:, :, 1:] - candidates[:, :, :-1]) / self.dt
            other_velocity = (others[:, :, 1:] - others[:, :, :-1]) / self.dt
            rel_pos = others[:, None, :, 1:] - candidates[:, :, None, 1:]
            rel_vel = other_velocity[:, None] - candidate_velocity[:, :, None]
            rel_dist = torch.norm(rel_pos, dim=-1).clamp_min(1e-6)
            closing_speed = -(
                rel_pos * rel_vel
            ).sum(dim=-1) / rel_dist
            closing = closing_speed > 1e-3
            ttc = torch.full_like(closing_speed, float('inf'))
            ttc[closing] = rel_dist[closing] / closing_speed[closing].clamp_min(1e-3)
            min_ttc = ttc.amin(dim=(-1, -2))
            max_closing_speed = closing_speed.clamp_min(0.0).amax(dim=(-1, -2))
            required_deceleration = (
                max_closing_speed.square() / (2.0 * min_distance.clamp_min(1e-3))
            )

        hard_collision = (
            distances < self.collision_distance
        ).any(dim=-1).any(dim=-1)
        near_miss = (
            (min_distance >= self.collision_distance)
            & (min_distance <= float(near_miss_distance))
            & (min_ttc <= float(ttc_threshold))
        )
        reward = (
            F.relu(float(near_miss_distance) - min_distance)
            + F.relu(float(ttc_threshold) - min_ttc.clamp_max(float(ttc_threshold)))
            / max(float(ttc_threshold), 1e-3)
            + 0.1 * required_deceleration
        )
        reward = reward.masked_fill(~near_miss | hard_collision, 0.0)

        result['min_distance'] = min_distance
        result['min_ttc'] = min_ttc
        result['required_deceleration'] = required_deceleration
        result['critical_reward'] = reward
        result['near_miss'] = near_miss
        result['hard_collision'] = hard_collision
        return result

    @staticmethod
    def _expand_reference_positions(
        reference: Optional[torch.Tensor],
        num_rows: int,
        device: torch.device,
        dtype: torch.dtype,
        name: str,
    ) -> Optional[torch.Tensor]:
        if reference is None:
            return None
        reference = reference.to(device=device, dtype=dtype)
        if reference.dim() == 2:
            return reference.unsqueeze(0).expand(num_rows, -1, -1)
        if reference.dim() == 3:
            if reference.shape[0] == num_rows:
                return reference
            if reference.shape[0] == 1:
                return reference.expand(num_rows, -1, -1)
        if reference.dim() == 4 and reference.shape[1] == 1:
            reference = reference[:, 0]
            if reference.shape[0] == num_rows:
                return reference
            if reference.shape[0] == 1:
                return reference.expand(num_rows, -1, -1)
        raise ValueError(f"{name} batch must match candidate rows.")

    @staticmethod
    def _expand_reference_headings(
        reference: Optional[torch.Tensor],
        num_rows: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if reference is None:
            return None
        reference = reference.to(device=device, dtype=dtype)
        if reference.dim() == 1:
            return reference.unsqueeze(0).expand(num_rows, -1)
        if reference.dim() == 2:
            if reference.shape[0] == num_rows:
                return reference
            if reference.shape[0] == 1:
                return reference.expand(num_rows, -1)
        if reference.dim() == 3 and reference.shape[1] == 1:
            reference = reference[:, 0]
            if reference.shape[0] == num_rows:
                return reference
            if reference.shape[0] == 1:
                return reference.expand(num_rows, -1)
        raise ValueError("ego_ref_heading batch must match candidate rows.")

    def _route_distances(self, candidates, route):
        num_rows, topk, num_steps, _ = candidates.shape
        result = candidates.new_full((num_rows, topk, num_steps), float('inf'))
        for row_idx in range(num_rows):
            route_points = route[row_idx]
            if route_points.numel() == 0:
                continue
            finite = torch.isfinite(route_points).all(dim=-1)
            if not finite.any():
                continue
            distances = torch.cdist(
                candidates[row_idx].reshape(-1, 2),
                route_points[finite, :2],
            )
            result[row_idx] = distances.min(dim=-1).values.reshape(
                topk,
                num_steps,
            )
        return result

    def _ego_forward_vectors(self, ego_positions, ego_headings):
        num_rows, num_steps, _ = ego_positions.shape
        if ego_headings is None:
            if num_steps <= 1:
                return ego_positions.new_zeros(num_rows, num_steps, 2)
            delta = ego_positions[:, 1:] - ego_positions[:, :-1]
            last = delta[:, -1:]
            delta = torch.cat([delta, last], dim=1)
            norm = torch.norm(delta, dim=-1, keepdim=True).clamp_min(1e-6)
            return delta / norm
        step_count = min(num_steps, int(ego_headings.shape[-1]))
        headings = ego_headings[:, :step_count]
        forward = torch.stack([headings.cos(), headings.sin()], dim=-1)
        if step_count < num_steps:
            pad = forward[:, -1:].expand(-1, num_steps - step_count, -1)
            forward = torch.cat([forward, pad], dim=1)
        return forward

    def ego_interaction_metrics(
        self,
        candidate_positions: torch.Tensor,
        ego_positions: Optional[torch.Tensor],
        ego_headings: Optional[torch.Tensor] = None,
        ego_route_corridor: Optional[torch.Tensor] = None,
        path_corridor_width: float = 2.0,
        near_miss_distance: float = 2.0,
        ttc_threshold: float = 3.0,
        conflict_tta_threshold: float = 1.0,
        target_spec: str = 'ego_risk',
    ):
        """Measure target-agent risk only against the ego/SDC reference path."""
        num_rows, topk, num_steps, _ = candidate_positions.shape
        device = candidate_positions.device
        dtype = candidate_positions.dtype
        inf = torch.tensor(float('inf'), device=device, dtype=dtype)
        zeros = candidate_positions.new_zeros(num_rows, topk)
        result = {
            'ego_min_distance': zeros.clone(),
            'ego_min_ttc': torch.full((num_rows, topk), inf, device=device, dtype=dtype),
            'ego_required_decel': zeros.clone(),
            'ego_path_intrusion_rate': zeros.clone(),
            'ego_route_lateral_distance': zeros.clone(),
            'ego_conflict_tta_error': zeros.clone(),
            'ego_interaction_reward': zeros.clone(),
            'ego_risk_reward': zeros.clone(),
            'target_event_reward': zeros.clone(),
            'ego_near_miss_success': torch.zeros(num_rows, topk, dtype=torch.bool, device=device),
            'ego_risk_success': torch.zeros(num_rows, topk, dtype=torch.bool, device=device),
            'target_event_success': torch.zeros(num_rows, topk, dtype=torch.bool, device=device),
            'hard_collision': torch.zeros(num_rows, topk, dtype=torch.bool, device=device),
        }
        if ego_positions is None or ego_positions.numel() == 0 or num_steps == 0:
            return result

        ego_positions = self._expand_reference_positions(
            ego_positions,
            num_rows,
            device,
            dtype,
            'ego_ref_traj',
        )
        ego_headings = self._expand_reference_headings(
            ego_headings,
            num_rows,
            device,
            dtype,
        )
        step_count = min(num_steps, int(ego_positions.shape[1]))
        if step_count <= 0:
            return result
        candidates = candidate_positions[:, :, :step_count]
        ego = ego_positions[:, :step_count]
        corridor_width = max(float(path_corridor_width), 1e-3)

        time_aligned_distance = torch.norm(
            candidates - ego[:, None],
            dim=-1,
        ).clamp_min(1e-6)
        min_distance = time_aligned_distance.amin(dim=-1)
        hard_collision = (time_aligned_distance < self.collision_distance).any(dim=-1)

        route = self._expand_reference_positions(
            ego_route_corridor,
            num_rows,
            device,
            dtype,
            'ego_route_corridor',
        )
        if route is None:
            route = ego
        route_distance = self._route_distances(candidates, route)
        finite_route = torch.isfinite(route_distance)
        route_lateral_distance = torch.where(
            finite_route,
            route_distance,
            torch.full_like(route_distance, corridor_width * 10.0),
        ).mean(dim=-1)
        path_intrusion_rate = (
            route_distance <= corridor_width
        ).to(dtype=dtype).mean(dim=-1)

        conflict_tta_error = candidates.new_zeros(num_rows, topk)
        for row_idx in range(num_rows):
            pair_distance = torch.cdist(
                candidates[row_idx].reshape(-1, 2),
                ego[row_idx],
            ).reshape(topk, step_count, step_count)
            flat_index = pair_distance.reshape(topk, -1).argmin(dim=-1)
            target_time = torch.div(flat_index, step_count, rounding_mode='floor')
            ego_time = flat_index.remainder(step_count)
            conflict_tta_error[row_idx] = (
                (target_time - ego_time).abs().to(dtype=dtype) * self.dt
            )

        min_ttc = torch.full_like(min_distance, float('inf'))
        required_decel = torch.zeros_like(min_distance)
        if step_count >= 2:
            candidate_velocity = (
                candidates[:, :, 1:] - candidates[:, :, :-1]
            ) / self.dt
            ego_velocity = (ego[:, 1:] - ego[:, :-1]) / self.dt
            rel_pos = candidates[:, :, 1:] - ego[:, None, 1:]
            rel_vel = candidate_velocity - ego_velocity[:, None]
            rel_dist = torch.norm(rel_pos, dim=-1).clamp_min(1e-6)
            closing_speed = -(
                rel_pos * rel_vel
            ).sum(dim=-1) / rel_dist
            closing = closing_speed > 1e-3
            ttc = torch.full_like(closing_speed, float('inf'))
            ttc[closing] = rel_dist[closing] / closing_speed[closing].clamp_min(1e-3)
            min_ttc = ttc.amin(dim=-1)
            max_closing_speed = closing_speed.clamp_min(0.0).amax(dim=-1)
            required_decel = (
                max_closing_speed.square()
                / (2.0 * min_distance.clamp_min(1e-3))
            )

        ttc_limit = max(float(ttc_threshold), 1e-3)
        tta_limit = max(float(conflict_tta_threshold), 1e-3)
        near_miss = (
            (min_distance >= self.collision_distance)
            & (min_distance <= float(near_miss_distance))
            & (
                (min_ttc <= float(ttc_threshold))
                | (path_intrusion_rate > 0.0)
                | (conflict_tta_error <= tta_limit)
            )
            & ~hard_collision
        )
        interaction_reward = (
            F.relu(float(near_miss_distance) - min_distance)
            + F.relu(ttc_limit - min_ttc.clamp_max(ttc_limit)) / ttc_limit
            + path_intrusion_rate
            + F.relu(corridor_width - route_lateral_distance.clamp_max(corridor_width))
            / corridor_width
            + F.relu(tta_limit - conflict_tta_error.clamp_max(tta_limit))
            / tta_limit
            + 0.1 * required_decel
        )
        interaction_reward = interaction_reward.masked_fill(hard_collision, 0.0)
        finite_ttc = torch.isfinite(min_ttc)
        risk_distance = max(float(near_miss_distance), self.collision_distance + 1e-3)
        low_ttc_reward = F.relu(
            ttc_limit - min_ttc.clamp_max(ttc_limit),
        ) / ttc_limit
        close_distance_reward = F.relu(
            risk_distance - min_distance.clamp_max(risk_distance),
        ) / risk_distance
        conflict_timing_reward = F.relu(
            tta_limit - conflict_tta_error.clamp_max(tta_limit),
        ) / tta_limit
        route_reward = F.relu(
            corridor_width - route_lateral_distance.clamp_max(corridor_width),
        ) / corridor_width
        ego_risk_reward = (
            2.0 * low_ttc_reward
            + close_distance_reward
            + 0.5 * path_intrusion_rate
            + 0.5 * route_reward
            + 0.5 * conflict_timing_reward
            + 0.1 * required_decel
        ).masked_fill(hard_collision, 0.0)
        ego_risk_success = (
            finite_ttc
            & (min_ttc <= ttc_limit)
            & ~hard_collision
        )

        spec = str(target_spec or '').lower().replace('-', '_')
        event_reward = zeros.clone()
        event_success = torch.zeros(num_rows, topk, dtype=torch.bool, device=device)
        if spec in ('ego_risk', 'risk', 'low_ttc', 'ttc'):
            event_reward = ego_risk_reward
            event_success = ego_risk_success
        elif spec in ('cut_in', 'cutin', 'lane_cut_in'):
            start_lateral = route_distance[:, :, 0]
            min_lateral = route_distance.amin(dim=-1)
            cut_in = (
                (start_lateral > corridor_width)
                & (min_lateral <= corridor_width)
                & (conflict_tta_error <= tta_limit)
                & ~hard_collision
            )
            event_reward = (
                path_intrusion_rate
                + F.relu(start_lateral - corridor_width) / corridor_width
                + F.relu(tta_limit - conflict_tta_error.clamp_max(tta_limit))
                / tta_limit
            ).masked_fill(~cut_in, 0.0)
            event_success = cut_in
        elif spec in ('lead_hard_brake', 'hard_brake', 'lead_brake'):
            ego_forward = self._ego_forward_vectors(ego, ego_headings)
            rel = candidates - ego[:, None]
            longitudinal = (rel * ego_forward[:, None]).sum(dim=-1)
            in_lane = route_distance <= corridor_width
            lead_mask = (longitudinal > 0.0) & in_lane
            brake_amount = zeros.clone()
            if step_count >= 3:
                candidate_velocity = (
                    candidates[:, :, 1:] - candidates[:, :, :-1]
                ) / self.dt
                speed = torch.norm(candidate_velocity, dim=-1)
                acceleration = (speed[:, :, 1:] - speed[:, :, :-1]) / self.dt
                brake_amount = F.relu(-acceleration.amin(dim=-1) - 0.3)
            lead_brake = (
                lead_mask.any(dim=-1)
                & (brake_amount > 0.0)
                & ~hard_collision
            )
            event_reward = (
                brake_amount + 0.2 * required_decel
            ).masked_fill(~lead_brake, 0.0)
            event_success = lead_brake
        elif spec in ('crossing_conflict', 'yield_failure'):
            event_success = near_miss & (conflict_tta_error <= tta_limit)
            event_reward = interaction_reward.masked_fill(~event_success, 0.0)

        result['ego_min_distance'] = min_distance
        result['ego_min_ttc'] = min_ttc
        result['ego_required_decel'] = required_decel
        result['ego_path_intrusion_rate'] = path_intrusion_rate
        result['ego_route_lateral_distance'] = route_lateral_distance
        result['ego_conflict_tta_error'] = conflict_tta_error
        result['ego_interaction_reward'] = interaction_reward
        result['ego_risk_reward'] = ego_risk_reward
        result['target_event_reward'] = event_reward
        result['ego_near_miss_success'] = near_miss
        result['ego_risk_success'] = ego_risk_success
        result['target_event_success'] = event_success
        result['hard_collision'] = hard_collision
        return result
