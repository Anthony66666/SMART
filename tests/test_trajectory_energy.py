import unittest

import torch

from smart.modules.trajectory_energy import TrajectoryEnergy


class TrajectoryEnergyTest(unittest.TestCase):
    def test_lane_energy_prefers_aligned_candidate(self):
        energy = TrajectoryEnergy()
        candidate_positions = torch.tensor([[
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            [[0.0, 2.0], [1.0, 2.0], [2.0, 2.0]],
        ]])
        candidate_headings = torch.zeros(1, 2, 3)
        map_positions = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        map_orientations = torch.zeros(3)

        lane_distance, lane_heading = energy.lane_energy(
            candidate_positions,
            candidate_headings,
            map_positions,
            map_orientations,
        )

        self.assertLess(float(lane_distance[0, 0]), float(lane_distance[0, 1]))
        self.assertAlmostEqual(float(lane_heading[0, 0]), 0.0, places=5)

    def test_dynamics_energy_penalizes_abrupt_acceleration(self):
        energy = TrajectoryEnergy(dt=0.1, max_acceleration=4.0)
        smooth = torch.tensor([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0]])
        abrupt = torch.tensor([[0.0, 0.0], [0.1, 0.0], [1.5, 0.0], [1.6, 0.0]])
        positions = torch.stack([smooth, abrupt]).view(1, 2, 4, 2)
        headings = torch.zeros(1, 2, 4)

        dynamics = energy.dynamics_energy(positions, headings)

        self.assertLess(float(dynamics[0, 0]), float(dynamics[0, 1]))

    def test_collision_energy_penalizes_overlapping_trajectory(self):
        energy = TrajectoryEnergy(collision_distance=1.0)
        candidates = torch.tensor([[
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            [[0.0, 3.0], [1.0, 3.0], [2.0, 3.0]],
        ]])
        other = torch.tensor([[[0.0, 0.2], [1.0, 0.2], [2.0, 0.2]]])

        collision = energy.collision_energy(candidates, other)

        self.assertGreater(float(collision[0, 0]), float(collision[0, 1]))

    def test_criticality_rewards_near_miss_without_counting_collision_success(self):
        energy = TrajectoryEnergy(dt=0.1, collision_distance=1.0)
        near_miss = torch.tensor([
            [0.0, 1.2],
            [0.5, 1.2],
            [1.0, 1.2],
            [1.5, 1.2],
        ])
        hard_collision = torch.tensor([
            [0.0, 0.2],
            [0.5, 0.2],
            [1.0, 0.2],
            [1.5, 0.2],
        ])
        far_safe = torch.tensor([
            [0.0, 5.0],
            [0.5, 5.0],
            [1.0, 5.0],
            [1.5, 5.0],
        ])
        candidates = torch.stack([near_miss, hard_collision, far_safe]).view(1, 3, 4, 2)
        other = torch.stack([
            torch.tensor([0.5, 0.0]),
            torch.tensor([0.8, 0.0]),
            torch.tensor([1.1, 0.0]),
            torch.tensor([1.4, 0.0]),
        ]).view(1, 4, 2)

        metrics = energy.criticality_metrics(
            candidates,
            other,
            near_miss_distance=2.0,
            ttc_threshold=3.0,
        )

        self.assertGreater(
            float(metrics['critical_reward'][0, 0]),
            float(metrics['critical_reward'][0, 2]),
        )
        self.assertEqual(float(metrics['critical_reward'][0, 1]), 0.0)
        self.assertFalse(bool(metrics['hard_collision'][0, 0]))
        self.assertTrue(bool(metrics['hard_collision'][0, 1]))

    def test_criticality_supports_multiple_reference_agents(self):
        energy = TrajectoryEnergy(dt=0.1, collision_distance=1.0)
        candidate = torch.tensor([[
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        ]])
        references = torch.tensor([
            [[10.0, 0.0], [10.0, 0.0], [10.0, 0.0]],
            [[4.0, 1.2], [3.0, 1.2], [2.0, 1.2]],
        ])

        metrics = energy.criticality_metrics(
            candidate,
            references,
            near_miss_distance=1.5,
            ttc_threshold=3.0,
        )

        self.assertTrue(bool(metrics['near_miss'][0, 0]))
        self.assertFalse(bool(metrics['hard_collision'][0, 0]))

    def test_ego_interaction_rewards_target_intruding_ego_path_corridor(self):
        energy = TrajectoryEnergy(dt=1.0, collision_distance=0.5)
        ego = torch.tensor([
            [0.0, 0.0],
            [2.0, 0.0],
            [4.0, 0.0],
            [6.0, 0.0],
        ])
        intrusive = torch.tensor([
            [1.0, 1.8],
            [2.0, 0.8],
            [3.0, 0.2],
            [4.0, 0.2],
        ])
        far = torch.tensor([
            [1.0, 5.0],
            [2.0, 5.0],
            [3.0, 5.0],
            [4.0, 5.0],
        ])
        candidates = torch.stack([intrusive, far]).view(1, 2, 4, 2)

        metrics = energy.ego_interaction_metrics(
            candidates,
            ego,
            path_corridor_width=1.0,
            near_miss_distance=2.0,
            ttc_threshold=5.0,
            target_spec='cut_in',
        )

        self.assertGreater(
            float(metrics['ego_path_intrusion_rate'][0, 0]),
            float(metrics['ego_path_intrusion_rate'][0, 1]),
        )
        self.assertLess(
            float(metrics['ego_route_lateral_distance'][0, 0]),
            float(metrics['ego_route_lateral_distance'][0, 1]),
        )
        self.assertGreater(
            float(metrics['ego_interaction_reward'][0, 0]),
            float(metrics['ego_interaction_reward'][0, 1]),
        )
        self.assertTrue(bool(metrics['target_event_success'][0, 0]))

    def test_ego_risk_rewards_low_ttc_without_predefined_event(self):
        energy = TrajectoryEnergy(dt=1.0, collision_distance=0.5)
        ego = torch.tensor([
            [0.0, 0.0],
            [2.0, 0.0],
            [4.0, 0.0],
            [6.0, 0.0],
        ])
        closing = torch.tensor([
            [10.0, 0.0],
            [8.0, 0.0],
            [6.0, 0.0],
            [4.0, 0.0],
        ])
        distant = torch.tensor([
            [20.0, 8.0],
            [21.0, 8.0],
            [22.0, 8.0],
            [23.0, 8.0],
        ])
        candidates = torch.stack([closing, distant]).view(1, 2, 4, 2)

        metrics = energy.ego_interaction_metrics(
            candidates,
            ego,
            path_corridor_width=2.0,
            near_miss_distance=8.0,
            ttc_threshold=3.0,
            conflict_tta_threshold=2.0,
            target_spec='ego_risk',
        )

        self.assertIn('ego_risk_reward', metrics)
        self.assertIn('ego_risk_success', metrics)
        self.assertGreater(
            float(metrics['ego_risk_reward'][0, 0]),
            float(metrics['ego_risk_reward'][0, 1]),
        )
        self.assertTrue(bool(metrics['ego_risk_success'][0, 0]))
        self.assertFalse(bool(metrics['ego_risk_success'][0, 1]))
        self.assertTrue(bool(metrics['target_event_success'][0, 0]))
        self.assertGreater(float(metrics['target_event_reward'][0, 0]), 0.0)

    def test_ego_interaction_rewards_lead_hard_brake_event(self):
        energy = TrajectoryEnergy(dt=1.0, collision_distance=0.5)
        ego = torch.tensor([
            [0.0, 0.0],
            [2.0, 0.0],
            [4.0, 0.0],
            [6.0, 0.0],
        ])
        hard_brake = torch.tensor([
            [5.0, 0.0],
            [6.5, 0.0],
            [7.0, 0.0],
            [7.1, 0.0],
        ])
        cruising_lead = torch.tensor([
            [5.0, 2.5],
            [7.0, 2.5],
            [9.0, 2.5],
            [11.0, 2.5],
        ])
        candidates = torch.stack([hard_brake, cruising_lead]).view(1, 2, 4, 2)

        metrics = energy.ego_interaction_metrics(
            candidates,
            ego,
            path_corridor_width=1.0,
            near_miss_distance=3.0,
            ttc_threshold=5.0,
            target_spec='lead_hard_brake',
        )

        self.assertGreater(
            float(metrics['ego_required_decel'][0, 0]),
            float(metrics['ego_required_decel'][0, 1]),
        )
        self.assertTrue(bool(metrics['target_event_success'][0, 0]))
        self.assertFalse(bool(metrics['target_event_success'][0, 1]))


if __name__ == '__main__':
    unittest.main()
