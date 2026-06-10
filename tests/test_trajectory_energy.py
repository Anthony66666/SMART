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


if __name__ == '__main__':
    unittest.main()
