import torch
from torchmetrics import Metric


def _agent_bbox_corners(positions, headings, shapes):
    """Compute 4-corner bounding boxes for all agents at all timesteps.

    Args:
        positions: [A, T, 2] center positions
        headings:  [A, T] yaw angles
        shapes:    [A, 3] (length, width, height)

    Returns:
        corners: [A, T, 4, 2] corner points
    """
    A, T, _ = positions.shape
    device = positions.device
    length = shapes[:, 0:1].unsqueeze(-1)  # [A, 1, 1]
    width = shapes[:, 1:2].unsqueeze(-1)   # [A, 1, 1]

    cos = headings.cos().unsqueeze(-1)  # [A, T, 1]
    sin = headings.sin().unsqueeze(-1)

    half_l = 0.5 * length
    half_w = 0.5 * width

    local_corners = torch.stack([
        torch.cat([ half_l,  half_w], dim=-1),
        torch.cat([ half_l, -half_w], dim=-1),
        torch.cat([-half_l, -half_w], dim=-1),
        torch.cat([-half_l,  half_w], dim=-1),
    ], dim=2)  # [A, 1, 4, 2]

    rot = torch.zeros(A, T, 2, 2, device=device)
    rot[..., 0, 0] = cos.squeeze(-1)
    rot[..., 0, 1] = sin.squeeze(-1)
    rot[..., 1, 0] = -sin.squeeze(-1)
    rot[..., 1, 1] = cos.squeeze(-1)

    rotated = torch.matmul(local_corners.expand(-1, T, -1, -1), rot)
    return rotated + positions.unsqueeze(2)


def _bbox_iou(corners_a, corners_b):
    """Compute approximate IoU via separating axis theorem on oriented boxes.

    Args:
        corners_a: [4, 2] first box corners
        corners_b: [4, 2] second box corners

    Returns:
        iou: scalar float
    """
    def _project(pts, axis):
        proj = (pts * axis).sum(dim=-1)
        return proj.min(), proj.max()

    for box in [corners_a, corners_b]:
        for i in range(4):
            edge = box[(i + 1) % 4] - box[i]
            axis = torch.tensor([-edge[1], edge[0]], device=box.device)
            axis = axis / (axis.norm() + 1e-10)
            min_a, max_a = _project(corners_a, axis)
            min_b, max_b = _project(corners_b, axis)
            if max_a < min_b or max_b < min_a:
                return torch.tensor(0.0, device=box.device)

    # Approximate area via shoelace formula
    def _area(corners):
        x, y = corners[:, 0], corners[:, 1]
        return 0.5 * (x[0] * y[1] + x[1] * y[2] + x[2] * y[3] + x[3] * y[0]
                      - x[1] * y[0] - x[2] * y[1] - x[3] * y[2] - x[0] * y[3]).abs()

    area_a = _area(corners_a)
    area_b = _area(corners_b)

    # Approximate intersection via center + half-extent overlap
    center_a = corners_a.mean(dim=0)
    center_b = corners_b.mean(dim=0)
    half_a = (corners_a.max(dim=0).values - corners_a.min(dim=0).values) / 2
    half_b = (corners_b.max(dim=0).values - corners_b.min(dim=0).values) / 2

    overlap_x = (half_a[0] + half_b[0]) - (center_a[0] - center_b[0]).abs()
    overlap_y = (half_a[1] + half_b[1]) - (center_a[1] - center_b[1]).abs()
    inter = overlap_x.clamp(min=0) * overlap_y.clamp(min=0)

    union = area_a + area_b - inter
    if union < 1e-10:
        return torch.tensor(0.0, device=corners_a.device)
    return inter / union


class ConflictRate(Metric):
    """Fraction of scene timesteps where any 2+ agent bboxes overlap."""

    full_state_update = False

    def __init__(self, iou_threshold=0.1):
        super().__init__()
        self.iou_threshold = iou_threshold
        self.add_state('conflict_timesteps', default=torch.tensor(0.0),
                       dist_reduce_fx='sum')
        self.add_state('total_timesteps', default=torch.tensor(0),
                       dist_reduce_fx='sum')

    def update(self, pred_traj, pred_head, agent_shapes, valid_mask):
        """Accumulate conflict statistics.

        Args:
            pred_traj:    [A, T, 2] predicted positions
            pred_head:    [A, T] predicted headings
            agent_shapes: [A, 3] (length, width, height)
            valid_mask:   [A, T] bool
        """
        if pred_traj.shape[0] < 2:
            return

        A, T, _ = pred_traj.shape
        corners = _agent_bbox_corners(pred_traj, pred_head, agent_shapes)

        for t in range(T):
            active = valid_mask[:, t].nonzero(as_tuple=False).squeeze(-1)
            if active.numel() < 2:
                continue
            self.total_timesteps += 1
            has_conflict = False
            for i_idx in range(active.numel()):
                for j_idx in range(i_idx + 1, active.numel()):
                    i, j = active[i_idx].item(), active[j_idx].item()
                    iou = _bbox_iou(corners[i, t], corners[j, t])
                    if iou > self.iou_threshold:
                        has_conflict = True
                        break
                if has_conflict:
                    break
            if has_conflict:
                self.conflict_timesteps += 1.0

    def compute(self):
        if self.total_timesteps == 0:
            return self.conflict_timesteps.new_tensor(0.0)
        return self.conflict_timesteps / self.total_timesteps


class InteractionConsistency(Metric):
    """Agreement between predicted and GT pairwise agent ordering.

    For each pair of agents at each timestep, checks whether the
    relative ordering (who is ahead) matches ground truth.
    """

    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state('agreement', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total_pairs', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, pred_traj, gt_traj, valid_mask):
        """Compare predicted vs ground truth relative positions.

        Args:
            pred_traj:  [A, T, 2]
            gt_traj:    [A, T, 2]
            valid_mask: [A, T] bool
        """
        A, T, _ = pred_traj.shape
        if A < 2:
            return

        for t in range(T):
            active = valid_mask[:, t].nonzero(as_tuple=False).squeeze(-1)
            if active.numel() < 2:
                continue
            for i_idx in range(active.numel()):
                for j_idx in range(i_idx + 1, active.numel()):
                    i, j = active[i_idx].item(), active[j_idx].item()
                    p_rel = pred_traj[j, t] - pred_traj[i, t]
                    g_rel = gt_traj[j, t] - gt_traj[i, t]
                    p_dot_g = (p_rel * g_rel).sum()
                    if p_dot_g > 0:
                        self.agreement += 1.0
                    self.total_pairs += 1

    def compute(self):
        if self.total_pairs == 0:
            return self.agreement.new_tensor(0.0)
        return self.agreement / self.total_pairs
