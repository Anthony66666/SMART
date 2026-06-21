import types

import pytorch_lightning as pl
import torch

from smart.model.smart_causal_diffusion import SMARTCausalDiffusion
from smart.utils.torch_compat import torch_load_compat


def _balanced_commit_speed_energy_impl(
    self,
    candidate_positions,
    anchor_positions,
    current_velocities,
    frontier_chunk_ids,
    reference_speeds=None,
):
    energy = candidate_positions.new_zeros(candidate_positions.shape[:2])
    if current_velocities is None or candidate_positions.numel() == 0:
        return energy
    commit_mask = frontier_chunk_ids.to(
        device=candidate_positions.device,
    ) == 0
    if not commit_mask.any():
        return energy

    current_velocities = current_velocities.to(
        device=candidate_positions.device,
        dtype=candidate_positions.dtype,
    )
    anchor_positions = anchor_positions.to(
        device=candidate_positions.device,
        dtype=candidate_positions.dtype,
    )
    current_speed = torch.norm(current_velocities, dim=-1)
    if reference_speeds is not None:
        reference_speeds = reference_speeds.to(
            device=candidate_positions.device,
            dtype=candidate_positions.dtype,
        )
        current_speed = torch.maximum(current_speed, reference_speeds.clamp_min(0.0))

    moving_mask = (
        commit_mask
        & (current_speed >= float(getattr(self, 'commit_speed_threshold', 1.0)))
    )
    if not moving_mask.any():
        return energy

    dt = float(getattr(getattr(self, 'trajectory_energy', None), 'dt', 0.1))
    step_dt = max(dt, 1e-3)
    anchor = anchor_positions[:, None, None, :].expand(
        -1,
        candidate_positions.shape[1],
        1,
        -1,
    )
    position_chain = torch.cat([anchor, candidate_positions], dim=-2)
    step_speed = torch.norm(
        position_chain[:, :, 1:] - position_chain[:, :, :-1],
        dim=-1,
    ) / step_dt
    candidate_speed = step_speed.median(dim=-1).values

    minimum_speed = (
        current_speed * float(getattr(self, 'commit_min_speed_ratio', 0.75))
    ).unsqueeze(-1)
    maximum_speed = (
        current_speed * float(getattr(self, 'commit_max_speed_ratio', 1.25))
    ).unsqueeze(-1)
    speed_deficit = (minimum_speed - candidate_speed).clamp_min(0.0)
    speed_excess = (candidate_speed - maximum_speed).clamp_min(0.0)
    scale = max(float(getattr(self, 'commit_speed_threshold', 1.0)), 1.0)
    energy[moving_mask] = (
        (speed_deficit[moving_mask] / scale) ** 2
        + (speed_excess[moving_mask] / scale) ** 2
    )
    return energy


class SMARTHybridDiffusion(pl.LightningModule):
    """Composition-based closed-loop SMART diffusion predictor.

    The public predictor intentionally does not inherit existing SMART predictor
    classes. It keeps the original SMART batch input and delegates the mature
    SMART-token rollout machinery to an internal core while replacing commit
    speed scoring with a bidirectional speed band for the hybrid objective.
    """

    def __init__(self, model_config) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model_config = model_config
        diffusion_cfg = getattr(model_config, 'diffusion', None)
        if diffusion_cfg is None:
            raise ValueError("SMARTHybridDiffusion requires Model.diffusion config.")
        self.hybrid_objective = str(
            getattr(diffusion_cfg, 'hybrid_objective', 'closed_loop_frontier_v1')
        ).lower()
        if self.hybrid_objective != 'closed_loop_frontier_v1':
            raise ValueError(
                "diffusion.hybrid_objective must be closed_loop_frontier_v1."
            )
        self.use_original_smart_input = bool(
            getattr(diffusion_cfg, 'use_original_smart_input', True)
        )
        if not self.use_original_smart_input:
            raise ValueError(
                "SMARTHybridDiffusion expects the original SMART HeteroData/Batch input."
            )
        self.commit_max_speed_ratio = max(
            1.0,
            float(getattr(diffusion_cfg, 'commit_max_speed_ratio', 1.25)),
        )

        self.core = SMARTCausalDiffusion(model_config)
        self.core.hybrid_objective = self.hybrid_objective
        self.core.use_original_smart_input = self.use_original_smart_input
        self.core.commit_max_speed_ratio = self.commit_max_speed_ratio
        self.core._commit_speed_energy = types.MethodType(
            _balanced_commit_speed_energy_impl,
            self.core,
        )
        self.core.log = self.log

    @property
    def encoder(self):
        return self.core.encoder

    @property
    def mask_token_id(self):
        return self.core.mask_token_id

    @property
    def token_size(self):
        return self.core.token_size

    def _balanced_commit_speed_energy(self, *args, **kwargs):
        return _balanced_commit_speed_energy_impl(self, *args, **kwargs)

    def forward(self, data):
        return self.core(data)

    def training_step(self, data, batch_idx):
        self.core.log = self.log
        return self.core.training_step(data, batch_idx)

    def validation_step(self, data, batch_idx):
        self.core.log = self.log
        return self.core.validation_step(data, batch_idx)

    def inference(self, data):
        return self.core.inference(data)

    def configure_optimizers(self):
        return self.core.configure_optimizers()

    def load_params_from_file(self, filename, logger, to_cpu=False):
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch_load_compat(
            filename,
            map_location=loc_type,
            weights_only=False,
        )
        state = checkpoint.get('state_dict', {})
        if any(key.startswith('core.') for key in state):
            missing_keys, unexpected_keys = self.load_state_dict(state, strict=False)
            logger.info(f'Missing keys: {missing_keys}')
            logger.info(f'The number of missing keys: {len(missing_keys)}')
            logger.info(f'The number of unexpected keys: {len(unexpected_keys)}')
            return checkpoint.get('it', 0.0), checkpoint.get('epoch', -1)
        return self.core.load_params_from_file(filename, logger=logger, to_cpu=to_cpu)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError as exc:
            core = self.__dict__.get('_modules', {}).get('core')
            if core is not None and hasattr(core, name):
                return getattr(core, name)
            raise exc
