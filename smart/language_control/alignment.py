from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SceneTextAlignmentHead(nn.Module):
    """Aligns JEPA scene latents with text prompt latents."""

    def __init__(self, hidden_dim: int, projection_dim: Optional[int] = None, init_temperature: float = 0.07) -> None:
        super().__init__()
        projection_dim = projection_dim or hidden_dim
        self.scene_projector = self._make_projector(hidden_dim, projection_dim)
        self.text_projector = self._make_projector(hidden_dim, projection_dim)
        self.matching_head = nn.Sequential(
            nn.Linear(projection_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / init_temperature)))

    def forward(
        self,
        scene_latent: torch.Tensor,
        text_latent: torch.Tensor,
        matching_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        scene_embedding = F.normalize(self.scene_projector(scene_latent), dim=-1)
        text_embedding = F.normalize(self.text_projector(text_latent), dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        contrastive_logits = scene_embedding @ text_embedding.transpose(0, 1) * scale

        if scene_embedding.shape[0] != text_embedding.shape[0]:
            raise ValueError("scene_latent and text_latent must have the same batch size for alignment.")
        targets = torch.arange(scene_embedding.shape[0], device=scene_embedding.device)
        contrastive_loss = 0.5 * (
            F.cross_entropy(contrastive_logits, targets)
            + F.cross_entropy(contrastive_logits.transpose(0, 1), targets)
        )

        pair_features = torch.cat(
            [
                scene_embedding,
                text_embedding,
                torch.abs(scene_embedding - text_embedding),
                scene_embedding * text_embedding,
            ],
            dim=-1,
        )
        matching_logits = self.matching_head(pair_features).squeeze(-1)
        if matching_labels is None:
            matching_labels = torch.ones_like(matching_logits)
        matching_loss = F.binary_cross_entropy_with_logits(matching_logits, matching_labels.float())
        total_loss = contrastive_loss + matching_loss
        return {
            "loss": total_loss,
            "contrastive_loss": contrastive_loss,
            "matching_loss": matching_loss,
            "contrastive_logits": contrastive_logits,
            "matching_logits": matching_logits,
            "scene_embedding": scene_embedding,
            "text_embedding": text_embedding,
        }

    def _make_projector(self, input_dim: int, output_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )
