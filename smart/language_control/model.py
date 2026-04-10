from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn as nn

from smart.language_control.alignment import SceneTextAlignmentHead
from smart.language_control.policy_condition import PolicyConditioner
from smart.language_control.text_encoder import TextPromptEncoder


class LanguageControlHead(nn.Module):
    """Minimal text-to-policy-token head for JEPA-SMART language control."""

    def __init__(
        self,
        hidden_dim: int,
        token_size: int,
        vocab_size: int = 32768,
        max_prompt_tokens: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.text_encoder = TextPromptEncoder(
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            max_tokens=max_prompt_tokens,
            dropout=dropout,
        )
        self.alignment = SceneTextAlignmentHead(hidden_dim=hidden_dim)
        self.policy_conditioner = PolicyConditioner(hidden_dim=hidden_dim, dropout=dropout)
        self.token_predictor = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, token_size),
        )

    def forward(
        self,
        prompts: Union[Sequence[str], torch.Tensor],
        scene_latent: torch.Tensor,
        agent_tokens: torch.Tensor,
        agent_batch: Optional[torch.Tensor] = None,
        agent_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        text_batch = self.text_encoder(prompts)
        if scene_latent.shape[0] != text_batch.embedding.shape[0]:
            raise ValueError("prompts and scene_latent must have the same batch size.")
        alignment = self.alignment(scene_latent, text_batch.embedding)
        policy_condition = self.policy_conditioner(
            agent_tokens=agent_tokens,
            text_latent=text_batch.embedding,
            agent_batch=agent_batch,
            agent_mask=agent_mask,
        )
        token_logits = self.token_predictor(policy_condition.conditioned_tokens)
        return {
            "token_logits": token_logits,
            "text_embedding": text_batch.embedding,
            "text_token_ids": text_batch.token_ids,
            "scene_text_loss": alignment["loss"],
            "scene_text_contrastive_loss": alignment["contrastive_loss"],
            "scene_text_matching_loss": alignment["matching_loss"],
            "policy_query": policy_condition.policy_query,
            "conditioned_agent_tokens": policy_condition.conditioned_tokens,
            "language_gate": policy_condition.gate,
        }
