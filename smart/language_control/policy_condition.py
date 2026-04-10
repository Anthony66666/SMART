from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class LanguagePolicyCondition:
    conditioned_tokens: torch.Tensor
    policy_query: torch.Tensor
    text_per_agent: torch.Tensor
    gate: torch.Tensor


class PolicyConditioner(nn.Module):
    """Injects language prompt latents into SMART agent tokens.

    The module produces a ProSim-style policy query per agent and uses it as a
    residual condition on the per-agent token sequence. It does not mutate the
    SMART decoder, so it can be tested independently before deeper integration.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.policy_query = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        agent_tokens: torch.Tensor,
        text_latent: torch.Tensor,
        agent_batch: Optional[torch.Tensor] = None,
        agent_mask: Optional[torch.Tensor] = None,
    ) -> LanguagePolicyCondition:
        if agent_tokens.dim() != 3:
            raise ValueError("agent_tokens must have shape [num_agents, num_steps, hidden_dim].")
        if agent_tokens.shape[-1] != self.hidden_dim or text_latent.shape[-1] != self.hidden_dim:
            raise ValueError("agent_tokens and text_latent must match PolicyConditioner.hidden_dim.")

        num_agents = agent_tokens.shape[0]
        if agent_batch is None:
            agent_batch = torch.zeros(num_agents, dtype=torch.long, device=agent_tokens.device)
        if agent_batch.shape[0] != num_agents:
            raise ValueError("agent_batch must contain one graph index per agent.")
        if int(agent_batch.max().item()) >= text_latent.shape[0]:
            raise ValueError("agent_batch references a text_latent row that does not exist.")

        if agent_mask is None:
            agent_summary = agent_tokens.mean(dim=1)
        else:
            token_mask = agent_mask.to(dtype=agent_tokens.dtype).unsqueeze(-1)
            agent_summary = (agent_tokens * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp_min(1.0)

        text_per_agent = text_latent[agent_batch]
        fused = torch.cat([agent_summary, text_per_agent], dim=-1)
        policy_query = self.policy_query(fused)
        gate = self.gate(fused)
        conditioned_tokens = self.output_norm(agent_tokens + gate.unsqueeze(1) * policy_query.unsqueeze(1))
        return LanguagePolicyCondition(
            conditioned_tokens=conditioned_tokens,
            policy_query=policy_query,
            text_per_agent=text_per_agent,
            gate=gate,
        )
