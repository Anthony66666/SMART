import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Union

import torch
import torch.nn as nn


@dataclass
class TextPromptBatch:
    token_ids: torch.Tensor
    token_mask: torch.Tensor
    token_embeddings: torch.Tensor
    embedding: torch.Tensor


class TextPromptEncoder(nn.Module):
    """Small dependency-free prompt encoder used as the language-control adapter.

    This encoder is intentionally lightweight so the language-control path can be
    smoke-tested before wiring in a larger LLM. It hashes prompt tokens into a
    fixed vocabulary and projects the pooled token embedding to the SMART hidden
    dimension.
    """

    _TOKEN_PATTERN = re.compile(r"<[^>]+>|[A-Za-z0-9_./:-]+")

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int = 32768,
        max_tokens: int = 64,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        if vocab_size <= 1:
            raise ValueError("vocab_size must leave room for a non-padding token.")
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_tokens = max_tokens
        self.pad_token_id = pad_token_id
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_token_id)
        self.position_embedding = nn.Embedding(max_tokens, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, prompts: Union[Sequence[str], torch.Tensor]) -> TextPromptBatch:
        if isinstance(prompts, torch.Tensor):
            token_ids = prompts.to(device=self.token_embedding.weight.device, dtype=torch.long)
            if token_ids.dim() == 1:
                token_ids = token_ids.unsqueeze(0)
            token_ids = token_ids[:, : self.max_tokens]
        else:
            token_ids = self.tokenize(prompts, device=self.token_embedding.weight.device)

        token_mask = token_ids.ne(self.pad_token_id)
        positions = torch.arange(token_ids.shape[1], device=token_ids.device).unsqueeze(0)
        token_embeddings = self.token_embedding(token_ids) + self.position_embedding(positions)
        token_embeddings = self.dropout(token_embeddings)
        masked_embeddings = token_embeddings * token_mask.unsqueeze(-1)
        pooled = masked_embeddings.sum(dim=1) / token_mask.sum(dim=1, keepdim=True).clamp_min(1)
        embedding = self.projection(pooled)
        return TextPromptBatch(
            token_ids=token_ids,
            token_mask=token_mask,
            token_embeddings=token_embeddings,
            embedding=embedding,
        )

    def tokenize(self, prompts: Iterable[str], device=None) -> torch.Tensor:
        rows: List[List[int]] = []
        for prompt in prompts:
            tokens = self._TOKEN_PATTERN.findall(str(prompt).lower())[: self.max_tokens]
            ids = [self._stable_hash(token) for token in tokens]
            rows.append(ids)

        if not rows:
            return torch.empty((0, self.max_tokens), dtype=torch.long, device=device)

        max_len = min(self.max_tokens, max(1, max(len(row) for row in rows)))
        token_ids = torch.full((len(rows), max_len), self.pad_token_id, dtype=torch.long, device=device)
        for row_idx, row in enumerate(rows):
            if row:
                token_ids[row_idx, : min(len(row), max_len)] = torch.tensor(row[:max_len], dtype=torch.long, device=device)
        return token_ids

    def _stable_hash(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
        return int.from_bytes(digest, "little") % (self.vocab_size - 1) + 1
