import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _xavier(weight: torch.Tensor) -> None:
    nn.init.xavier_uniform_(weight)


def _zeros(tensor: torch.Tensor) -> None:
    nn.init.zeros_(tensor)


def _normal_002(tensor: torch.Tensor) -> None:
    nn.init.normal_(tensor, mean=0.0, std=0.02)


def _linear(in_features: int, out_features: int, bias: bool = True) -> nn.Linear:
    layer = nn.Linear(in_features, out_features, bias=bias)
    _xavier(layer.weight)
    if bias:
        _zeros(layer.bias)
    return layer


class RMSNorm(nn.Module):
    """RMSNorm used by the official ELF transformer blocks."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps).to(dtype)
        return self.weight.to(dtype) * hidden_states


class TimestepEmbedder(nn.Module):
    """Official ELF-style scalar timestep MLP."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp_0 = _linear(frequency_embedding_size, hidden_size)
        self.mlp_2 = _linear(hidden_size, hidden_size)
        _normal_002(self.mlp_0.weight)
        _normal_002(self.mlp_2.weight)

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor,
        dim: int,
        max_period: int = 10000,
    ) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(0, half, dtype=torch.float32, device=t.device)
            / max(half, 1)
        )
        args = t[:, None].to(torch.float32) * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])],
                dim=-1,
            )
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp_2(F.silu(self.mlp_0(t_emb)))


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward layer from official ELF."""

    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0) -> None:
        super().__init__()
        hidden_dim_eff = int(hidden_dim * 2 / 3)
        self.drop = drop
        self.w12 = _linear(dim, 2 * hidden_dim_eff)
        self.w3 = _linear(hidden_dim_eff, dim)

    def forward(self, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        if self.drop > 0.0:
            hidden = F.dropout(hidden, p=self.drop, training=not deterministic)
        return self.w3(hidden)


class ELFBlock(nn.Module):
    """Transformer block with optional pairwise attention masking."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.norm1 = RMSNorm(hidden_size)
        self.qkv = _linear(hidden_size, hidden_size * 3)
        self.q_norm = RMSNorm(hidden_size // num_heads)
        self.k_norm = RMSNorm(hidden_size // num_heads)
        self.proj = _linear(hidden_size, hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLUFFN(hidden_size, int(hidden_size * mlp_ratio), drop=proj_drop)

    def _attention(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        deterministic: bool,
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        head_dim = hidden_size // self.num_heads
        qkv = self.qkv(x).reshape(
            batch_size,
            seq_len,
            3,
            self.num_heads,
            head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        query = self.q_norm(query)
        key = self.k_norm(key)

        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(float(head_dim))
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            if attention_mask.dim() == 2:
                key_mask = attention_mask[:, None, None, :]
            elif attention_mask.dim() == 3:
                key_mask = attention_mask[:, None, :, :]
            else:
                raise ValueError("attention_mask must have shape [B, S] or [B, Q, K].")
            scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        if self.attn_drop > 0.0:
            attn = F.dropout(attn, p=self.attn_drop, training=not deterministic)
        out = attn.matmul(value).transpose(1, 2).reshape(
            batch_size,
            seq_len,
            hidden_size,
        )
        out = self.proj(out)
        if self.proj_drop > 0.0:
            out = F.dropout(out, p=self.proj_drop, training=not deterministic)
        return out

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        deterministic: bool = True,
        query_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self._attention(self.norm1(x), attention_mask, deterministic)
        x = x + self.mlp(self.norm2(x), deterministic=deterministic)
        if query_mask is None and attention_mask is not None and attention_mask.dim() == 2:
            query_mask = attention_mask
        if query_mask is not None:
            x = x * query_mask.bool().unsqueeze(-1).to(dtype=x.dtype)
        return x


class FinalLayer(nn.Module):
    """Zero-initialized ELF flow output head."""

    def __init__(self, hidden_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = _linear(hidden_size, out_channels)
        _zeros(self.linear.weight)
        _zeros(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm_final(x))


class EmbeddedLanguageFlowDecoder(nn.Module):
    """Official ELF-style decoder adapted to SMART motion token embeddings.

    Inputs and outputs live in the SMART physical token embedding space. The
    module is deliberately independent of SMART diffusion decoders: it performs
    chunk-causal receding-window sequence modeling with an embedding-space flow
    head and a factored token decoder head.
    """

    def __init__(
        self,
        text_encoder_dim: int,
        max_length: int,
        hidden_size: int,
        depth: int,
        num_heads: int,
        vocab_size: int,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        bottleneck_dim: Optional[int] = None,
        num_time_tokens: int = 4,
        num_model_mode_tokens: int = 1,
        num_token_types: int = 4,
    ) -> None:
        super().__init__()
        if num_time_tokens <= 0:
            raise ValueError("num_time_tokens must be positive.")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")
        self.text_encoder_dim = text_encoder_dim
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.mask_token_id = vocab_size
        bottleneck_dim = bottleneck_dim or min(text_encoder_dim, hidden_size)

        self.self_cond_proj = _linear(2 * text_encoder_dim, text_encoder_dim)
        self.text_proj1 = _linear(text_encoder_dim, bottleneck_dim, bias=False)
        self.text_proj2 = _linear(bottleneck_dim, hidden_size)
        self.context_projection = _linear(text_encoder_dim, hidden_size)
        self.chunk_embedding = nn.Embedding(max_length, hidden_size)
        self.type_embedding = nn.Embedding(num_token_types, hidden_size)

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.t_emb_tokens = nn.Parameter(torch.empty(1, num_time_tokens, hidden_size))
        _normal_002(self.t_emb_tokens)
        self.mode_tokens = nn.Parameter(
            torch.empty(1, num_model_mode_tokens, hidden_size)
        )
        _normal_002(self.mode_tokens)

        self.blocks = nn.ModuleList()
        q1, q3 = depth // 4, depth // 4 * 3
        for layer_idx in range(depth):
            in_drop_range = q3 > layer_idx >= q1
            self.blocks.append(
                ELFBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop if in_drop_range else 0.0,
                    proj_drop=proj_drop if in_drop_range else 0.0,
                )
            )

        self.final_layer = FinalLayer(hidden_size, text_encoder_dim)
        self.proj_kernel = nn.Parameter(torch.empty(hidden_size, text_encoder_dim))
        self.proj_bias = nn.Parameter(torch.empty(text_encoder_dim))
        self.unembed_kernel = nn.Parameter(torch.empty(text_encoder_dim, vocab_size))
        self.unembed_bias = nn.Parameter(torch.empty(vocab_size))
        _xavier(self.proj_kernel)
        _zeros(self.proj_bias)
        _xavier(self.unembed_kernel)
        _zeros(self.unembed_bias)

    def _project_text(self, x: torch.Tensor) -> torch.Tensor:
        return self.text_proj2(self.text_proj1(x.float()))

    def _mode_prefix(
        self,
        batch_size: int,
        decoder_step_active: Optional[object],
    ) -> torch.Tensor:
        mode_tokens = self.mode_tokens.expand(batch_size, -1, -1)
        if decoder_step_active is None:
            return mode_tokens * 0.0
        if isinstance(decoder_step_active, torch.Tensor):
            gate = decoder_step_active.to(mode_tokens.dtype).view(-1, 1, 1)
        else:
            gate = float(decoder_step_active)
        return mode_tokens * gate

    def _build_receding_attention_mask(
        self,
        attention_mask: torch.Tensor,
        chunk_ids: Optional[torch.Tensor],
        prefix_len: int,
    ) -> torch.Tensor:
        batch_size, seq_len = attention_mask.shape
        attention_mask = attention_mask.bool()
        total_len = prefix_len + seq_len
        full_mask = torch.zeros(
            batch_size,
            total_len,
            total_len,
            dtype=torch.bool,
            device=attention_mask.device,
        )
        if prefix_len > 0:
            full_mask[:, :prefix_len, :prefix_len] = True
            full_mask[:, prefix_len:, :prefix_len] = True
        if chunk_ids is None:
            data_mask = attention_mask[:, None, :].expand(batch_size, seq_len, seq_len)
        else:
            chunk_ids = chunk_ids.to(device=attention_mask.device)
            query_chunk = chunk_ids[:, :, None]
            key_chunk = chunk_ids[:, None, :]
            data_mask = key_chunk <= query_chunk
            data_mask = data_mask & attention_mask[:, None, :]
        full_mask[:, prefix_len:, prefix_len:] = data_mask
        return full_mask

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        deterministic: bool = True,
        decoder_step_active: Optional[object] = None,
        context: Optional[torch.Tensor] = None,
        chunk_ids: Optional[torch.Tensor] = None,
        agent_type_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = x.shape
        if x.shape[-1] == 2 * self.text_encoder_dim:
            x = self.self_cond_proj(x.float())

        hidden = self._project_text(x)
        if context is not None:
            hidden = hidden + self.context_projection(context.to(dtype=hidden.dtype))
        if chunk_ids is not None:
            hidden = hidden + self.chunk_embedding(
                chunk_ids.clamp(min=0, max=self.max_length - 1)
            )
        if agent_type_ids is not None:
            hidden = hidden + self.type_embedding(
                agent_type_ids.clamp(min=0, max=self.type_embedding.num_embeddings - 1)
            )

        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size,
                seq_len,
                dtype=torch.bool,
                device=x.device,
            )
        else:
            attention_mask = attention_mask.bool()
        hidden = hidden * attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)

        time_tokens = self.t_emb_tokens.expand(batch_size, -1, -1)
        time_tokens = time_tokens + self.t_embedder(t).unsqueeze(1)
        mode_tokens = self._mode_prefix(batch_size, decoder_step_active)
        prefix = torch.cat([mode_tokens, time_tokens], dim=1)
        hidden = torch.cat([prefix, hidden], dim=1)
        prefix_mask = torch.ones(
            batch_size,
            prefix.shape[1],
            dtype=torch.bool,
            device=attention_mask.device,
        )
        full_valid_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        full_attention_mask = self._build_receding_attention_mask(
            attention_mask,
            chunk_ids,
            prefix.shape[1],
        )

        for block in self.blocks:
            hidden = block(
                hidden,
                attention_mask=full_attention_mask,
                deterministic=deterministic,
                query_mask=full_valid_mask,
            )

        hidden = hidden[:, prefix.shape[1]:]
        hidden = hidden * attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        output = self.final_layer(hidden.float())

        decoder_logits = None
        if decoder_step_active is not None:
            hidden_f32 = hidden.float()
            decoder_hidden = F.gelu(
                hidden_f32.matmul(self.proj_kernel) + self.proj_bias,
                approximate="tanh",
            )
            decoder_logits = decoder_hidden.matmul(self.unembed_kernel) + self.unembed_bias
        return output, decoder_logits
