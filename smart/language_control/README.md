# JEPA-SMART Language Control

This package contains the first-stage language-control scaffolding for the
JEPA-SMART branch. It follows the ProSim-style split:

1. encode the existing scene initialization with JEPA-SMART,
2. encode a natural-language prompt,
3. align text and scene latents,
4. turn the prompt latent into per-agent policy queries,
5. predict SMART motion-token logits from conditioned agent tokens.

The v1 scope is prompt-conditioned rollout from an existing scene
initialization. It intentionally does not implement text-to-blueprint or
text-to-initialization generation yet.
