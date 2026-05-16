# Decisions

## 2026-05-16
- Use `smart_self_distill` as a separate predictor instead of changing `SMART`, so baseline training and inference stay stable.
- Use an EMA copy of `SMARTDecoder` as the v1 teacher; privileged future teacher is deferred to avoid information-leakage ambiguity.
- Use tensor-only proxy rewards in v1 and keep offroad/map compliance disabled by default until map geometry scoring is added.
- Encourage valid diversity through reward-weighted rollout NLL and entropy regularization rather than replacing GT CE.
