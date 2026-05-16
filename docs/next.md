# Next

- Run a one-batch GPU smoke test with processed WOMD data and `configs/train/train_scalable_self_distill.yaml`.
- Compare `distill.enabled: false` against baseline CE on the same batch.
- Add batch-aware collision masking if training uses batch size greater than 1.
- Add map/offroad reward once lane polygon or road-edge geometry is available in the rollout scorer.
- Reproduce CAT-K as a named ablation using only `catk_ce` before full quality-diverse distillation experiments.
