# Progress

## 2026-05-16
- Added `SMARTSelfDistill` predictor with EMA teacher, closed-loop rollout scoring, CAT-K recovery, reward-weighted rollout NLL, KL distillation, and entropy regularization.
- Added decoder training APIs for rollout token sampling and forced-prefix scoring while preserving existing inference outputs.
- Added distillation utility functions, unit tests, self-distillation train config, and training export wiring.
- Verified static compilation, unit tests, import, config/model instantiation, a CPU one-batch self-distillation smoke run, and baseline inference output shape after decoder refactor in the `smart` conda environment.
