from pathlib import Path
import time
from typing import Iterable, Optional

import pytorch_lightning as pl
import torch
from torch_geometric.data import Batch

from smart.callbacks.validation_visualization import save_validation_visualization


class StepVisualizationCallback(pl.Callback):
    """Visualize predictions every N training steps."""

    def __init__(
        self,
        interval_steps: int = 2000,
        sample_indices: Optional[Iterable[int]] = None,
        output_dir: str = "outputs/step_visualizations",
        max_agents: int = 0,
    ):
        super().__init__()
        self.interval_steps = max(1, int(interval_steps))
        self.sample_indices = [int(i) for i in (sample_indices or [0])]
        self.output_dir = output_dir
        self.max_agents = int(max_agents)
        self._last_step_logged = -1

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not trainer.is_global_zero:
            return
        global_step = trainer.global_step
        if global_step <= 0 or global_step % self.interval_steps != 0:
            return
        if global_step == self._last_step_logged:
            return
        self._last_step_logged = global_step

        datamodule = trainer.datamodule
        if datamodule is None or not hasattr(datamodule, "val_dataset"):
            return
        dataset = datamodule.val_dataset

        predictor_name = pl_module.model_config.predictor
        debug_logging = bool(
            getattr(getattr(pl_module.model_config, 'diffusion', None), 'debug_validation_logging', False)
        )
        output_root = Path(self.output_dir) / predictor_name / f"step_{global_step:06d}"
        output_root.mkdir(parents=True, exist_ok=True)
        if debug_logging:
            print(
                f"[StepVisualization][rank=0 step={global_step}] start samples={self.sample_indices}",
                flush=True,
            )

        was_training = pl_module.training
        pl_module.eval()
        try:
            with torch.no_grad():
                for sample_index in self.sample_indices:
                    if sample_index < 0 or sample_index >= len(dataset):
                        continue
                    sample_start = time.perf_counter()
                    if debug_logging:
                        print(
                            f"[StepVisualization][rank=0 step={global_step}] sample_start index={sample_index}",
                            flush=True,
                        )
                    graph = dataset[sample_index]
                    batch = Batch.from_data_list([graph]).to(pl_module.device)
                    prepared = self._prepare_batch(pl_module, batch)
                    prediction = pl_module.inference(prepared)
                    if prediction is None:
                        if debug_logging:
                            print(
                                f"[StepVisualization][rank=0 step={global_step}] sample_skip index={sample_index} prediction=None",
                                flush=True,
                            )
                        continue
                    scenario_id = _scenario_id(graph)
                    filename = f"idx_{sample_index:05d}_{scenario_id}.png"
                    save_validation_visualization(
                        data=prepared.cpu(),
                        prediction={
                            k: v.detach().cpu() if torch.is_tensor(v) else v
                            for k, v in prediction.items()
                        },
                        output_path=output_root / filename,
                        title=f"{predictor_name} step={global_step} idx={sample_index} scenario={scenario_id}",
                        max_agents=self.max_agents,
                    )
                    if debug_logging:
                        print(
                            f"[StepVisualization][rank=0 step={global_step}] sample_done index={sample_index} "
                            f"elapsed={time.perf_counter() - sample_start:.2f}s output={output_root / filename}",
                            flush=True,
                        )
        finally:
            if was_training:
                pl_module.train()

    @staticmethod
    def _prepare_batch(pl_module, batch: Batch) -> Batch:
        if hasattr(pl_module, "_prepare_batch"):
            return pl_module._prepare_batch(batch)
        data = pl_module.match_token_map(batch)
        data = pl_module.sample_pt_pred(data)
        if isinstance(data, Batch):
            data["agent"]["av_index"] += data["agent"]["ptr"][:-1]
        return data


def _scenario_id(graph) -> str:
    for attr in ["scenario_id", "scenario_id"]:
        val = None
        if hasattr(graph, attr):
            val = getattr(graph, attr, None)
        elif hasattr(graph, "get") and graph.get(attr) is not None:
            val = graph[attr]
        if val is not None:
            return str(val).replace("/", "_")
    return "unknown"
