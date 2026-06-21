from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch_geometric.data import Batch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from smart.callbacks.validation_visualization import save_validation_visualization
from smart.datasets.scalable_dataset import MultiDataset
from smart.model import (
    SMART,
    SMARTAutoregressiveDiffusion,
    SMARTCausalDiffusion,
    SMARTCausalFlowMatching,
    SMARTDiffusion,
    SMARTEmbeddedLanguageFlow,
    SMARTHybridDiffusion,
    SMARTActionChunkDiffusion,
    SMARTContinuousActionDiffusion,
    SMARTDiscreteDiffusionPolicy,
)
from smart.transforms import WaymoTargetBuilder
from smart.utils.config import load_config_act
from smart.utils.log import Logging
from smart.utils.torch_compat import register_checkpoint_safe_globals


@dataclass(frozen=True)
class ModelSpec:
    name: str
    config_path: str
    ckpt_path: str


PREDICTORS = {
    "smart": SMART,
    "smart_diffusion": SMARTDiffusion,
    "smart_ar_diffusion": SMARTAutoregressiveDiffusion,
    "smart_causal_diffusion": SMARTCausalDiffusion,
    "smart_causal_flow_matching": SMARTCausalFlowMatching,
    "smart_elf": SMARTEmbeddedLanguageFlow,
    "smart_hybrid_diffusion": SMARTHybridDiffusion,
    "smart_action_chunk_diffusion": SMARTActionChunkDiffusion,
    "smart_continuous_action_diffusion": SMARTContinuousActionDiffusion,
    "smart_discrete_diffusion_policy": SMARTDiscreteDiffusionPolicy,
}


def parse_model_spec(value: str) -> ModelSpec:
    parts = value.split("=", 2)
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError(
            "model spec must be name=config_path=checkpoint_path"
        )
    return ModelSpec(name=parts[0], config_path=parts[1], ckpt_path=parts[2])


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _last_valid_fde(displacement: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    values = []
    for agent_idx in range(displacement.shape[0]):
        indices = torch.nonzero(valid_mask[agent_idx], as_tuple=False).squeeze(-1)
        if indices.numel() == 0:
            continue
        values.append(displacement[agent_idx, indices[-1]])
    if not values:
        return displacement.new_tensor(float("nan"))
    return torch.stack(values).mean()


def compute_metrics(prediction: dict, eval_agent_mask: torch.Tensor, eval_valid: torch.Tensor) -> dict:
    pred = prediction["pred_traj"][eval_agent_mask]
    gt = prediction["gt"][eval_agent_mask]
    valid = eval_valid[eval_agent_mask].bool()
    pred_valid = prediction.get("pred_valid_mask")
    if pred_valid is not None:
        valid = valid & pred_valid[eval_agent_mask].bool()
    if pred.numel() == 0 or not valid.any():
        return {
            "ade": float("nan"),
            "fde": float("nan"),
            "coverage": 0.0,
            "num_agents": int(eval_agent_mask.sum().item()),
            "num_valid_frames": 0,
        }
    displacement = torch.norm(pred - gt, dim=-1)
    ade = displacement[valid].mean()
    fde = _last_valid_fde(displacement, valid)
    requested_valid = eval_valid[eval_agent_mask].bool()
    coverage = valid.float().sum() / requested_valid.float().sum().clamp_min(1.0)
    return {
        "ade": float(ade.item()),
        "fde": float(fde.item()),
        "coverage": float(coverage.item()),
        "num_agents": int(eval_agent_mask.sum().item()),
        "num_valid_frames": int(valid.sum().item()),
    }


def _prepare_batch(model, batch: Batch) -> Batch:
    if hasattr(model, "_prepare_batch"):
        return model._prepare_batch(batch)
    data = model.match_token_map(batch)
    data = model.sample_pt_pred(data)
    if isinstance(data, Batch):
        data["agent"]["av_index"] += data["agent"]["ptr"][:-1]
    return data


def _scenario_id(graph) -> str:
    scenario_id = getattr(graph, "scenario_id", None)
    if scenario_id is None and hasattr(graph, "get"):
        scenario_id = graph.get("scenario_id", None)
    if scenario_id is None and hasattr(graph, "__contains__") and "scenario_id" in graph:
        scenario_id = graph["scenario_id"]
    if scenario_id is None:
        return "unknown"
    return str(scenario_id).replace("/", "_")


def _build_dataset(config):
    data_config = config.Dataset
    return MultiDataset(
        root=data_config.root,
        split="val",
        raw_dir=data_config.val_raw_dir,
        processed_dir=data_config.val_processed_dir,
        transform=WaymoTargetBuilder(
            config.Model.num_historical_steps,
            config.Model.decoder.num_future_steps,
        ),
        token_size=int(
            getattr(
                data_config,
                "token_size",
                getattr(config.Model.decoder, "token_size", 512),
            )
        ),
    )


def _load_model(spec: ModelSpec, device: torch.device):
    config = load_config_act(spec.config_path)
    predictor = PREDICTORS[config.Model.predictor]
    model = predictor(config.Model)
    if spec.ckpt_path:
        logger = Logging().log(level="DEBUG")
        model.load_params_from_file(spec.ckpt_path, logger=logger, to_cpu=True)
    model.to(device)
    model.eval()
    return config, model


def evaluate_model(
    spec: ModelSpec,
    indices: Iterable[int],
    output_dir: Path,
    device: torch.device,
    max_agents: int,
) -> list[dict]:
    config, model = _load_model(spec, device)
    dataset = _build_dataset(config)
    rows = []
    model_dir = output_dir / spec.name
    model_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for index in indices:
            graph = dataset[int(index)]
            batch = Batch.from_data_list([graph]).to(device)
            prepared = _prepare_batch(model, batch)
            prediction = model.inference(prepared)
            scenario_id = _scenario_id(graph)
            if prediction is None:
                rows.append({
                    "model": spec.name,
                    "index": int(index),
                    "scenario_id": scenario_id,
                    "ade": float("nan"),
                    "fde": float("nan"),
                    "coverage": 0.0,
                    "num_agents": 0,
                    "num_valid_frames": 0,
                    "visualization": "",
                })
                continue
            eval_agent_mask = model._metric_agent_mask(prepared)
            eval_valid = model._validation_eval_valid_mask(prepared, prediction)
            metrics = compute_metrics(prediction, eval_agent_mask, eval_valid)
            image_path = model_dir / f"idx_{int(index):05d}_{scenario_id}.png"
            save_validation_visualization(
                data=prepared.detach().cpu() if hasattr(prepared, "detach") else prepared.cpu(),
                prediction={
                    key: value.detach().cpu() if torch.is_tensor(value) else value
                    for key, value in prediction.items()
                },
                output_path=image_path,
                title=f"{spec.name} idx={int(index)} scenario={scenario_id}",
                max_agents=max_agents,
            )
            rows.append({
                "model": spec.name,
                "index": int(index),
                "scenario_id": scenario_id,
                **metrics,
                "visualization": str(image_path),
            })
    return rows


def summarize_records(rows: list[dict]) -> list[dict]:
    by_model: dict[str, list[dict]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)
    summary = []
    for model_name, model_rows in sorted(by_model.items()):
        finite_ade = [
            float(row["ade"])
            for row in model_rows
            if torch.isfinite(torch.tensor(float(row["ade"])))
        ]
        finite_fde = [
            float(row["fde"])
            for row in model_rows
            if torch.isfinite(torch.tensor(float(row["fde"])))
        ]
        summary.append({
            "model": model_name,
            "num_scenes": len(model_rows),
            "ade_mean": sum(finite_ade) / max(len(finite_ade), 1),
            "fde_mean": sum(finite_fde) / max(len(finite_fde), 1),
            "coverage_mean": sum(float(row["coverage"]) for row in model_rows) / max(len(model_rows), 1),
        })
    return summary


def _parse_indices(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    register_checkpoint_safe_globals()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", type=parse_model_spec, required=True)
    parser.add_argument("--indices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--output-dir", default="outputs/model_comparison_1000")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-agents", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    indices = _parse_indices(args.indices)
    device = torch.device(args.device)
    records = []
    for spec in args.model:
        records.extend(
            evaluate_model(
                spec=spec,
                indices=indices,
                output_dir=output_dir,
                device=device,
                max_agents=args.max_agents,
            )
        )
    summary = summarize_records(records)
    _write_csv(output_dir / "records.csv", records)
    _write_csv(output_dir / "summary.csv", summary)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "models": [spec.__dict__ for spec in args.model],
                "indices": indices,
                "records_csv": str(output_dir / "records.csv"),
                "summary_csv": str(output_dir / "summary.csv"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
