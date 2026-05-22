from __future__ import annotations

import argparse
import contextlib
import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from google.protobuf import text_format
from google.protobuf.json_format import MessageToDict
from tqdm import tqdm

from smart.datasets.scalable_dataset import MultiDataset
from smart.model import SMART
from smart.model import SMARTAutoregressiveDiffusion
from smart.model import SMARTDiffusion
from smart.model import SMARTJEPA
from smart.transforms import WaymoTargetBuilder
from smart.utils.config import load_config_act
from smart.utils.log import Logging
from smart.utils.torch_compat import register_checkpoint_safe_globals

# python eval_waymo_official.py \
#   --config configs/validation/validation_scalable.yaml \
#   --checkpoint checkpoints/baseline/epoch=05.ckpt \
#   --waymo_scenario_dir /path/to/waymo/validation_raw \
#   --scenario_index_path cache/waymo_val_index.pkl \
#   --output_json outputs/waymo_official_summary.json \
#   --per_scenario_jsonl outputs/waymo_official_per_scenario.jsonl


try:
    import tensorflow as tf
    from waymo_open_dataset.protos import scenario_pb2
    from waymo_open_dataset.protos import sim_agents_metrics_pb2
    from waymo_open_dataset.protos import sim_agents_submission_pb2
    from waymo_open_dataset.utils.sim_agents import submission_specs
    from waymo_open_dataset.wdl_limited.sim_agents_metrics import metrics as waymo_metrics
except ImportError as exc:
    raise ImportError(
        "Waymo official evaluation requires tensorflow and waymo-open-dataset. "
        "Install the Waymo package in this environment before running this script."
    ) from exc


register_checkpoint_safe_globals()
with contextlib.suppress(Exception):
    tf.config.set_visible_devices([], "GPU")


PREDICTOR_HASH = {
    "smart": SMART,
    "smart_diffusion": SMARTDiffusion,
    "smart_ar_diffusion": SMARTAutoregressiveDiffusion,
    "smart_jepa": SMARTJEPA,
}


@dataclass(frozen=True)
class SubmissionConfigCompat:
    current_time_index: int
    n_simulation_steps: int
    n_rollouts: int
    step_duration_seconds: float


def get_sim_agents_challenge_type():
    challenge_enum = getattr(submission_specs, "ChallengeType", None)
    if challenge_enum is None:
        return None
    return challenge_enum.SIM_AGENTS


def get_submission_config_compat() -> SubmissionConfigCompat:
    challenge_type = get_sim_agents_challenge_type()
    get_submission_config = getattr(submission_specs, "get_submission_config", None)
    if get_submission_config is not None:
        if challenge_type is None:
            config = get_submission_config()
        else:
            config = get_submission_config(challenge_type)
        return SubmissionConfigCompat(
            current_time_index=int(config.current_time_index),
            n_simulation_steps=int(config.n_simulation_steps),
            n_rollouts=int(config.n_rollouts),
            step_duration_seconds=float(config.step_duration_seconds),
        )

    return SubmissionConfigCompat(
        current_time_index=int(submission_specs.CURRENT_TIME_INDEX),
        n_simulation_steps=int(submission_specs.N_SIMULATION_STEPS),
        n_rollouts=int(submission_specs.N_ROLLOUTS),
        step_duration_seconds=float(submission_specs.STEP_DURATION_SECONDS),
    )


def get_sim_agent_ids_compat(scenario: scenario_pb2.Scenario):
    challenge_type = get_sim_agents_challenge_type()
    if challenge_type is None:
        return submission_specs.get_sim_agent_ids(scenario)
    return submission_specs.get_sim_agent_ids(scenario, challenge_type)


def validate_scenario_rollouts_compat(
    scenario_rollouts: sim_agents_submission_pb2.ScenarioRollouts,
    scenario: scenario_pb2.Scenario,
) -> None:
    challenge_type = get_sim_agents_challenge_type()
    if challenge_type is None:
        submission_specs.validate_scenario_rollouts(scenario_rollouts, scenario)
        return
    submission_specs.validate_scenario_rollouts(
        scenario_rollouts,
        scenario,
        challenge_type=challenge_type,
    )


def _load_metrics_config_from_search() -> sim_agents_metrics_pb2.SimAgentMetricsConfig:
    candidate_names = [
        "challenge_2025_sim_agents_config.textproto",
        "challenge_2024_config.textproto",
        "challenge_config.textproto",
    ]
    roots = []
    with contextlib.suppress(Exception):
        import waymo_open_dataset
        roots.append(Path(waymo_open_dataset.__file__).resolve().parent)
    roots.append(Path(waymo_metrics.__file__).resolve().parent)

    checked = []
    for root in roots:
        for candidate in [root, root / "wdl_limited" / "sim_agents_metrics", root / "sim_agents_metrics"]:
            for name in candidate_names:
                path = candidate / name
                checked.append(str(path))
                if path.exists():
                    config = sim_agents_metrics_pb2.SimAgentMetricsConfig()
                    text_format.Parse(path.read_text(encoding="utf-8"), config)
                    return config

    package_root = roots[0] if roots else Path(waymo_metrics.__file__).resolve().parent
    for path in package_root.rglob("*.textproto"):
        if path.name in candidate_names:
            config = sim_agents_metrics_pb2.SimAgentMetricsConfig()
            text_format.Parse(path.read_text(encoding="utf-8"), config)
            return config

    raise FileNotFoundError(
        "Could not locate a Waymo Sim Agents metrics config textproto. Checked: "
        + ", ".join(checked[:12])
    )


def load_metrics_config_compat():
    challenge_type = get_sim_agents_challenge_type()
    try:
        if challenge_type is None:
            return waymo_metrics.load_metrics_config()
        return waymo_metrics.load_metrics_config(challenge_type)
    except FileNotFoundError:
        return _load_metrics_config_from_search()


def compute_scenario_metrics_for_bundle_compat(metric_config, scenario, scenario_rollouts):
    challenge_type = get_sim_agents_challenge_type()
    if challenge_type is None:
        return waymo_metrics.compute_scenario_metrics_for_bundle(
            metric_config,
            scenario,
            scenario_rollouts,
        )
    return waymo_metrics.compute_scenario_metrics_for_bundle(
        metric_config,
        scenario,
        scenario_rollouts,
        challenge_type=challenge_type,
    )


class ScenarioLookup:
    def __init__(self, scenario_files: Sequence[Path], index_path: str | None = None) -> None:
        self.scenario_files = [Path(path) for path in scenario_files]
        self.index_path = Path(index_path) if index_path else None
        self.scenario_to_file: Dict[str, str] = {}
        self._active_file: str | None = None
        self._active_scenarios: Dict[str, scenario_pb2.Scenario] = {}
        self._load_index()

    def _load_index(self) -> None:
        if self.index_path is None or not self.index_path.exists():
            return
        with open(self.index_path, "rb") as handle:
            cached = pickle.load(handle)
        self.scenario_to_file = {str(key): str(value) for key, value in cached.items()}

    def _save_index(self) -> None:
        if self.index_path is None:
            return
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "wb") as handle:
            pickle.dump(self.scenario_to_file, handle)

    def build_index(self, required_ids: Iterable[str]) -> None:
        pending = set(required_ids) - set(self.scenario_to_file)
        if not pending:
            return

        for file_path in tqdm(self.scenario_files, desc="Index raw Waymo scenarios"):
            dataset = tf.data.TFRecordDataset(str(file_path), compression_type="")
            for record in dataset:
                scenario = scenario_pb2.Scenario()
                scenario.ParseFromString(record.numpy())
                scenario_id = scenario.scenario_id
                if scenario_id in pending:
                    self.scenario_to_file[scenario_id] = str(file_path)
                    pending.remove(scenario_id)
                    if not pending:
                        self._save_index()
                        return
        if pending:
            missing = ", ".join(sorted(list(pending))[:10])
            raise KeyError(f"Could not locate {len(pending)} scenario ids in raw Waymo data. Example missing ids: {missing}")

    def get(self, scenario_id: str) -> scenario_pb2.Scenario:
        if scenario_id not in self.scenario_to_file:
            raise KeyError(f"Scenario id {scenario_id} is missing from the raw-scenario index.")

        file_path = self.scenario_to_file[scenario_id]
        if file_path != self._active_file:
            scenarios: Dict[str, scenario_pb2.Scenario] = {}
            dataset = tf.data.TFRecordDataset(file_path, compression_type="")
            for record in dataset:
                scenario = scenario_pb2.Scenario()
                scenario.ParseFromString(record.numpy())
                scenarios[scenario.scenario_id] = scenario
            self._active_file = file_path
            self._active_scenarios = scenarios

        if scenario_id not in self._active_scenarios:
            raise KeyError(f"Scenario id {scenario_id} was indexed in {file_path} but could not be loaded from that file.")
        return self._active_scenarios[scenario_id]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SMART checkpoints with Waymo official Sim Agents metrics.")
    parser.add_argument("--config", type=str, default="configs/validation/validation_scalable.yaml")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--pretrain_ckpt", type=str, default="")
    parser.add_argument("--ckpt_path", type=str, default="")
    parser.add_argument("--waymo_scenario_dir", type=str, required=True,
                        help="Directory containing raw Waymo Scenario TFRecord shards for the evaluated split.")
    parser.add_argument("--scenario_index_path", type=str, default="",
                        help="Optional pickle cache for scenario_id -> TFRecord path.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--max_scenarios", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--n_rollouts", type=int, default=32)
    parser.add_argument("--z_mode", type=str, default="last_history", choices=["last_history", "log_future"])
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--per_scenario_jsonl", type=str, default="")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_checkpoint(args: argparse.Namespace) -> str:
    for candidate in (args.checkpoint, args.pretrain_ckpt, args.ckpt_path):
        if candidate:
            return candidate
    raise ValueError("A checkpoint path is required. Pass --checkpoint, --pretrain_ckpt, or --ckpt_path.")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list_scenario_files(root: str) -> List[Path]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Raw Waymo scenario directory does not exist: {root}")
    scenario_files = sorted(path for path in root_path.rglob("*") if path.is_file())
    if not scenario_files:
        raise FileNotFoundError(f"No raw Waymo scenario files found under: {root}")
    return scenario_files


def build_dataset(config) -> MultiDataset:
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
    )


def normalize_agent_ids(agent_ids: Sequence) -> List[int]:
    return [int(agent_id) for agent_id in agent_ids]


def build_track_index(scenario: scenario_pb2.Scenario) -> Dict[int, scenario_pb2.Track]:
    return {int(track.id): track for track in scenario.tracks}


def get_future_z(track: scenario_pb2.Track, current_time_index: int, future_steps: int, mode: str) -> List[float]:
    if mode == "log_future":
        future_states = track.states[current_time_index + 1: current_time_index + 1 + future_steps]
        if len(future_states) != future_steps:
            raise ValueError(
                f"Track {track.id} does not have {future_steps} future steps available for log_future z reconstruction."
            )
        return [float(state.center_z) for state in future_states]

    current_z = float(track.states[current_time_index].center_z)
    return [current_z] * future_steps


def prepare_data_for_inference(model, data, seed: int):
    model.noise = False
    data = model.match_token_map(data)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    data = model.sample_pt_pred(data)
    return data


def _validate_sim_agent_prediction(
    prediction: Dict[str, torch.Tensor],
    data,
    object_id: int,
    agent_index: int,
    future_steps: int,
    scenario_id: str,
) -> None:
    agent_ids = normalize_agent_ids(data["agent"]["id"])
    if int(object_id) not in set(agent_ids):
        raise ValueError(
            f"Scenario {scenario_id} object {object_id}: sim-agent is missing from the processed SMART sample."
        )

    if "pred_traj" not in prediction or "pred_head" not in prediction:
        raise ValueError(
            f"Scenario {scenario_id} object {object_id}: prediction is missing pred_traj or pred_head."
        )
    if "pred_valid_mask" not in prediction:
        raise ValueError(
            f"Scenario {scenario_id} object {object_id}: prediction is missing pred_valid_mask."
        )

    pred_traj = prediction["pred_traj"].detach().cpu().float()
    pred_head = prediction["pred_head"].detach().cpu().float()
    pred_valid = prediction["pred_valid_mask"].detach().cpu().bool()

    if agent_index < 0 or agent_index >= pred_traj.shape[0]:
        raise ValueError(
            f"Scenario {scenario_id} object {object_id}: agent index {agent_index} is outside pred_traj."
        )
    if pred_traj.shape[1] < future_steps or pred_head.shape[1] < future_steps:
        raise ValueError(
            f"Scenario {scenario_id} object {object_id}: prediction has fewer than {future_steps} future steps."
        )
    if pred_valid.shape[0] <= agent_index or pred_valid.shape[1] < future_steps:
        raise ValueError(
            f"Scenario {scenario_id} object {object_id}: pred_valid_mask does not cover {future_steps} future steps."
        )

    traj = pred_traj[agent_index, :future_steps]
    head = pred_head[agent_index, :future_steps]
    valid = pred_valid[agent_index, :future_steps]
    if not torch.isfinite(traj).all() or not torch.isfinite(head).all():
        raise ValueError(
            f"Scenario {scenario_id} object {object_id}: prediction contains non-finite values."
        )
    if not valid.all():
        missing = torch.nonzero(~valid, as_tuple=False).flatten().tolist()
        raise ValueError(
            f"Scenario {scenario_id} object {object_id}: pred_valid_mask misses future steps {missing[:10]}."
        )
    if torch.allclose(traj, torch.zeros_like(traj)):
        raise ValueError(
            f"Scenario {scenario_id} object {object_id}: zero fallback trajectory detected; refusing official export."
        )


def build_joint_scene(
    prediction: Dict[str, torch.Tensor],
    data,
    scenario: scenario_pb2.Scenario,
    z_mode: str,
) -> sim_agents_submission_pb2.JointScene:
    submission_config = get_submission_config_compat()
    future_steps = submission_config.n_simulation_steps
    sim_agent_ids = get_sim_agent_ids_compat(scenario)
    id_to_index = {agent_id: index for index, agent_id in enumerate(normalize_agent_ids(data["agent"]["id"]))}
    track_by_id = build_track_index(scenario)

    pred_traj = prediction["pred_traj"].detach().cpu().float()
    pred_head = prediction["pred_head"].detach().cpu().float()

    missing_ids = sorted(set(int(object_id) for object_id in sim_agent_ids) - set(id_to_index))
    if missing_ids:
        raise ValueError(
            "The processed SMART sample is missing sim-agent object IDs required by Waymo official evaluation: "
            f"{missing_ids[:10]}"
        )

    simulated_trajectories = []
    for object_id in sim_agent_ids:
        object_id = int(object_id)
        track = track_by_id[object_id]
        agent_index = id_to_index[object_id]
        _validate_sim_agent_prediction(
            prediction=prediction,
            data=data,
            object_id=object_id,
            agent_index=agent_index,
            future_steps=future_steps,
            scenario_id=str(scenario.scenario_id),
        )
        z_values = get_future_z(
            track=track,
            current_time_index=submission_config.current_time_index,
            future_steps=future_steps,
            mode=z_mode,
        )
        simulated_trajectories.append(
            sim_agents_submission_pb2.SimulatedTrajectory(
                center_x=pred_traj[agent_index, :, 0].tolist(),
                center_y=pred_traj[agent_index, :, 1].tolist(),
                center_z=z_values,
                heading=pred_head[agent_index].tolist(),
                object_id=object_id,
            )
        )
    return sim_agents_submission_pb2.JointScene(simulated_trajectories=simulated_trajectories)


def evaluate_scenario(
    model,
    data,
    scenario: scenario_pb2.Scenario,
    metric_config,
    n_rollouts: int,
    z_mode: str,
):
    joint_scenes = []
    for _ in range(n_rollouts):
        prediction = model.inference(data)
        joint_scenes.append(build_joint_scene(prediction, data, scenario, z_mode))

    scenario_rollouts = sim_agents_submission_pb2.ScenarioRollouts(
        scenario_id=scenario.scenario_id,
        joint_scenes=joint_scenes,
    )
    validate_scenario_rollouts_compat(scenario_rollouts, scenario)
    scenario_metrics = compute_scenario_metrics_for_bundle_compat(
        metric_config,
        scenario,
        scenario_rollouts,
    )
    bucketed_metrics = waymo_metrics.aggregate_metrics_to_buckets(metric_config, scenario_metrics)
    return scenario_metrics, bucketed_metrics


def proto_to_dict(message) -> Dict:
    return MessageToDict(message, preserving_proto_field_name=True)


def write_json(path: str, payload: Dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_jsonl(path: str, rows: Sequence[Dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    logger = Logging().log(level="INFO")

    config = load_config_act(args.config)
    checkpoint_path = resolve_checkpoint(args)
    device = resolve_device(args.device)
    dataset = build_dataset(config)

    predictor_cls = PREDICTOR_HASH[config.Model.predictor]
    model = predictor_cls(config.Model)
    model.load_params_from_file(filename=checkpoint_path, logger=logger, to_cpu=device.type == "cpu")
    model = model.to(device)
    model.eval()

    metric_config = load_metrics_config_compat()
    submission_config = get_submission_config_compat()
    if args.n_rollouts != submission_config.n_rollouts:
        raise ValueError(
            f"Waymo official Sim Agents metrics require exactly {submission_config.n_rollouts} rollouts, got {args.n_rollouts}."
        )
    if int(config.Model.decoder.num_future_steps) != submission_config.n_simulation_steps:
        raise ValueError(
            "This checkpoint/config does not match Waymo Sim Agents official horizon. "
            f"Expected {submission_config.n_simulation_steps} future steps, got {config.Model.decoder.num_future_steps}."
        )

    start_index = max(args.start_index, 0)
    end_index = len(dataset) if args.end_index is None else min(args.end_index, len(dataset))
    selected_indices = list(range(start_index, end_index))
    if args.max_scenarios is not None:
        selected_indices = selected_indices[:args.max_scenarios]
    if not selected_indices:
        raise ValueError("No validation scenarios selected. Check --start_index, --end_index, and --max_scenarios.")

    selected_scenario_ids = [Path(dataset.raw_paths[index]).stem for index in selected_indices]
    scenario_lookup = ScenarioLookup(
        scenario_files=list_scenario_files(args.waymo_scenario_dir),
        index_path=args.scenario_index_path or None,
    )
    scenario_lookup.build_index(selected_scenario_ids)

    all_scenario_metrics = []
    per_scenario_rows = []

    progress = tqdm(selected_indices, desc="Evaluate Waymo official metrics")
    with torch.no_grad():
        for offset, dataset_index in enumerate(progress):
            data = dataset[dataset_index]
            scenario_id = str(data["scenario_id"])
            scenario = scenario_lookup.get(scenario_id)

            data = prepare_data_for_inference(model, data, seed=args.seed + offset)
            data = data.to(device)

            scenario_metrics, bucketed_metrics = evaluate_scenario(
                model=model,
                data=data,
                scenario=scenario,
                metric_config=metric_config,
                n_rollouts=args.n_rollouts,
                z_mode=args.z_mode,
            )
            all_scenario_metrics.append(scenario_metrics)

            row = {
                "scenario_id": scenario_id,
                "scenario_metrics": proto_to_dict(scenario_metrics),
                "bucketed_metrics": proto_to_dict(bucketed_metrics),
            }
            per_scenario_rows.append(row)

            progress.set_postfix({
                "metametric": f"{scenario_metrics.metametric:.4f}",
                "minADE": f"{scenario_metrics.min_average_displacement_error:.4f}",
            })

    dataset_metrics = waymo_metrics.aggregate_scenario_metrics(all_scenario_metrics)
    bucketed_metrics = waymo_metrics.aggregate_metrics_to_buckets(metric_config, dataset_metrics)

    summary = {
        "num_scenarios": len(all_scenario_metrics),
        "checkpoint": checkpoint_path,
        "config": args.config,
        "waymo_scenario_dir": args.waymo_scenario_dir,
        "z_mode": args.z_mode,
        "dataset_metrics": proto_to_dict(dataset_metrics),
        "bucketed_metrics": proto_to_dict(bucketed_metrics),
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.output_json:
        write_json(args.output_json, summary)
    if args.per_scenario_jsonl:
        write_jsonl(args.per_scenario_jsonl, per_scenario_rows)


if __name__ == "__main__":
    main()
