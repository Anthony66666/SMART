import argparse
import copy
import csv
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import smoke_causal_guidance_modes as smoke
from smart.utils.config import load_config_act


TUNABLE_ATTRIBUTES = (
    "guidance_safe_topk",
    "safety_topk",
    "safety_energy_weight",
    "commit_safety_weight",
    "commit_speed_energy_weight",
    "commit_min_speed_ratio",
    "commit_speed_threshold",
    "commit_speed_reference_decay",
    "lane_distance_energy_weight",
    "lane_heading_energy_weight",
    "dynamics_energy_weight",
    "collision_energy_weight",
)


DEFAULT_SETTINGS = (
    {
        "label": "none_sample",
        "mode": "none",
        "overrides": {},
    },
    {
        "label": "safe_top1_no_energy",
        "mode": "safe",
        "overrides": {
            "safety_energy_weight": 0.0,
            "commit_safety_weight": 0.0,
        },
    },
    {
        "label": "safe_default",
        "mode": "safe",
        "overrides": {},
    },
    {
        "label": "safe_energy025",
        "mode": "safe",
        "overrides": {
            "safety_energy_weight": 0.25,
            "commit_safety_weight": 0.25,
        },
    },
    {
        "label": "safe_energy050",
        "mode": "safe",
        "overrides": {
            "safety_energy_weight": 0.5,
            "commit_safety_weight": 0.5,
        },
    },
    {
        "label": "safe_commit025_tail1",
        "mode": "safe",
        "overrides": {
            "commit_safety_weight": 0.25,
        },
    },
    {
        "label": "safe_commit050_tail1",
        "mode": "safe",
        "overrides": {
            "commit_safety_weight": 0.5,
        },
    },
    {
        "label": "safe_commit050_speed4",
        "mode": "safe",
        "overrides": {
            "commit_safety_weight": 0.5,
            "commit_speed_energy_weight": 4.0,
        },
    },
    {
        "label": "safe_commit100_speed4",
        "mode": "safe",
        "overrides": {
            "commit_speed_energy_weight": 4.0,
        },
    },
    {
        "label": "safe_topk8_energy050",
        "mode": "safe",
        "overrides": {
            "guidance_safe_topk": 8,
            "safety_topk": 8,
            "safety_energy_weight": 0.5,
            "commit_safety_weight": 0.5,
        },
    },
    {
        "label": "safe_topk32_energy050",
        "mode": "safe",
        "overrides": {
            "guidance_safe_topk": 32,
            "safety_topk": 32,
            "safety_energy_weight": 0.5,
            "commit_safety_weight": 0.5,
        },
    },
    {
        "label": "safe_no_collision_energy050",
        "mode": "safe",
        "overrides": {
            "safety_energy_weight": 0.5,
            "commit_safety_weight": 0.5,
            "collision_energy_weight": 0.0,
        },
    },
)


def _parse_override_values(value):
    if value is None or str(value).strip() == "":
        return []
    overrides = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Override must be name=value, got: {item}")
        name, raw = item.split("=", 1)
        name = name.strip()
        if name not in TUNABLE_ATTRIBUTES:
            raise ValueError(f"Unsupported override attribute: {name}")
        raw = raw.strip()
        if name in {"guidance_safe_topk", "safety_topk"}:
            cast_value = int(raw)
        else:
            cast_value = float(raw)
        overrides.append((name, cast_value))
    return overrides


def _parse_setting_specs(values):
    if not values:
        return list(DEFAULT_SETTINGS)
    settings = []
    for spec in values:
        parts = [part.strip() for part in spec.split(":") if part.strip()]
        if not parts:
            continue
        label = parts[0]
        mode = "safe"
        override_part = ""
        if len(parts) >= 2:
            if "=" in parts[1]:
                override_part = parts[1]
            else:
                mode = parts[1].lower()
        if len(parts) >= 3:
            override_part = parts[2]
        if mode not in {"none", "safe"}:
            raise ValueError(f"Unsupported sweep mode for {label}: {mode}")
        settings.append(
            {
                "label": label,
                "mode": mode,
                "overrides": dict(_parse_override_values(override_part)),
            }
        )
    if not settings:
        raise RuntimeError("No sweep settings selected.")
    return settings


def _restore_defaults(model, defaults):
    for attribute, value in defaults.items():
        setattr(model, attribute, value)


def _apply_setting(model, defaults, setting):
    _restore_defaults(model, defaults)
    for attribute, value in setting.get("overrides", {}).items():
        setattr(model, attribute, value)


def _finite_or_none(value):
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def _score_summary(row, target_ratio):
    ade = float(row.get("ade_mean", 0.0))
    fde = float(row.get("fde_mean", 0.0))
    speed_ratio = float(row.get("moving_speed_ratio_mean", 0.0))
    invalidity = (
        float(row.get("guidance_hard_collision_rate_mean", 0.0))
        + float(row.get("guidance_offroad_rate_mean", 0.0))
        + 0.1 * float(row.get("guidance_dynamics_energy_mean", 0.0))
    )
    return (
        ade
        + 0.2 * fde
        + 2.0 * abs(speed_ratio - float(target_ratio))
        + invalidity
    )


def _ranked_summary_rows(summary, target_ratio):
    rows = smoke._summary_rows(summary)
    for row in rows:
        row["selection_score"] = _score_summary(row, target_ratio)
    return sorted(rows, key=lambda item: item["selection_score"])


def _settings_rows(settings):
    return [
        {
            "label": setting["label"],
            "base_mode": setting["mode"],
            "overrides": json.dumps(setting.get("overrides", {}), sort_keys=True),
        }
        for setting in settings
    ]


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _smoke_args(args):
    return SimpleNamespace(
        seed=int(args.seed),
        target_agents=args.target_agents,
        max_target_agents=int(args.max_target_agents),
        target_time_window=args.target_time_window,
        target_spec=args.target_spec,
        ego_interaction_alpha=None,
        target_event_eta=None,
        edit_gamma=None,
        invalid_beta=None,
        path_corridor_width=None,
        conflict_tta_threshold=None,
        near_miss_distance=None,
        ttc_threshold=None,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Sweep inference-only guidance settings for causal diffusion."
    )
    parser.add_argument(
        "--config",
        default="configs/train/train_scalable_causal_diffusion_local.yaml",
    )
    parser.add_argument(
        "--ckpt",
        default="/mnt/d/casual_v2_epoch=04.ckpt",
    )
    parser.add_argument("--raw-dir", default="data/valid_demo")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--indices", default="")
    parser.add_argument("--num-scenes", type=int, default=11)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-agents", default="")
    parser.add_argument("--max-target-agents", type=int, default=1)
    parser.add_argument("--target-time-window", default="2,4")
    parser.add_argument("--target-spec", default="ego_risk")
    parser.add_argument(
        "--target-speed-ratio",
        type=float,
        default=0.8,
        help="Speed-ratio target used only for ranking summaries.",
    )
    parser.add_argument(
        "--setting",
        action="append",
        default=[],
        help=(
            "Custom setting as label[:mode][:attr=value,attr=value]. "
            "Mode must be none or safe."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/causal_guidance_epoch04_setting_sweep",
    )
    args = parser.parse_args()

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    config = load_config_act(args.config)
    model_config = copy.deepcopy(config)
    model, load_info = smoke._load_model(model_config, args.ckpt, device)
    dataset = smoke._load_dataset(config, args.raw_dir)
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty.")

    scene_indices = smoke._parse_indices(
        args.index,
        args.indices,
        args.num_scenes,
        len(dataset),
    )
    if not scene_indices:
        raise RuntimeError("No valid scene indices selected.")

    settings = _parse_setting_specs(args.setting)
    defaults = {
        attribute: getattr(model, attribute)
        for attribute in TUNABLE_ATTRIBUTES
        if hasattr(model, attribute)
    }
    run_args = _smoke_args(args)
    records = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for setting in settings:
        _apply_setting(model, defaults, setting)
        for scene_index in scene_indices:
            print(
                "[sweep] "
                f"setting={setting['label']} "
                f"mode={setting['mode']} "
                f"index={scene_index}"
            )
            record = smoke._run_mode(
                model,
                dataset,
                scene_index,
                device,
                setting["mode"],
                run_args,
            )
            record["mode"] = setting["label"]
            record["base_mode"] = setting["mode"]
            record["setting_overrides"] = json.dumps(
                setting.get("overrides", {}),
                sort_keys=True,
            )
            for attribute in TUNABLE_ATTRIBUTES:
                if hasattr(model, attribute):
                    record[f"setting_{attribute}"] = _finite_or_none(
                        getattr(model, attribute)
                    )
            records.append(record)

    summary = smoke._aggregate_results(records)
    ranked_rows = _ranked_summary_rows(summary, args.target_speed_ratio)
    output = {
        "config": args.config,
        "ckpt": args.ckpt,
        "raw_dir": args.raw_dir,
        "indices": scene_indices,
        "seed": args.seed,
        "device": args.device,
        "load_info": load_info,
        "defaults": defaults,
        "settings": settings,
        "summary": summary,
        "ranked_modes": ranked_rows,
        "results": records,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(output, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    smoke._write_csv(output_dir / "records.csv", records)
    smoke._write_csv(output_dir / "summary.csv", smoke._summary_rows(summary))
    _write_csv(output_dir / "ranked_summary.csv", ranked_rows)
    _write_csv(output_dir / "settings.csv", _settings_rows(settings))
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
