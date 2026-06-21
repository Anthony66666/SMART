import argparse
import json
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch_geometric.data import Batch

from smart.datasets.scalable_dataset import MultiDataset
from smart.model import SMARTCausalDiffusion
from smart.transforms import WaymoTargetBuilder
from smart.utils.config import load_config_act


def compute_type_thresholds(error_values, quantile=0.99):
    """Return conservative empirical per-type quantiles."""
    quantile = min(max(float(quantile), 0.0), 1.0)
    thresholds = {}
    for type_name in ('veh', 'ped', 'cyc'):
        values = sorted(float(value) for value in error_values.get(type_name, []))
        if not values:
            thresholds[type_name] = None
            continue
        rank = max(0, math.ceil(quantile * len(values)) - 1)
        thresholds[type_name] = values[rank]
    return thresholds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='configs/train/train_scalable_causal_diffusion_local.yaml',
    )
    parser.add_argument('--split', choices=('train', 'val'), default='train')
    parser.add_argument('--max_samples', type=int, default=1000)
    parser.add_argument('--quantile', type=float, default=0.99)
    parser.add_argument('--with_perturbation', action='store_true')
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--output_json', default='')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    config = load_config_act(args.config)
    data_config = config.Dataset
    raw_dir = getattr(data_config, f'{args.split}_raw_dir')
    processed_dir = getattr(data_config, f'{args.split}_processed_dir')
    dataset = MultiDataset(
        root=data_config.root,
        split=args.split,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        transform=WaymoTargetBuilder(
            config.Model.num_historical_steps,
            config.Model.decoder.num_future_steps,
        ),
        token_size=int(data_config.token_size),
    )
    model = SMARTCausalDiffusion(config.Model).train()
    error_values = {'veh': [], 'ped': [], 'cyc': []}
    type_names = ('veh', 'ped', 'cyc')
    sample_count = min(max(0, args.max_samples), len(dataset))

    with torch.no_grad():
        for sample_idx in range(sample_count):
            data = model._prepare_batch(Batch.from_data_list([dataset[sample_idx]]))
            view, _tokens, _valid, _anchor = model._build_ar_training_view(
                data,
                perturb=False,
            )
            if args.with_perturbation:
                model._perturb_ar_history_state(view)
            agent = view['agent']
            future_start = model.num_historical_steps
            future_end = (
                future_start
                + model.ar_prediction_tokens * model.ar_token_steps
            )
            future_positions = agent['position'][
                :,
                future_start:future_end,
                :2,
            ].reshape(
                -1,
                model.ar_prediction_tokens,
                model.ar_token_steps,
                2,
            )
            future_valid = agent['valid_mask'][
                :,
                future_start:future_end,
            ].reshape(
                -1,
                model.ar_prediction_tokens,
                model.ar_token_steps,
            )
            _ids, errors, _valid_targets, _endpoints = model._retokenize_future(
                future_positions=future_positions,
                future_valid=future_valid,
                start_positions=agent['position'][
                    :,
                    model.num_historical_steps - 1,
                    :2,
                ],
                start_headings=agent['heading'][
                    :,
                    model.num_historical_steps - 1,
                ],
                agent_types=agent['type'],
            )
            target_valid = agent['agent_valid_mask'][
                :,
                model.ar_history_tokens:
                model.ar_history_tokens + model.ar_prediction_tokens,
            ].bool()
            supervision_agents = (
                agent['valid_mask'][:, model.num_historical_steps - 1].bool()
                & (agent['category'] == 3)
                & (agent['type'] != 3)
            )
            eligible = (
                target_valid
                & future_valid.any(dim=-1)
                & supervision_agents.unsqueeze(-1)
            )
            for type_id, type_name in enumerate(type_names):
                type_mask = eligible & (agent['type'] == type_id).unsqueeze(-1)
                error_values[type_name].extend(
                    errors[type_mask].cpu().tolist()
                )

    thresholds = compute_type_thresholds(
        error_values,
        quantile=args.quantile,
    )
    result = {
        'config': args.config,
        'split': args.split,
        'samples': sample_count,
        'quantile': args.quantile,
        'with_perturbation': args.with_perturbation,
        'counts': {
            key: len(values)
            for key, values in error_values.items()
        },
        'thresholds': thresholds,
        'config_order': [
            thresholds['veh'],
            thresholds['ped'],
            thresholds['cyc'],
        ],
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
