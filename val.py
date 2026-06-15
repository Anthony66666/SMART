
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from torch_geometric.loader import DataLoader
from smart.datasets.scalable_dataset import MultiDataset
from smart.model import SMART
from smart.model import SMARTDiffusion
from smart.model import SMARTAutoregressiveDiffusion
from smart.model import SMARTCausalDiffusion
from smart.model import SMARTCausalFlowMatching
from smart.model import SMARTEmbeddedLanguageFlow
from smart.model import SMARTHybridDiffusion
from smart.transforms import WaymoTargetBuilder
from smart.utils.config import load_config_act
from smart.utils.log import Logging
from smart.utils.torch_compat import register_checkpoint_safe_globals

register_checkpoint_safe_globals()


def build_strategy(trainer_config):
    strategy_name = getattr(trainer_config, 'strategy', None)
    if strategy_name in (None, "", "auto"):
        return "auto"
    if strategy_name == "ddp_find_unused_parameters_false":
        return DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True)
    if strategy_name == "ddp_find_unused_parameters_true":
        return DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True)
    return strategy_name


if __name__ == '__main__':
    pl.seed_everything(2, workers=True)
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/validation/validation_scalable.yaml")
    parser.add_argument('--pretrain_ckpt', type=str, default="")
    parser.add_argument('--ckpt_path', type=str, default="")
    parser.add_argument('--save_ckpt_path', type=str, default="")
    args = parser.parse_args()
    config = load_config_act(args.config)
    Predictor_hash = {
        "smart": SMART,
        "smart_diffusion": SMARTDiffusion,
        "smart_ar_diffusion": SMARTAutoregressiveDiffusion,
        "smart_causal_diffusion": SMARTCausalDiffusion,
        "smart_causal_flow_matching": SMARTCausalFlowMatching,
        "smart_elf": SMARTEmbeddedLanguageFlow,
        "smart_hybrid_diffusion": SMARTHybridDiffusion,
    }

    data_config = config.Dataset
    val_dataset = {
        "scalable": MultiDataset,
    }[data_config.dataset](root=data_config.root, split='val',
                           raw_dir=data_config.val_raw_dir,
                           processed_dir=data_config.val_processed_dir,
                           transform=WaymoTargetBuilder(config.Model.num_historical_steps, config.Model.decoder.num_future_steps),
                           token_size=int(getattr(data_config, "token_size", getattr(config.Model.decoder, "token_size", 512))))
    batch_size = int(getattr(data_config, "batch_size", getattr(data_config, "val_batch_size", 1)))
    dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=data_config.num_workers,
                            pin_memory=data_config.pin_memory, persistent_workers=True if data_config.num_workers > 0 else False)
    Predictor = Predictor_hash[config.Model.predictor]
    if args.pretrain_ckpt == "":
        model = Predictor(config.Model)
    else:
        logger = Logging().log(level='DEBUG')
        model = Predictor(config.Model)
        model.load_params_from_file(filename=args.pretrain_ckpt,
                                    logger=logger,
                                    to_cpu=True)

    trainer_config = config.Trainer
    strategy = build_strategy(trainer_config)
    trainer = pl.Trainer(accelerator=trainer_config.accelerator,
                         devices=trainer_config.devices,
                         strategy=strategy,
                         num_nodes=getattr(trainer_config, 'num_nodes', 1),
                         precision=getattr(trainer_config, 'precision', 32),
                         num_sanity_val_steps=0)
    trainer.validate(model, dataloader)
