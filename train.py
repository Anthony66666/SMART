
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from smart.callbacks import ValidationVisualizationCallback
from smart.callbacks.step_visualization import StepVisualizationCallback
from smart.utils.config import load_config_act
from smart.datamodules import MultiDataModule
from smart.model import SMART
from smart.model import SMARTDiffusion
from smart.model import SMARTAutoregressiveDiffusion
from smart.model import SMARTCausalDiffusion
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
    parser = ArgumentParser()
    Predictor_hash = {
        "smart": SMART,
        "smart_diffusion": SMARTDiffusion,
        "smart_ar_diffusion": SMARTAutoregressiveDiffusion,
        "smart_causal_diffusion": SMARTCausalDiffusion,
    }
    parser.add_argument('--config', type=str, default='configs/train/train_scalable.yaml')
    parser.add_argument('--pretrain_ckpt', type=str, default="")
    parser.add_argument('--ckpt_path', type=str, default="")
    parser.add_argument('--save_ckpt_path', type=str, default="")
    args = parser.parse_args()
    config = load_config_act(args.config)
    Predictor = Predictor_hash[config.Model.predictor]
    Data_config = config.Dataset
    datamodule = MultiDataModule(**vars(Data_config))

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
    monitor_metric = getattr(trainer_config, 'monitor_metric', 'val_cls_acc')
    monitor_mode = getattr(trainer_config, 'monitor_mode', 'max')
    model_checkpoint = ModelCheckpoint(dirpath=args.save_ckpt_path,
                                       filename="{epoch:02d}",
                                       monitor=monitor_metric,
                                       every_n_epochs=1,
                                       save_top_k=5,
                                       mode=monitor_mode)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks = [model_checkpoint, lr_monitor]
    visualization_config = getattr(config, 'Visualization', None)
    if visualization_config is not None and getattr(visualization_config, 'enabled', False):
        callbacks.append(
            ValidationVisualizationCallback(
                enabled=True,
                interval_epochs=getattr(visualization_config, 'interval_epochs', 1),
                sample_indices=getattr(visualization_config, 'sample_indices', [0]),
                output_dir=getattr(visualization_config, 'output_dir', 'outputs/val_visualizations'),
                max_agents=getattr(visualization_config, 'max_agents', 0),
            )
        )
        step_viz_cfg = getattr(visualization_config, 'step_viz', None)
        if step_viz_cfg is not None and getattr(step_viz_cfg, 'enabled', False):
            callbacks.append(
                StepVisualizationCallback(
                    interval_steps=getattr(step_viz_cfg, 'interval_steps', 2000),
                    sample_indices=getattr(step_viz_cfg, 'sample_indices',
                                           getattr(visualization_config, 'sample_indices', [0])),
                    output_dir=getattr(step_viz_cfg, 'output_dir', 'outputs/step_visualizations'),
                    max_agents=getattr(visualization_config, 'max_agents', 0),
                )
            )
    trainer = pl.Trainer(accelerator=trainer_config.accelerator, devices=trainer_config.devices,
                         strategy=strategy,
                         accumulate_grad_batches=trainer_config.accumulate_grad_batches,
                         num_nodes=trainer_config.num_nodes,
                         precision=getattr(trainer_config, 'precision', 32),
                         callbacks=callbacks,
                         max_epochs=trainer_config.max_epochs,
                         limit_val_batches=getattr(trainer_config, 'limit_val_batches', 1.0),
                         check_val_every_n_epoch=getattr(trainer_config, 'check_val_every_n_epoch', 1),
                         num_sanity_val_steps=0,
                         gradient_clip_val=0.5)
    if args.ckpt_path == "":
        trainer.fit(model,
                    datamodule)
    else:
        trainer.fit(model,
                    datamodule,
                    ckpt_path=args.ckpt_path)
