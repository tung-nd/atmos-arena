import os

from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from s2s.direct_module import DirectForecastingModule
from s2s.direct_datamodule import WindowForecastingDataModule

def get_best_checkpoint(dir):
    ckpt_paths = os.listdir(os.path.join(dir, 'checkpoints'))
    for ckpt_path in ckpt_paths:
        if 'epoch' in ckpt_path:
            return os.path.join(dir, 'checkpoints/', ckpt_path)

def main():
    # Initialize Lightning with the model and data modules, and instruct it to parse the config yml
    cli = LightningCLI(
        model_class=DirectForecastingModule,
        datamodule_class=WindowForecastingDataModule,
        seed_everything_default=42,
        save_config_callback=SaveConfigCallback,
        save_config_kwargs={"overwrite": True},
        run=False,
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None},
    )
    os.makedirs(cli.trainer.default_root_dir, exist_ok=True)

    normalization = cli.datamodule.out_transforms
    mean, std = normalization.mean, normalization.std
    std_denorm = 1 / std
    mean_denorm = -mean * std_denorm
    cli.model.set_denormalization(mean_denorm, std_denorm)
    cli.model.set_lat_lon(*cli.datamodule.get_lat_lon())
    cli.model.set_lead_time(cli.datamodule.hparams.lead_time)
    
    logger_name = cli.trainer.logger._name
    for i in range(len(cli.trainer.callbacks)):
        if isinstance(cli.trainer.callbacks[i], ModelCheckpoint):
            cli.trainer.callbacks[i] = ModelCheckpoint(
                dirpath=os.path.join(cli.trainer.default_root_dir, logger_name, 'checkpoints'),
                monitor=cli.trainer.callbacks[i].monitor,
                mode=cli.trainer.callbacks[i].mode,
                save_top_k=cli.trainer.callbacks[i].save_top_k,
                save_last=cli.trainer.callbacks[i].save_last,
                verbose=cli.trainer.callbacks[i].verbose,
                filename=cli.trainer.callbacks[i].filename,
                auto_insert_metric_name=cli.trainer.callbacks[i].auto_insert_metric_name
            )
    
    cli.trainer.logger = WandbLogger(
       name=logger_name,
       project=cli.trainer.logger._wandb_init['project'],
       save_dir=os.path.join(cli.trainer.default_root_dir, logger_name)
    )

    best_checkpoint_path = get_best_checkpoint(os.path.join(cli.trainer.default_root_dir, logger_name))
    cli.trainer.test(cli.model, datamodule=cli.datamodule, ckpt_path=best_checkpoint_path)


if __name__ == "__main__":
    main()