import os

from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.loggers import CSVLogger
from s2s.window_module import WindowForecastingModule
from s2s.window_datamodule import WindowDataModule
# from climate_projection.module import ClimateBenchModule
# from climate_projection.datamodule import ClimateBenchDataModule


def main():
    # Initialize Lightning with the model and data modules, and instruct it to parse the config yml
    cli = LightningCLI(
        model_class=WindowForecastingModule,
        datamodule_class=WindowDataModule,
        seed_everything_default=42,
        save_config_callback=SaveConfigCallback,
        save_config_kwargs={"overwrite": True},
        run=False,
        parser_kwargs={"parser_mode": "yaml", "error_handler": None},
    )
    os.makedirs(cli.trainer.default_root_dir, exist_ok=True)

    # normalization = cli.datamodule.dataset_train.out_transform
    # mean_norm, std_norm = normalization.mean, normalization.std
    # mean_denorm, std_denorm = -mean_norm / std_norm, 1 / std_norm
    # cli.model.set_denormalization(mean_denorm, std_denorm)
    # cli.model.set_lat_lon(*cli.datamodule.get_lat_lon())
    # cli.model.set_test_clim(cli.datamodule.get_test_clim())
    
    cli.model.set_transforms(cli.datamodule.in_transforms, cli.datamodule.out_transforms)
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
    
    cli.trainer.logger = CSVLogger(
       name=logger_name,
       save_dir=os.path.join(cli.trainer.default_root_dir, logger_name)
    )

    # test the trained model
    # cli.trainer.test(cli.model, datamodule=cli.datamodule, ckpt_path='best')
    cli.trainer.test(cli.model, datamodule=cli.datamodule)


if __name__ == "__main__":
    main()