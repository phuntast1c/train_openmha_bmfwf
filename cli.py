import os
from bmfwf import base, datasets, models, utils  # noqa: F401
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import CSVLogger

torch.set_float32_matmul_precision("high")

SAVE_DIR = utils.get_save_dir()


class MyLightningCLI(LightningCLI):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def before_instantiate_classes(self) -> None:
        if self.subcommand == "test":
            print(
                f"evaluating following metrics: {self.config[self.subcommand].model.init_args.metrics_test}"
            )

        # self.trainer.logger = CSVLogger(
        #     save_dir=SAVE_DIR,
        # )
        return super().before_instantiate_classes()

    def add_arguments_to_parser(self, parser):
        parser.add_argument(
            "--checkpoint",
            type=str,
            default=None,
            help="specify wandb run path to continue training from a previous run",
        )
        parser.add_argument(
            "--job_type",
            type=str,
            default="fit",
        )

        parser.link_arguments("model.init_args.batch_size", "data.init_args.batch_size")
        parser.link_arguments(
            "model.init_args.num_channels", "data.init_args.num_channels"
        )
        parser.link_arguments("data.init_args.fs", "model.init_args.fs")


def main():
    print(f"saving to {SAVE_DIR}")
    _ = MyLightningCLI(
        model_class=base.BaseLitModel,
        subclass_mode_model=True,
        seed_everything_default=1337,
        save_config_kwargs={"config_filename": os.path.join(SAVE_DIR, "config.yaml")},
        run=True,
        auto_configure_optimizers=False,
        trainer_defaults={
            "num_sanity_val_steps": 1,
            "log_every_n_steps": 10,
            "enable_progress_bar": True,
            "gradient_clip_val": 5.0,
            "deterministic": False,
            "benchmark": True,
            "devices": 1,
            "accelerator": "gpu",
            "logger": CSVLogger(save_dir=SAVE_DIR),
            "callbacks": [
                ModelCheckpoint(
                    monitor="val/loss",
                    save_last=True,
                    save_top_k=1,
                    dirpath=SAVE_DIR,
                    verbose=True,
                ),
                EarlyStopping(patience=10, monitor="val/loss"),
                LearningRateMonitor(),
            ],
            "default_root_dir": SAVE_DIR,
            "strategy": "ddp",
        },
    )


if __name__ == "__main__":
    main()
