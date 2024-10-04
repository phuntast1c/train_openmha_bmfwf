import os
from abc import abstractmethod
from typing import Union

import lightning as pl
from . import losses, metrics, utils
import pandas as pd
import torch
from torchmetrics import MetricCollection


class BaseLitModel(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        batch_size: int,
        loss: str,
        metrics_test: Union[tuple, str],
        metrics_val: Union[tuple, str],
        model_name: str,
        my_optimizer: str = "AdamW",
        my_lr_scheduler: str = "ReduceLROnPlateau",
        **kwargs,
    ):
        super().__init__()

        self.learning_rate = lr
        self.batch_size = batch_size
        self.loss = loss
        self.model_name = model_name
        self.my_optimizer = my_optimizer
        self.my_lr_scheduler = my_lr_scheduler

        self.nan_batch_counter = 0.0

        self.criterion = getattr(losses, self.loss)(**kwargs)
        if isinstance(metrics_test, str):
            metrics_test = metrics_test.split(",")
        if isinstance(metrics_val, str):
            metrics_val = metrics_val.split(",")
        self.metric_collections = {
            "test": MetricCollection(
                [getattr(metrics, met)() for met in metrics_test if met != ""]
            ),
            # ).to(device=self.device),
            "val": MetricCollection(
                [getattr(metrics, met)() for met in metrics_val if met != ""]
            ),
            # ).to(device=self.device),
        }

        self.test_outputs = []

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def configure_optimizers(self):
        if self.my_optimizer == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        elif self.my_optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            raise NotImplementedError

        if self.my_lr_scheduler == "ReduceLROnPlateau":
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=3, factor=0.5
            )
        elif self.my_lr_scheduler == "OneCycleLR":
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=self.learning_rate,
                    total_steps=self.trainer.estimated_stepping_batches,
                    verbose=False,
                ),
                "interval": "step",
            }
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val/loss",
        }

    # skip batches including NaN gradients
    def on_after_backward(self) -> None:
        increase_nan_batch_counter = False
        for param in self.parameters():
            if param.grad is not None:
                nan_grads = torch.isnan(param.grad)
                if torch.any(nan_grads):
                    param.grad[nan_grads] = 0.0
                    increase_nan_batch_counter = True
        if increase_nan_batch_counter:
            self.nan_batch_counter += 1

        self.log(
            "ptl/nan_batch_counter",
            self.nan_batch_counter,
            batch_size=self.batch_size,
        )
        return super().on_after_backward()

    def training_step(self, batch, idx):
        output = self(batch.signals)
        loss = self.criterion(output, batch.signals, batch.meta)
        self.log_dict(
            {f"train/{x}": y for x, y in loss.items()},
            reduce_fx="mean",
            batch_size=self.batch_size,
            prog_bar=False,
        )
        return {"loss": loss["loss"]}

    # def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
    #     # # delete early stopping callback state
    #     checkpoint["callbacks"] = {
    #         x: y for x, y in checkpoint["callbacks"].items() if "EarlyStopping" not in x
    #     }
    #     # delete optimizer and LR scheduler states
    #     checkpoint["optimizer_states"][0]["param_groups"][0]["lr"] = self.learning_rate
    #     checkpoint["lr_schedulers"][0]["best"] = 1e6
    #     return super().on_load_checkpoint(checkpoint)

    # def configure_callbacks(self) -> Sequence[Callback] | Callback:
    #     return super().configure_callbacks()

    def validation_step(self, batch, idx):
        output = self(batch.signals)
        loss = self.criterion(output, batch.signals, batch.meta)
        self.log_dict(
            {f"val/{x}": y for x, y in loss.items()},
            reduce_fx="mean",
            batch_size=self.batch_size,
            prog_bar=False,
            sync_dist=True,
        )
        metrics_dict = {}
        for metric in self.metric_collections["val"]:
            (
                metrics_dict["metrics/val/enh_" + metric.__name__.upper()],
                metrics_dict["metrics/val/" + metric.__name__.upper()],
            ) = utils.get_measure_enhanced_noisy(output, batch.signals, metric)

        self.log_dict(metrics_dict)
        return {"loss_val": loss["loss"], "metrics": metrics_dict}

    def test_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> dict:
        """
        get loss, metric values and save .wav
        """

        signals, meta = batch.signals, batch.meta
        meta_dict = meta[0]
        meta_dict["dataloader_idx"] = dataloader_idx

        output = self(signals)

        # save outputs
        filename = meta[0]["filename"].replace(".wav", "_enh.wav")

        self.save_individual_wave(dataloader_idx, output, filename)

        inputs = signals["input"]
        preds = output["input_proc"]

        if self.trainer.datamodule.datasets_test[dataloader_idx].reference_available:
            loss = self.criterion(output, signals, meta)
            metrics_dict = {}
            metrics_dict.update({f"test/{x}": y.item() for x, y in loss.items()})
            target = signals["target"]
        else:
            target = None

        self.metric_collections["test"].update(
            inputs, preds, target, meta_dict, dataloader_idx
        )

    def on_test_epoch_end(self) -> None:
        # log average metric values
        metrics_output_dict = self.metric_collections["test"].compute()
        self.log_dict(metrics_output_dict, sync_dist=True)

        # collect all results and save to dataframe
        dataframe = [x.dataframe for x in self.metric_collections["test"].values()]
        dataframe = pd.concat(dataframe, axis=1)
        dataframe = dataframe.loc[:, ~dataframe.columns.duplicated()].copy()
        dataframe["filename"] = dataframe.index

        save_dir = os.path.join(self.logger.experiment.notes, "test")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dataframe.to_csv(os.path.join(save_dir, "results.csv"), index=False)

        self.metric_collections["test"].reset()

    def train_dataloader(self):
        return self.datamodule.train_dataloader()

    def save_individual_wave(self, dataloader_idx, output, filename):
        save_dir = os.path.join(
            self.logger.experiment.notes,
            "test",
            str(dataloader_idx),
        )
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        utils.save_wave(
            data=output["input_proc"][0],
            filename=os.path.join(save_dir, filename),
            fs=self.fs,
            normalize=True,
        )
