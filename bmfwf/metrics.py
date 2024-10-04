import pandas as pd
import torch
import torchmetrics
from pesq import pesq as pesq_
from pypesq import pesq as pesq_nb_raw
from pystoi import stoi as stoi_
from torch import tensor
from .utils import dcn


class BaseMetric(torchmetrics.Metric):
    def __init__(
        self,
        *args,
        requires_reference: bool = True,
        requires_numpy: bool = True,
        name: str = "",
        **kwargs,
    ):
        super().__init__(
            compute_on_cpu=True,
            *args,
            **kwargs,
        )
        self.requires_reference = requires_reference
        self.requires_numpy = requires_numpy
        self.add_state("noisy_total", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("enhanced_total", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("numel", default=tensor(0), dist_reduce_fx="sum")
        self.dataframe = pd.DataFrame()

    def _get_values(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def update(
        self,
        inputs: torch.Tensor,
        preds: torch.Tensor,
        target: torch.Tensor,
        meta: dict,
        dataloader_idx: int,
    ) -> None:
        if self.requires_reference and target is None:
            noisy = torch.as_tensor(torch.nan)
            enhanced = torch.as_tensor(torch.nan)
        elif self.requires_numpy:
            noisy = self._get_values(dcn(inputs), dcn(target))
            enhanced = self._get_values(dcn(preds), dcn(target))
        else:
            noisy = self._get_values(inputs, target)
            enhanced = self._get_values(preds, target)

        self.noisy_total = (self.noisy_total + noisy).to(device=inputs.device)
        self.enhanced_total = (self.enhanced_total + enhanced).to(device=inputs.device)
        self.numel = (self.numel + enhanced.numel()).to(device=inputs.device)

        self.update_dataframe(meta, dataloader_idx, noisy, enhanced)

    def update_dataframe(self, meta, dataloader_idx, noisy, enhanced):
        self.dataframe = pd.concat(
            [
                self.dataframe,
                pd.DataFrame(
                    {
                        "dataloader_idx": dataloader_idx,
                        self.__class__.__name__ + "_noisy": dcn(noisy),
                        self.__class__.__name__ + "_enhanced": dcn(enhanced),
                        **meta,
                    },
                    index=[meta["filename"]],
                ),
            ]
        )

    def compute(self):
        # move to cuda required due to lightning and torchmetrics quirk...
        return {
            "noisy": (self.noisy_total.float() / self.numel).to("cuda"),
            "enhanced": (self.enhanced_total.float() / self.numel).to("cuda"),
        }


class BasePESQ(BaseMetric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = True

    def __init__(
        self,
        *args,
        mode: str = "wb",
        fs: int = 16000,
        **kwargs,
    ):
        super().__init__(
            requires_reference=True,
            requires_numpy=True,
            *args,
            **kwargs,
        )
        assert fs in {8000, 16000}
        assert mode in {"wb", "nb"}

        self.fs = fs
        self.mode = mode

    def _get_values(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(
            [
                pesq_(fs=self.fs, ref=ref, deg=est, mode=self.mode)
                for ref, est in zip(
                    target.reshape((-1, target.shape[-1])),
                    preds.reshape((-1, preds.shape[-1])),
                )
            ],
            device=self.device,
        ).mean()


class PESQWB(BasePESQ):
    def __init__(self, *args, **kwargs):
        super().__init__(mode="wb", *args, **kwargs)


class PESQNB(BasePESQ):
    def __init__(self, *args, **kwargs):
        super().__init__(mode="nb", *args, **kwargs)


class PESQNBRAW(BasePESQ):
    def __init__(self, *args, **kwargs):
        super().__init__(mode="nb", *args, **kwargs)

    def _get_values(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(
            [
                pesq_nb_raw(fs=self.fs, ref=ref, deg=est)
                for ref, est in zip(
                    target.reshape((-1, target.shape[-1])),
                    preds.reshape((-1, preds.shape[-1])),
                )
            ],
            device=self.device,
        ).mean()


class BaseSTOI(BaseMetric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = True

    def __init__(
        self,
        *args,
        extended: bool = False,
        fs: int = 16000,
        **kwargs,
    ):
        super().__init__(
            requires_reference=True,
            requires_numpy=True,
            *args,
            **kwargs,
        )
        self.extended = extended
        self.fs = fs

    def _get_values(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(
            [
                stoi_(x=ref, y=est, fs_sig=self.fs, extended=self.extended)
                for ref, est in zip(
                    target.reshape((-1, target.shape[-1])),
                    preds.reshape((-1, preds.shape[-1])),
                )
            ],
            device=self.device,
        ).mean()


class STOI(BaseSTOI):
    def __init__(self, *args, **kwargs):
        super().__init__(name="STOI", extended=False, *args, **kwargs)


class ESTOI(BaseSTOI):
    def __init__(self, *args, **kwargs):
        super().__init__(name="ESTOI", extended=True, *args, **kwargs)


class SISDR(BaseMetric):
    is_differentiable = True
    higher_is_better = True
    full_state_update = True

    def __init__(self, *args, **kwargs):
        super().__init__(
            name="SISDR",
            requires_reference=True,
            requires_numpy=False,
            *args,
            **kwargs,
        )

    def _get_values(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_power = target.pow(2).sum(-1) + 1e-8
        scale = (target * preds).sum(-1) / target_power

        target_scaled = scale * target
        residual = preds - target_scaled

        true_power = target_scaled.pow(2).sum(-1)
        res_power = residual.pow(2).sum(-1)

        return (
            (10 * true_power.log10() - 10 * res_power.log10())
            .mean()
            .to(device=self.device)
        )

