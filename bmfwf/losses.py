import torch
import torch.nn.functional as F
from . import utils

EPS = torch.as_tensor(torch.finfo(torch.get_default_dtype()).eps)


def nll_loss(output, target):
    return F.nll_loss(output, target)


class BaseSELoss(torch.nn.Module):
    def __init__(
        self,
        use_stft: bool = False,
        multichannel_handling: str = "average",
        **kwargs,
    ) -> None:
        super().__init__()
        self.use_stft = use_stft
        self.multichannel_handling = multichannel_handling
        self.kwargs = kwargs

        if self.use_stft:
            self.stft = utils.STFTTorch(
                frame_length=self.kwargs["frame_length"],
                overlap_length=self.kwargs["overlap_length"],
                window=self.kwargs["window_fn"],
                sqrt=self.kwargs["sqrt"],
            )

    def forward(self, outputs: dict, batch: dict, meta: dict = None):
        target = batch["target"]
        estimate = outputs["input_proc"]

        assert target.ndim <= 3
        multichannel = target.ndim == 3  # (B x M x T)

        if self.use_stft:
            if multichannel:
                target = torch.stack([self.stft.get_stft(x) for x in target], dim=0)
                estimate = torch.stack([self.stft.get_stft(x) for x in estimate], dim=0)
            else:
                target = self.stft.get_stft(target)
                estimate = self.stft.get_stft(estimate)

        if multichannel:
            if self.multichannel_handling == "cat":
                # concatenate channels temporally
                target = torch.cat(
                    [target[:, idx] for idx in torch.arange(target.shape[1])], dim=-1
                )
                estimate = torch.cat(
                    [estimate[:, idx] for idx in torch.arange(estimate.shape[1])],
                    dim=-1,
                )
            elif self.multichannel_handling == "average":  # changes come later...
                pass
            else:
                raise ValueError(
                    f"unknown multichannel handling type {self.multichannel_handling}!"
                )

        assert target.shape == estimate.shape

        return {"loss": self.get_loss(target, estimate)}

    def get_loss(self, target: torch.Tensor, estimate: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MagnitudeAbsoluteError(BaseSELoss):
    def __init__(
        self,
        frame_length=512,
        overlap_length=None,
        window_fn=torch.hann_window,
        sqrt=True,
        use_mask=False,
        beta: float = 0.4,
        kind: str = "combined",
        **kwargs,
    ):
        self.kind = kind
        self.beta = beta
        self.overlap_length = (
            int(0.5 * frame_length) if overlap_length is None else overlap_length
        )

        super().__init__(
            use_stft=True,
            frame_length=frame_length,
            overlap_length=self.overlap_length,
            use_mask=use_mask,
            beta=beta,
            kind=kind,
            window_fn=window_fn,
            sqrt=sqrt,
        )

    def get_loss(self, target: torch.Tensor, estimate: torch.Tensor) -> torch.Tensor:
        loss_magnitude = (estimate.abs() - target.abs()).abs().mean()
        loss_complex = (estimate - target).abs().mean()

        if self.kind == "combined":
            loss = self.beta * loss_complex + (1.0 - self.beta) * loss_magnitude
        elif self.kind == "complex":
            loss = loss_complex
        elif self.kind == "magnitude":
            loss = loss_magnitude
        else:
            raise ValueError(f"unknown loss kind {self.kind}!")
        return loss


class CompressedLoss(BaseSELoss):
    def __init__(
        self,
        frame_length=512,
        overlap_length=None,
        window_fn=torch.hann_window,
        sqrt=True,
        use_mask=False,
        beta: float = 0.3,
        kind: str = "combined",
        compression_exponent: float = 0.3,
        **kwargs,
    ):
        self.compression_exponent = compression_exponent
        self.kind = kind
        self.beta = beta
        self.overlap_length = (
            int(0.5 * frame_length) if overlap_length is None else overlap_length
        )

        super().__init__(
            use_stft=True,
            frame_length=frame_length,
            overlap_length=self.overlap_length,
            use_mask=use_mask,
            beta=beta,
            kind=kind,
            compression_exponent=compression_exponent,
            window_fn=window_fn,
            sqrt=sqrt,
            **kwargs,
        )

    def get_loss(self, target: torch.Tensor, estimate: torch.Tensor) -> torch.Tensor:
        loss_magnitude = (
            (
                estimate.abs().pow(self.compression_exponent)
                - target.abs().pow(self.compression_exponent)
            )
            .abs()
            .pow(2)
        )
        loss_complex = (
            (
                estimate.abs().pow(self.compression_exponent)
                * estimate
                / (estimate.abs() + EPS)
                - target.abs().pow(self.compression_exponent)
                * target
                / (target.abs() + EPS)
            )
            .abs()
            .pow(2)
        )

        loss_magnitude = loss_magnitude.mean()
        loss_complex = loss_complex.mean()

        if self.kind == "combined":
            loss = self.beta * loss_complex + (1.0 - self.beta) * loss_magnitude
        elif self.kind == "complex":
            loss = loss_complex
        elif self.kind == "magnitude":
            loss = loss_magnitude
        else:
            raise ValueError(f"unknown loss kind {self.kind}!")
        return loss
