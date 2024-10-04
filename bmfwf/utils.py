import datetime
import math
import os
from dataclasses import dataclass
from typing import Optional, Union

import human_readable_ids as hri
import numpy as np
import torch
import torchaudio as ta
from torch.utils.data._utils.collate import default_collate

ta.set_audio_backend("soundfile")

EPS = torch.as_tensor(torch.finfo(torch.get_default_dtype()).eps)
PI = math.pi


class STFTTorch:
    """
    class used to simplify handling of STFT & iSTFT
    """

    def __init__(
        self,
        frame_length=64,
        overlap_length=48,
        window=torch.hann_window,
        sqrt=True,
        normalized: bool = False,
        center: bool = True,
        fft_length=None,
        fft_length_synth=None,
        synthesis_window=None,
    ):
        self.frame_length = frame_length
        if fft_length is None:
            self.fft_length = frame_length
        else:
            self.fft_length = fft_length

        if fft_length_synth is None:
            self.fft_length_synth = fft_length
        else:
            self.fft_length_synth = fft_length_synth

        self.num_bins = int((self.fft_length / 2) + 1)
        self.overlap_length = overlap_length
        self.shift_length = self.frame_length - self.overlap_length
        self.sqrt = sqrt
        self.normalized = normalized
        self.center = center

        if type(window) is str:
            if window == "hann":
                window = torch.hann_window
            elif window == "hamming":
                window = torch.hamming_window
            elif window == "bartlett":
                window = torch.bartlett_window
            elif window == "blackman":
                window = torch.blackman_window
            else:
                raise ValueError("unknown window type!")
            self.window = window(
                self.frame_length,
                periodic=True,
                dtype=torch.get_default_dtype(),
            )
        elif callable(window):
            self.window = window(
                self.frame_length,
                periodic=True,
                dtype=torch.get_default_dtype(),
            )
        elif type(window) is torch.Tensor:
            self.window = window
        else:
            raise NotImplementedError()

        if self.sqrt:
            self.window = self.window.sqrt()

        if synthesis_window is None:
            self.synthesis_window = self.window
        else:
            self.synthesis_window = synthesis_window

    def get_stft(self, wave):
        if self.window.device != wave.device:
            # move to device
            self.window = self.window.to(device=wave.device)
        shape_orig = wave.shape
        if wave.ndim > 2:  # reshape required
            wave = wave.reshape(-1, shape_orig[-1])
        stft = torch.stft(
            wave,
            window=self.window,
            n_fft=self.fft_length,
            hop_length=self.shift_length,
            win_length=self.frame_length,
            normalized=self.normalized,
            center=self.center,
            pad_mode="constant",
            return_complex=True,
        )
        return stft.reshape((*shape_orig[:-1], *stft.shape[-2:]))

    def get_istft(self, stft, length=None):
        if self.synthesis_window.device != stft.device:
            # move to device
            self.synthesis_window = self.synthesis_window.to(stft.device)

        if stft.ndim == 3:  # batch x F x T
            istft = torch.istft(
                stft,
                window=self.synthesis_window,
                n_fft=self.fft_length_synth,
                hop_length=self.shift_length,
                win_length=self.frame_length,
                normalized=self.normalized,
                center=self.center,
                length=length,
                return_complex=False,
            )
        elif stft.ndim == 4:  # batch x M x F x T
            istft = torch.stack(
                [
                    torch.istft(
                        x,
                        window=self.synthesis_window,
                        n_fft=self.fft_length,
                        hop_length=self.shift_length,
                        win_length=self.frame_length,
                        normalized=self.normalized,
                        center=self.center,
                        length=length,
                        return_complex=False,
                    )
                    for x in stft
                ]
            )
        else:
            raise ValueError("unsupported STFT shape!")
        return istft


class STFTTorchScript(torch.nn.Module):
    def __init__(
        self,
        window: torch.Tensor,
        synthesis_window: Optional[torch.Tensor] = None,
        frame_length=64,
        overlap_length=32,
        sqrt=True,
        normalized: bool = False,
        center: bool = True,
        fft_length=None,
        fft_length_synth=None,
    ):
        super().__init__()

        self.frame_length = frame_length
        if fft_length is None:
            self.fft_length = frame_length
        else:
            self.fft_length = fft_length

        if fft_length_synth is None:
            self.fft_length_synth = fft_length
        else:
            self.fft_length_synth = fft_length_synth

        self.num_bins = int((self.fft_length / 2) + 1)
        self.overlap_length = overlap_length
        self.shift_length = self.frame_length - self.overlap_length
        self.sqrt = sqrt
        self.normalized = normalized
        self.center = center

        self.window = window

        if self.sqrt:
            self.window = self.window.sqrt()

        if synthesis_window is None:
            self.synthesis_window = self.window
        else:
            self.synthesis_window = synthesis_window

    @torch.jit.export
    def get_stft(self, wave):
        shape_orig = wave.shape
        if wave.ndim > 2:  # reshape required
            wave = wave.reshape(-1, shape_orig[-1])
        stft = torch.stft(
            wave,
            window=self.window,
            n_fft=self.fft_length,
            hop_length=self.shift_length,
            win_length=self.frame_length,
            normalized=self.normalized,
            center=self.center,
            pad_mode="constant",
            return_complex=True,
        )
        stft = stft.reshape(shape_orig[:-1] + stft.shape[-2:])
        return stft

    @torch.jit.export
    def get_istft(self, stft: torch.Tensor, length: Union[None, int] = None):
        if stft.ndim == 3:  # batch x F x T
            istft = torch.istft(
                stft,
                window=self.synthesis_window,
                n_fft=self.fft_length,
                hop_length=self.shift_length,
                win_length=self.frame_length,
                normalized=self.normalized,
                center=self.center,
                length=length,
                return_complex=False,
            )
        elif stft.ndim == 4:  # batch x M x F x T
            istft = torch.stack(
                [
                    torch.istft(
                        x,
                        window=self.synthesis_window,
                        n_fft=self.fft_length,
                        hop_length=self.shift_length,
                        win_length=self.frame_length,
                        normalized=self.normalized,
                        center=self.center,
                        length=length,
                        return_complex=False,
                    )
                    for x in stft
                ]
            )
        else:
            raise ValueError("unsupported STFT shape!")
        return istft


class CustomBatchBase:
    def __init__(self):
        pass

    def pin_memory(self):
        self.signals = {
            key: val.pin_memory() if val is torch.Tensor else val
            for key, val in self.signals.items()
        }
        return self

    def cuda(self, device=None, non_blocking=True):
        self.signals = {
            key: val.cuda(device=device, non_blocking=non_blocking)
            for key, val in self.signals.items()
        }
        return self

    def to(self, device=None, dtype=None, non_blocking=True):
        self.signals = {
            key: val.to(device=device, dtype=dtype, non_blocking=non_blocking)
            for key, val in self.signals.items()
        }
        return self


@dataclass
class CustomBatchSignalsMeta(CustomBatchBase):
    signals: dict
    meta: list

    def __init__(self, batch: list) -> None:
        super().__init__()

        self.signals = default_collate(
            [x[0] for x in batch],
        )
        self.meta = [x[1] for x in batch]


def collate_fn_signals_meta(batch):
    return CustomBatchSignalsMeta(batch)


@torch.jit.script
def trace(mat: torch.Tensor, keepdim: bool = False):
    """
    returns the trace of mat, taken over the last two dimensions
    :param mat:
    :return:
    """
    return torch.diagonal(mat, dim1=-2, dim2=-1).sum(-1, keepdim=keepdim)


def tik_reg(mat: torch.Tensor, reg: float = 1e-3, eps: float = EPS):
    """
    performs Tikhonov regularization
    mat: ... x M x M
    """
    M = mat.shape[-2]
    return (
        mat
        + ((reg * trace(mat.abs() + eps)) / M)[..., None, None]
        * torch.eye(M, device=mat.device)[None, None, ...]
    )


def rms(x):
    """
    compute rms, i.e., root-mean-square of vector x
    """
    return np.sqrt(np.mean(np.square(x)))


def time_to_smoothing_constant(time_constant, shift_length, fs=16000):
    """convert time constant to smoothing constant"""
    return np.exp(-shift_length / (fs * time_constant))


def save_wave(data, filename, fs=16000, normalize=False):
    """optionally normalize with rms and save data under filename using soundfile"""
    if data.ndim == 1:
        data = data.unsqueeze(0)
    if normalize:
        denominator = (data**2).mean().sqrt()
        data = data * (0.1 / denominator)
    ta.save(filepath=filename, src=data.cpu(), sample_rate=fs, bits_per_sample=32)


def get_measure_enhanced_noisy(output, signals, measure, **kwargs):
    """
    compute difference of measure on enhanced and noisy
    """
    try:
        result_enhanced = measure(output, signals, noisy=False, **kwargs)
        result_noisy = measure(output, signals, noisy=True, **kwargs)
    except KeyError:
        result_enhanced = np.nan
        result_noisy = np.nan

    return result_enhanced, result_noisy


def normalize(audio, target_level=-25):
    """Normalize the signal to the target level"""
    rms = (audio**2).mean() ** 0.5
    scalar = 10 ** (target_level / 20) / (rms + EPS)
    audio = audio * scalar
    return audio


def dcn(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def get_save_dir():
    path_current_file = os.path.dirname(os.path.abspath(__file__))
    str_year_month_day = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(
        path_current_file,
        "saved",
        f"{str_year_month_day}_" + hri.get_new_id().lower().replace(" ", "-"),
    )
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def clone_tensors(tuples):
    """Clone a tuple, list, or dict of tensors.

    Args:
        tuples (tuple, list, dict, or torch.Tensor): The input to clone.

    Returns:
        tuple, list, dict, or torch.Tensor: A clone of the input.
    """
    if isinstance(tuples, torch.Tensor):
        return tuples.clone()
    elif isinstance(tuples, tuple):
        return tuple(clone_tensors(t) for t in tuples)
    elif isinstance(tuples, list):
        return [clone_tensors(t) for t in tuples]
    elif isinstance(tuples, dict):
        return {k: clone_tensors(v) for k, v in tuples.items()}
    else:
        return tuples


def identity(x):
    return x
