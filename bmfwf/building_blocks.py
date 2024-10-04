from typing import Tuple
import numpy as np
import torch
from . import utils
from torch import nn


class cLN(nn.Module):
    def __init__(self, dimension, eps=1e-8, trainable=True):
        super().__init__()

        self.eps = eps
        self.gain = nn.Parameter(torch.ones((1, dimension, 1), requires_grad=trainable))
        self.bias = nn.Parameter(
            torch.zeros((1, dimension, 1), requires_grad=trainable)
        )

    def forward(self, input):
        # input size: (Batch, Freq, Time)
        # cumulative mean for each time step
        channel = input.size(1)
        time_step = input.size(2)

        step_sum = input.sum(1)  # B, T
        step_pow_sum = input.pow(2).sum(1)  # B, T
        cum_sum = torch.cumsum(step_sum, dim=1)  # B, T
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=1)  # B, T

        entry_cnt = torch.arange(
            channel, channel * (time_step + 1), channel, device=input.device
        )
        entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)

        cum_mean = cum_sum / entry_cnt  # B, T
        cum_var = (cum_pow_sum - 2 * cum_mean * cum_sum) / entry_cnt + cum_mean.pow(
            2
        )  # B, T
        cum_std = (cum_var + self.eps).sqrt()  # B, T

        cum_mean = cum_mean.unsqueeze(1)
        cum_std = cum_std.unsqueeze(1)

        x = (input - cum_mean.expand_as(input)) / cum_std.expand_as(input)
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(
            x.type()
        )


class RealTimecLN(cLN):
    def __init__(self, dimension, eps=1e-8, trainable=True):
        super().__init__(dimension, eps, trainable)

    def forward(
        self,
        inp: torch.Tensor,
        prev_state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ):
        # input size: (Batch, Freq, Time)
        cum_sum, cum_pow_sum, entry_cnt = prev_state

        # Update buffers
        cum_sum += inp.sum(1)  # B, T
        cum_pow_sum += inp.pow(2).sum(1)  # B, T
        entry_cnt += inp.size(1)  # B, T

        cum_mean = cum_sum / entry_cnt  # B, T
        cum_var = (cum_pow_sum - 2 * cum_mean * cum_sum) / entry_cnt + cum_mean.pow(
            2
        )  # B, T
        cum_std = (cum_var + self.eps).sqrt()  # B, T

        cum_mean = cum_mean.unsqueeze(1)
        cum_std = cum_std.unsqueeze(1)

        x = (inp - cum_mean.expand_as(inp)) / cum_std.expand_as(inp)  # B, F, T
        output = x * self.gain.expand_as(x) + self.bias.expand_as(x)

        new_state = (cum_sum, cum_pow_sum, entry_cnt)

        return output, new_state


class CLN3D(nn.Module):
    def __init__(self, dimension, eps=1e-8, trainable=True):
        super().__init__()

        self.eps = eps
        self.gain = nn.Parameter(
            torch.ones((1, dimension, 1, 1), requires_grad=trainable)
        )
        self.bias = nn.Parameter(
            torch.zeros((1, dimension, 1, 1), requires_grad=trainable)
        )

    def forward(self, input):
        channel = input.size(1)
        time_step = input.size(2)
        feature = input.size(3)

        step_sum = input.sum((1, 3))  # B, T
        step_pow_sum = input.pow(2).sum((1, 3))  # B, T
        cum_sum = torch.cumsum(step_sum, dim=1)  # B, T
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=1)  # B, T

        entry_cnt = torch.arange(
            channel * feature,
            channel * feature * (time_step + 1),
            channel * feature,
            device=input.device,
        )
        entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)

        cum_mean = cum_sum / entry_cnt  # B, T
        cum_var = (cum_pow_sum - 2 * cum_mean * cum_sum) / entry_cnt + cum_mean.pow(
            2
        )  # B, T
        cum_std = (cum_var + self.eps).sqrt()  # B, T

        cum_mean = cum_mean.unsqueeze(1).unsqueeze(-1)
        cum_std = cum_std.unsqueeze(1).unsqueeze(-1)

        x = (input - cum_mean.expand_as(input)) / cum_std.expand_as(input)
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(
            x.type()
        )


class RealTimeCLN3D(CLN3D):
    def __init__(self, dimension, eps=1e-8, trainable=True):
        super().__init__(dimension, eps, trainable)

    def forward(
        self,
        inp: torch.Tensor,
        prev_state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ):
        channel = inp.size(1)
        feature = inp.size(3)

        cum_sum, cum_pow_sum, entry_cnt = prev_state

        # Update buffers
        cum_sum += inp.sum((1, 3))  # B, T
        cum_pow_sum += inp.pow(2).sum((1, 3))  # B, T
        entry_cnt += channel * feature

        cum_mean = cum_sum / entry_cnt  # B, T
        cum_var = (cum_pow_sum - 2 * cum_mean * cum_sum) / entry_cnt + cum_mean.pow(
            2
        )  # B, T
        cum_std = (cum_var + self.eps).sqrt()  # B, T

        cum_mean = cum_mean.unsqueeze(1).unsqueeze(-1)
        cum_std = cum_std.unsqueeze(1).unsqueeze(-1)

        x = (inp - cum_mean.expand_as(inp)) / cum_std.expand_as(inp)
        output = x * self.gain.expand_as(x) + self.bias.expand_as(x)

        new_state = (cum_sum, cum_pow_sum, entry_cnt)

        return output, new_state


class Full_Band_Block(nn.Module):
    def __init__(self, D, E, I, J, Q, H, F, nonlin: str, norm_type: str):  # noqa: E741
        super(Full_Band_Block, self).__init__()
        self.pad = nn.ZeroPad2d((0, Q - F, 0, 0))
        self.conv2d = nn.Conv2d(D, E, (1, I), stride=(1, J))
        self.norm_type = norm_type
        self.A = int(E * ((Q - I) / J + 1))
        if self.norm_type == "cGLN":
            self.cGLN1 = cLN(dimension=self.A)
            self.cGLN2 = cLN(dimension=self.A)
        elif self.norm_type == "layer_norm":
            self.cGLN1 = nn.LayerNorm(self.A)
            self.cGLN2 = nn.LayerNorm(self.A)
        elif self.norm_type == "identity":
            self.cGLN1 = utils.identity
            self.cGLN2 = utils.identity
        else:
            raise ValueError("Unsupported normalization type: " + self.norm_type)

        self.prelu1 = getattr(nn, nonlin)()
        rnn = nn.GRU
        self.rnn = rnn(self.A, H, batch_first=True)
        self.linear = nn.Linear(H, self.A)
        self.prelu2 = getattr(nn, nonlin)()
        self.deconv2d = nn.ConvTranspose2d(E, D, (1, I), stride=(1, J))

    def forward(self, x):
        x_residual = x.clone()  # (B, D, T, F)
        x = self.pad(x)  # (B, D, T, Q)
        x = self.conv2d(x)  # (B, E, T, (Q-I)/J + 1)
        x = x.permute(0, 2, 1, 3)  # (B, T, E, (Q-I)/J + 1)
        x_shape = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1)  # (B, T, E*((Q-I)/J + 1))
        x = self.prelu1(x)  # (B, T, E*((Q-I)/J + 1))
        if self.norm_type in ["cGLN", "cGLNEMA"]:
            x = self.cGLN1(x.transpose(-2, -1)).transpose(
                -2, -1
            )  # (B, T, E*((Q-I)/J + 1))
        elif self.norm_type == "layer_norm":
            x = self.cGLN1(x)  # (B, T, E*((Q-I)/J + 1))
        elif self.norm_type == "identity":
            pass
        x, _ = self.rnn(x)  # (B, T, H)
        x = self.linear(x)  # (B, T, E*((Q-I)/J + 1))
        if self.norm_type == "cGLN":
            x = self.cGLN2(x.transpose(-2, -1)).transpose(
                -2, -1
            )  # (B, T, E*((Q-I)/J + 1))
        elif self.norm_type == "layer_norm":
            x = self.cGLN2(x)  # (B, T, E*((Q-I)/J + 1))
        elif self.norm_type == "identity":
            pass
        x = self.prelu2(x)  # (B, T, E*((Q-I)/J + 1))
        x = x.reshape(x.shape[0], x.shape[1], *x_shape[2:]).permute(
            0, 2, 1, 3
        )  # (B, E, T, ((Q-I)/J + 1))
        x = self.deconv2d(x)  # (B, D, T, Q)
        x = x[:, :, :, : x_residual.shape[3]]  # (B, D, T, F)
        out = x_residual + x  # (B, D, T, F)
        return out


class Sub_Band_Block(nn.Module):
    def __init__(
        self,
        D,
        E_prime,
        I_prime,
        J_prime,
        Q_prime,
        H_prime,
        F,
        nonlin: str,
        norm_type: str,
    ):
        super(Sub_Band_Block, self).__init__()

        self.pad = nn.ZeroPad2d((0, Q_prime - F, 0, 0))
        self.conv2d = nn.Conv2d(D, E_prime, (1, I_prime), stride=(1, J_prime))
        self.norm_type = norm_type
        if self.norm_type == "cGLN":
            self.cGLN = CLN3D(dimension=E_prime)
        elif self.norm_type == "layer_norm":
            self.cGLN = nn.LayerNorm((E_prime, int((Q_prime - I_prime) / J_prime + 1)))
        elif self.norm_type == "identity":
            self.cGLN = utils.identity
        else:
            raise ValueError("Unsupported normalization type: " + self.norm_type)
        self.prelu = getattr(nn, nonlin)()
        rnn_cls = nn.GRU
        self.rnn = rnn_cls(E_prime, H_prime, batch_first=True)

        self.upsample = utils.identity
        self.conv2d_after_upsample = utils.identity
        self.deconv2d = nn.ConvTranspose2d(
            H_prime, D, (1, I_prime), stride=(1, J_prime)
        )

    def forward(self, x):
        x_residual = x.clone()
        x = self.pad(x)  # (B, D, T, Q')
        x = self.conv2d(x)  # (B, E', T, (Q'-I')/J' + 1)
        x = self.prelu(x)  # (B, E', T, (Q'-I')/J' + 1)
        if self.norm_type == "cGLN":
            x = self.cGLN(x)  # (B, E', T, (Q'-I')/J' + 1)
            x = x.permute(0, 3, 2, 1)  # (B, (Q'-I')/J' + 1, T, E')
        elif self.norm_type == "identity":
            x = x.permute(0, 3, 2, 1)  # (B, (Q'-I')/J' + 1, T, E')
        elif self.norm_type == "layer_norm":
            x = self.cGLN(x.permute(0, 2, 1, 3)).permute(
                0, 3, 1, 2
            )  # (B, (Q'-I')/J' + 1, T, E')
        x_shape = x.shape
        # reshape such that each feature / "frequency-like" is seen as a separate sequence
        x = x.reshape(-1, x.shape[2], x.shape[3])  # (B*(Q'-I')/J' + 1, T, E')
        x, _ = self.rnn(x)  # (B*(Q'-I')/J' + 1, T, H')
        x = x.reshape(x_shape[:-1] + (-1,))  # (B, (Q'-I')/J' + 1, T, H')
        x = x.permute(0, 3, 2, 1)  # (B, H', T, (Q'-I')/J' + 1)
        x = self.deconv2d(x)  # (B, D, T, Q')
        x = x[:, :, :, : x_residual.shape[3]]  # (B, D, T, F)
        return x_residual + x  # (B, D, T, F)


class FSB_LSTMEstimator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_frequencies: int,
        D: int = 32,
        E: int = 8,
        I: int = 8,  # noqa: E741
        J: int = 4,
        Q: int = None,
        H: int = 256,
        E_prime: int = 64,
        I_prime: int = 5,
        J_prime: int = 5,
        Q_prime: int = None,
        H_prime: int = 64,
        B: int = 3,
        separate_encoders: bool = False,
        nonlin: str = "PReLU",
        normalization_type: str = "cGLN",
        use_first_norm: bool = True,
        use_bias: bool = True,
    ):
        """
        simple architecture combining fullband blocks (FBB) and subband blocks (SBB)

        Args:
            learning_rate (float, optional). Defaults to 1e-3.
            batch_size (int, optional). Defaults to 4.
            loss (str, optional). Defaults to "MagnitudeAbsoluteError".
            metrics_test (Union[ tuple, str ], optional). Defaults to "PESQWB,PESQNB,PESQNBRAW,STOI,ESTOI,SISDR".
            metrics_val (Union[tuple, str], optional). Defaults to "".
            frame_length (int, optional). Defaults to 256.
            shift_length (int, optional). Defaults to 128.
            num_channels (int, optional). Defaults to 1.
            binaural (bool, optional). Defaults to True.
            fs (int, optional). Defaults to 16000.
            D (int, optional): embedding dimension per TF unit. Defaults to 32.
            E (int, optional): output chanels Conv2D, FBB. Defaults to 8.
            I (int, optional): kernel size along frequency in Conv2D and Deconv2D, FBB. Defaults to 8.
            J (int, optional): stride size along frequency in Conv2D and Deconv2D, FBB. Defaults to 4.
            Q (int, optional): frequencies after padding, FBB. Defaults to None.
            H (int, optional): LSTM hidden units, FBB. Defaults to 256.
            E_prime (int, optional): output chanels Conv2D, SBB. Defaults to 64.
            I_prime (int, optional): kernel size along frequency in Conv2D and Deconv2D, SBB. Defaults to 5.
            J_prime (int, optional): stride size along frequency in Conv2D and Deconv2D, SBB. Defaults to 5.
            Q_prime (int, optional): frequencies after padding, SBB. Defaults to None.
            H_prime (int, optional): LSTM hidden units, FBB. Defaults to 64.
            B (int, optional): number of blocks. Defaults to 3.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.D = D
        self.E = E
        self.I = I
        self.J = J
        self.Q = Q
        self.H = H
        self.E_prime = E_prime
        self.I_prime = I_prime
        self.J_prime = J_prime
        self.Q_prime = Q_prime
        self.H_prime = H_prime
        self.B = B
        self.separate_encoders = separate_encoders
        self.nonlin = nonlin
        self.normalization_type = normalization_type
        self.use_first_norm = use_first_norm
        self.use_bias = use_bias

        self.F = num_frequencies
        self.P = input_dim

        if self.Q is None:
            self.Q = int(np.ceil((self.F - I) / J) * J + I)
        if self.Q_prime is None:
            self.Q_prime = int(
                np.ceil((self.F - I_prime) / J_prime) * J_prime + I_prime
            )

        self.conv2d = nn.ModuleList(
            [
                nn.Conv2d(self.P, D, (1, 3), padding="same", bias=use_bias)
                for _ in range((1 + self.separate_encoders))
            ]
        )

        self.full_band_blocks = nn.ModuleList(
            [
                Full_Band_Block(
                    D,
                    E,
                    I,
                    J,
                    self.Q,
                    H,
                    self.F,
                    nonlin=nonlin,
                    norm_type=normalization_type,
                )
                for _ in range(B)
            ]
        )
        self.sub_band_blocks = nn.ModuleList(
            [
                Sub_Band_Block(
                    D,
                    E_prime,
                    I_prime,
                    J_prime,
                    self.Q_prime,
                    H_prime,
                    self.F,
                    nonlin=nonlin,
                    norm_type=normalization_type,
                )
                for _ in range(B)
            ]
        )
        self.deconv2d = nn.ConvTranspose2d(
            D,
            output_dim,
            (1, 3),
            bias=use_bias,
        )
        if self.use_first_norm:
            if self.normalization_type == "cGLN":
                self.cGLN = CLN3D(dimension=D)
            else:
                raise ValueError(
                    "Unsupported normalization type: " + self.normalization_type
                )

    def forward(self, x):
        # shape of x: (2 - binaural) * B, input_dim, F, T
        x_stft = x.transpose(-1, -2)  # (2 - binaural) * B, input_dim, T, F
        if self.separate_encoders:
            batch_size = x_stft.shape[0] // 2
            x_stft_left = self.conv2d[0](x_stft[:batch_size])  # (B, D, T, F)
            x_stft_right = self.conv2d[1](x_stft[batch_size:])  # (B, D, T, F)
            x_stft = torch.cat([x_stft_left, x_stft_right], dim=0)
        else:
            x_stft = self.conv2d[0](x_stft)  # (B, D, T, F)
        if self.use_first_norm:
            if self.normalization_type == "cGLN":
                x_stft = self.cGLN(x_stft)  # (B, D, T, F)
            else:
                raise NotImplementedError

        for full_band_block, sub_band_block in zip(
            self.full_band_blocks, self.sub_band_blocks
        ):
            x_stft = full_band_block(x_stft)  # (B, D, T, F)
            x_stft = sub_band_block(x_stft)  # (B, D, T, F)
        x_stft = self.deconv2d(x_stft)[..., 1:-1]  # (B, output_dim, T, F)
        return x_stft.transpose(-2, -1)


class RealTimeFullBandBlock(Full_Band_Block):
    def __init__(self, D, E, I, J, Q, H, F, nonlin: str, norm_type: str):  # noqa: E741
        super().__init__(D, E, I, J, Q, H, F, nonlin=nonlin, norm_type=norm_type)
        A = int(E * ((Q - I) / J + 1))

        if self.norm_type == "cGLN":
            self.cGLN1 = RealTimecLN(dimension=A)
            self.cGLN2 = RealTimecLN(dimension=A)
        else:
            raise NotImplementedError

    def forward(
        self,
        x,
        full_band_hidden_state,
        full_band_cum_sum_1,
        full_band_cum_pow_sum_1,
        full_band_entry_cnt_1,
        full_band_cum_sum_2,
        full_band_cum_pow_sum_2,
        full_band_entry_cnt_2,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        cLN_states1 = (
            full_band_cum_sum_1,
            full_band_cum_pow_sum_1,
            full_band_entry_cnt_1,
        )
        cLN_states2 = (
            full_band_cum_sum_2,
            full_band_cum_pow_sum_2,
            full_band_entry_cnt_2,
        )
        x_residual = x.clone()  # (B, D, T, F)
        x = self.pad(x)  # (B, D, T, Q)
        x = self.conv2d(x)  # (B, E, T, (Q-I)/J + 1)
        x = x.permute(0, 2, 1, 3)  # (B, T, E, (Q-I)/J + 1)
        x_shape = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1)  # (B, T, E*((Q-I)/J + 1))
        x = self.prelu1(x)  # (B, T, E*((Q-I)/J + 1))
        if self.norm_type == "cGLN":
            x, cLN_states1 = self.cGLN1(
                x.transpose(-2, -1), cLN_states1
            )  # (B, E*((Q-I)/J + 1), T)
            x = x.transpose(-2, -1)  # (B, T, E*((Q-I)/J + 1))
        x, full_band_hidden_state = self.rnn(x, full_band_hidden_state)  # (B, T, H)
        x = self.linear(x)  # (B, T, E*((Q-I)/J + 1))
        if self.norm_type == "cGLN":
            x, cLN_states2 = self.cGLN2(
                x.transpose(-2, -1), cLN_states2
            )  # (B, E*((Q-I)/J + 1), T)
            x = x.transpose(-2, -1)  # (B, T, E*((Q-I)/J + 1))
        x = self.prelu2(x)  # (B, T, E*((Q-I)/J + 1))
        x = x.reshape(x.shape[0], x.shape[1], x_shape[2], x_shape[3]).permute(
            0, 2, 1, 3
        )  # (B, E, T, ((Q-I)/J + 1))
        x = self.deconv2d(x)  # (B, D, T, Q)
        x = x[:, :, :, : x_residual.shape[3]]  # (B, D, T, F)
        (
            full_band_cum_sum_1,
            full_band_cum_pow_sum_1,
            full_band_entry_cnt_1,
        ) = cLN_states1
        (
            full_band_cum_sum_2,
            full_band_cum_pow_sum_2,
            full_band_entry_cnt_2,
        ) = cLN_states2
        return (
            x_residual + x,  # (B, D, T, F)
            full_band_hidden_state,
            full_band_cum_sum_1,
            full_band_cum_pow_sum_1,
            full_band_entry_cnt_1,
            full_band_cum_sum_2,
            full_band_cum_pow_sum_2,
            full_band_entry_cnt_2,
        )


class RealTimeSubBandBlock(Sub_Band_Block):
    def __init__(
        self,
        D,
        E_prime,
        I_prime,
        J_prime,
        Q_prime,
        H_prime,
        F,
        nonlin: str,
        norm_type: str,
    ):
        super().__init__(
            D,
            E_prime,
            I_prime,
            J_prime,
            Q_prime,
            H_prime,
            F,
            nonlin=nonlin,
            norm_type=norm_type,
        )
        if self.norm_type == "cGLN":
            self.cGLN = RealTimeCLN3D(dimension=E_prime)
        else:
            raise NotImplementedError

    def forward(
        self,
        x,
        sub_band_hidden_state,
        sub_band_cum_sum,
        sub_band_cum_pow_sum,
        sub_band_entry_cnt,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        cLN_states = (sub_band_cum_sum, sub_band_cum_pow_sum, sub_band_entry_cnt)
        x_residual = x.clone()
        x = self.pad(x)  # (B, D, T, Q')
        x = self.conv2d(x)  # (B, E', T, (Q'-I')/J' + 1)
        x = self.prelu(x)  # (B, E', T, (Q'-I')/J' + 1)
        if self.norm_type == "cGLN":
            x, cLN_states = self.cGLN(x, cLN_states)  # (B, E', T, (Q'-I')/J' + 1)
            x = x.permute(0, 3, 2, 1)  # (B, (Q'-I')/J' + 1, T, E')
        x_shape = x.shape
        # reshape such that each feature / "frequency-like" is seen as a separate sequence
        x = x.reshape(-1, x.shape[2], x.shape[3])  # (B*((Q'-I')/J' + 1), T, E')
        x, sub_band_hidden_state = self.rnn(
            x, sub_band_hidden_state
        )  # (B*((Q'-I')/J' + 1), T, H')
        x = x.reshape(x_shape[:-1] + (-1,)).permute(
            0, 3, 2, 1
        )  # (B, E', T, (Q'-I')/J' + 1)
        x = self.deconv2d(x)  # (B, D, T, Q')
        x = x[:, :, :, : x_residual.shape[3]]  # (B, D, T, F)
        (sub_band_cum_sum, sub_band_cum_pow_sum, sub_band_entry_cnt) = cLN_states
        return (
            x_residual + x,  # (B, D, T, F)
            sub_band_hidden_state,
            sub_band_cum_sum,
            sub_band_cum_pow_sum,
            sub_band_entry_cnt,
        )


class RealTimeFSB_LSTMEstimator(FSB_LSTMEstimator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.full_band_blocks = nn.ModuleList(
            [
                RealTimeFullBandBlock(
                    self.D,
                    self.E,
                    self.I,
                    self.J,
                    self.Q,
                    self.H,
                    self.F,
                    nonlin=self.nonlin,
                    norm_type=self.normalization_type,
                )
                for _ in range(self.B)
            ]
        )
        self.sub_band_blocks = nn.ModuleList(
            [
                RealTimeSubBandBlock(
                    self.D,
                    self.E_prime,
                    self.I_prime,
                    self.J_prime,
                    self.Q_prime,
                    self.H_prime,
                    self.F,
                    nonlin=self.nonlin,
                    norm_type=self.normalization_type,
                )
                for _ in range(self.B)
            ]
        )
        if self.normalization_type == "cGLN":
            self.cGLN = RealTimeCLN3D(dimension=self.D)
        else:
            raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        full_band_hidden_state: torch.Tensor,
        full_band_cum_sum_1: torch.Tensor,
        full_band_cum_pow_sum_1: torch.Tensor,
        full_band_entry_cnt_1: torch.Tensor,
        full_band_cum_sum_2: torch.Tensor,
        full_band_cum_pow_sum_2: torch.Tensor,
        full_band_entry_cnt_2: torch.Tensor,
        sub_band_hidden_state: torch.Tensor,
        sub_band_cum_sum: torch.Tensor,
        sub_band_cum_pow_sum: torch.Tensor,
        sub_band_entry_cnt: torch.Tensor,
        cum_sum: torch.Tensor,
        cum_pow_sum: torch.Tensor,
        entry_cnt: torch.Tensor,
    ):
        # shape of x: (2 - binaural) * B, input_dim, F, 1
        x_stft = x.transpose(-1, -2)  # (2 - binaural) * B, input_dim, T, F
        x_stft = self.conv2d[0](x_stft)  # (B, D, T, F)

        cLN_states = (cum_sum, cum_pow_sum, entry_cnt)
        x_stft, cLN_states = self.cGLN(x_stft, cLN_states)  # (B, D, T, F)

        for idx, (full_band_block, sub_band_block) in enumerate(
            zip(self.full_band_blocks, self.sub_band_blocks)
        ):
            (
                x_stft,
                full_band_hidden_state[idx],
                full_band_cum_sum_1[idx],
                full_band_cum_pow_sum_1[idx],
                full_band_entry_cnt_1[idx],
                full_band_cum_sum_2[idx],
                full_band_cum_pow_sum_2[idx],
                full_band_entry_cnt_2[idx],
            ) = full_band_block(
                x_stft,
                full_band_hidden_state[idx],
                full_band_cum_sum_1[idx],
                full_band_cum_pow_sum_1[idx],
                full_band_entry_cnt_1[idx],
                full_band_cum_sum_2[idx],
                full_band_cum_pow_sum_2[idx],
                full_band_entry_cnt_2[idx],
            )  # (B, D, T, F)
            (
                x_stft,
                sub_band_hidden_state[idx],
                sub_band_cum_sum[idx],
                sub_band_cum_pow_sum[idx],
                sub_band_entry_cnt[idx],
            ) = sub_band_block(
                x_stft,
                sub_band_hidden_state[idx],
                sub_band_cum_sum[idx],
                sub_band_cum_pow_sum[idx],
                sub_band_entry_cnt[idx],
            )  # (B, D, T, F)

        x_stft = self.deconv2d(x_stft)[..., 1:-1]  # (B, output_dim, T, F)
        (cum_sum, cum_pow_sum, entry_cnt) = cLN_states
        return (
            x_stft.transpose(-2, -1),
            full_band_hidden_state,
            full_band_cum_sum_1,
            full_band_cum_pow_sum_1,
            full_band_entry_cnt_1,
            full_band_cum_sum_2,
            full_band_cum_pow_sum_2,
            full_band_entry_cnt_2,
            sub_band_hidden_state,
            sub_band_cum_sum,
            sub_band_cum_pow_sum,
            sub_band_entry_cnt,
            cum_sum,
            cum_pow_sum,
            entry_cnt,
        )
