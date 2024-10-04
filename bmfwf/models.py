import math
from typing import Union
from . import building_blocks as bb
from . import utils
import pyfar
import resampy as rs
import torch
import torch.nn.functional as F
from .base import BaseLitModel


EPS = torch.as_tensor(torch.finfo(torch.get_default_dtype()).eps)
PI = math.pi


class BMWF(BaseLitModel):
    """
    Binaural Deep Multi-Frame Wiener Filter (BMFWF) model.

    The DNN architecture is based on Wang et al., “Neural Speech Enhancement with Very Low
    Algorithmic Latency and Complexity via Integrated full- and sub-band Modeling”, in Proc. IEEE
    International Conference on Acoustics, Speech and Signal Processing (ICASSP) 2023. It was
    modified to decrease computational complexity.
    The MFWF implementation is similar to Wang et al., “TF-GridNet: Integrating Full- and Sub-
    Band Modeling for Speech Separation”, in IEEE/ACM Transactions on Audio, Speech, and
    Language Processing, vol. 31, 2023, Eq. (14). It was modified for online processing using
    recursive smoothing.

    Attributes:
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training.
        loss (str): Loss function to be used.
        metrics_test (Union[tuple, str]): Metrics for testing.
        metrics_val (Union[tuple, str]): Metrics for validation.
        my_optimizer (str): Optimizer to be used.
        my_lr_scheduler (str): Learning rate scheduler to be used.
        frame_length (int): Frame length for STFT.
        fft_length (int): FFT length for STFT.
        shift_length (int): Shift length for STFT.
        filter_length (int): Filter length for multi-frame processing.
        fs (int): Sampling frequency.
        num_channels (int): Number of channels.
        reg (float): Regularization parameter.
        time_constant_recursive_smoothing (float): Time constant for recursive smoothing.
        window_type (str): Type of window to be used for STFT.
        feature_representation (str): Type of feature representation ('real_imag', 'mag_phase', 'mag').
        auxiliary_input (str): Type of auxiliary input ('none', 'matched_filter').
        binaural (bool): Whether to use binaural processing.
        use_mwf (bool): Whether to use multi-frame Wiener filtering.
        D (int): DNN parameter.
        E (int): DNN parameter.
        I (int): DNN parameter.
        J (int): DNN parameter.
        Q (int): DNN parameter.
        H (int): DNN parameter.
        E_prime (int): DNN parameter.
        I_prime (int): DNN parameter.
        J_prime (int): DNN parameter.
        Q_prime (int): DNN parameter.
        H_prime (int): DNN parameter.
        B (int): DNN parameter.
        nonlin (str): Non-linearity to be used in DNN.
        use_log (bool): Whether to use logarithmic features.
        normalization_type (str): Type of normalization to be used.
        use_first_norm (bool): Whether to use first normalization.
        use_bias (bool): Whether to use bias in DNN layers.

    Methods:
        get_hrtf_front(self):
        Loads and processes the Head-Related Transfer Function (HRTF) for the front direction. Used to generate auxiliary input with a matched filter.

        get_estimators(self):
            Initializes the estimators for the model based on the feature representation.

        forward(self, x):
            Performs a forward pass through the model.

        get_binaural_multiframe_vector(self, singleframe):
            Computes the multi-frame vector for binaural processing.

        get_multiframe_vector(self, singleframe):
            Computes the multi-frame vector for single-channel processing.

        get_features(self, noisy, auxiliary_input):
            Extracts features from the noisy input and auxiliary input based on the feature representation.
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        batch_size: int = 16,
        loss: str = "MagnitudeAbsoluteError",
        metrics_test: Union[tuple, str] = "PESQWB,PESQNB,PESQNBRAW,STOI,ESTOI,SISDR",
        metrics_val: Union[tuple, str] = "",
        my_optimizer: str = "Adam",
        my_lr_scheduler: str = "ReduceLROnPlateau",
        frame_length: int = 64,
        fft_length: int = 128,
        shift_length: int = 32,
        filter_length: int = 4,
        fs: int = 16000,
        num_channels: int = 1,
        reg: float = 1e-3,
        time_constant_recursive_smoothing: float = 0.002,
        window_type: str = "hann",
        feature_representation: str = "real_imag",
        auxiliary_input: str = "none",  # is "matched filter" in original experiment, but requires SOFA HRTFs
        binaural: bool = True,  # binaural or bilateral
        use_mwf: bool = False,
        D: int = 32,
        E: int = 8,
        I: int = 8,  # noqa: E741
        J: int = 4,
        Q: int = None,
        H: int = 128,
        E_prime: int = 64,
        I_prime: int = 5,
        J_prime: int = 5,
        Q_prime: int = None,
        H_prime: int = 32,
        B: int = 2,
        nonlin: str = "Mish",
        use_log: bool = True,
        normalization_type: str = "cGLN",
        use_first_norm: bool = True,
        use_bias: bool = True,
        **kwargs,
    ):
        super().__init__(
            lr=learning_rate,
            batch_size=batch_size,
            loss=loss,
            metrics_test=metrics_test,
            metrics_val=metrics_val,
            model_name="BDMFMVDR",
            my_optimizer=my_optimizer,
            my_lr_scheduler=my_lr_scheduler,
        )
        self.frame_length = frame_length
        self.fft_length = fft_length
        self.shift_length = shift_length
        self.filter_length = filter_length
        self.fs = fs
        self.num_channels = num_channels
        self.reg = reg
        self.time_constant_recursive_smoothing = time_constant_recursive_smoothing
        self.window_type = window_type
        self.feature_representation = feature_representation
        self.loss = loss
        self.use_mwf = use_mwf
        self.auxiliary_input = auxiliary_input
        self.binaural = binaural
        self.use_log = use_log
        self.normalization_type = normalization_type
        self.use_first_norm = use_first_norm
        self.use_bias = use_bias

        self.dnn_params = {
            "D": D,
            "E": E,
            "I": I,
            "J": J,
            "Q": Q,
            "H": H,
            "E_prime": E_prime,
            "I_prime": I_prime,
            "J_prime": J_prime,
            "Q_prime": Q_prime,
            "H_prime": H_prime,
            "B": B,
            "nonlin": nonlin,
        }

        self.stft = utils.STFTTorch(
            frame_length=self.frame_length,
            overlap_length=self.frame_length - self.shift_length,
            window=torch.hann_window(self.frame_length, periodic=True),
            synthesis_window=torch.ones(self.frame_length),
            sqrt=False,
            fft_length=self.fft_length,
            fft_length_synth=self.fft_length,
        )
        self.frequency_bins = self.stft.num_bins
        self.smoothing_constant_recursive_smoothing = utils.time_to_smoothing_constant(
            self.time_constant_recursive_smoothing, self.shift_length
        )

        self.get_estimators()

        self.ref_indices = (
            self.filter_length - 1,
            self.filter_length * (self.num_channels + 1) - 1,
        )
        self.ref_channels = (0, self.num_channels)

        if self.auxiliary_input == "matched_filter":
            # load SOFA file to get HRTF for auxiliary input
            # the auxiliary input is the matched filter output, with the matched filter maximizing white noise
            # gain towards the front (where typically the target is positioned)
            self.hrtf_front = self.get_hrtf_front().to(self.device)
        else:
            self.hrtf_front = torch.eye(2 * self.filter_length * self.num_channels).to(
                self.device
            )

        self.num_params = self.count_parameters()
        self.save_hyperparameters()

    def get_hrtf_front(self):
        # the provided SOFA file is the one used in the original experiment
        # should match the actual HRTF!!!
        data_ir, source_coordinates, _ = pyfar.io.read_sofa(
            "Kemar_with_PHL_high_res.sofa"
        )
        index, *_ = source_coordinates.find_nearest_k(
            0,
            0,
            1.5,
            k=1,
            domain="sph",
            convention="top_elev",
            unit="deg",
            show=True,
        )
        hrir_front = data_ir[index, :, :]
        # resample to desired sampling rate and compute frequency response
        if hrir_front.sampling_rate != self.fs:
            hrir_front = rs.resample(
                hrir_front.time,
                sr_orig=hrir_front.sampling_rate,
                sr_new=self.fs,
                axis=1,
            )

        # choose / order channels
        # model expects: [l_front,l_back,r_front,r_back] or [l_front,r_front]
        self.channels = (0, 2, 1, 3)

        hrir_front = hrir_front[self.channels, :]
        hrir_front = torch.as_tensor(hrir_front, dtype=torch.get_default_dtype())[None]
        # compute RFFT, padding to FFT length
        hrtf_front = torch.fft.rfft(hrir_front, n=self.fft_length, dim=-1)  # 1 x 2M x F
        # compute relative transfer functions w.r.t. ref_channels
        hrtf_front_left = (
            hrtf_front / hrtf_front[:, self.ref_channels[:1]]
        )  # 1 x 2M x F
        hrtf_front_right = (
            hrtf_front / hrtf_front[:, self.ref_channels[1:2]]
        )  # 1 x 2M x F
        # normalize w.r.t. 2-norm
        hrtf_front_left = hrtf_front_left / (
            hrtf_front_left.abs().pow(2).sum(1, keepdim=True) + EPS
        )
        hrtf_front_right = hrtf_front_right / (
            hrtf_front_right.abs().pow(2).sum(1, keepdim=True) + EPS
        )
        hrtf_front = torch.stack(
            [hrtf_front_left, hrtf_front_right],
            dim=1,
        )  # 1 x 2 x 2M x F
        return hrtf_front.permute(0, 1, 3, 2)

    def get_estimators(self):
        if self.feature_representation == "mag_phase":
            self.input_size = (
                (1 + self.binaural) * self.num_channels
                + (2 * (self.auxiliary_input != "none"))
            ) * 3
        elif self.feature_representation == "real_imag":
            self.input_size = (
                (1 + self.binaural) * self.num_channels
                + (2 * (self.auxiliary_input != "none"))
            ) * 2
        elif self.feature_representation == "mag":
            self.input_size = (1 + self.binaural) * self.num_channels + (
                2 * (self.auxiliary_input != "none")
            )
        else:
            raise ValueError(
                f"unknown feature representation {self.feature_representation}!"
            )

        self.output_size = int(1 + self.binaural)

        self.estimator_mask_speech = bb.FSB_LSTMEstimator(
            input_dim=self.input_size,
            output_dim=self.output_size,
            num_frequencies=self.frequency_bins,
            **self.dnn_params,
            separate_encoders=not self.binaural,
            normalization_type=self.normalization_type,
            use_first_norm=self.use_first_norm,
            use_bias=self.use_bias,
        )

    def forward(
        self,
        x,
    ):
        noisy = x[
            "input"
        ]  # B x 2M x L; 2M: first all left channels, then all right channels
        num_samples = noisy.shape[-1]
        noisy = self.stft.get_stft(noisy)  # B x 2M x F x T

        if self.auxiliary_input == "matched_filter":
            auxiliary_input = (
                (
                    self.hrtf_front.unsqueeze(-1).unsqueeze(-3).to(noisy.device).mH
                    @ noisy.permute(0, 2, 3, 1).unsqueeze(-1).unsqueeze(1)
                )
                .squeeze(-1)
                .squeeze(-1)
            )  # B x 2 x F x T
        elif self.auxiliary_input == "none":
            auxiliary_input = None
        else:
            raise ValueError(f"unknown auxiliary input {self.auxiliary_input}!")

        if self.num_channels == 1 and noisy.shape[1] > 2:
            noisy = noisy[:, [0, 2]]

        noisy_valid_frequencies = noisy  # B x 2M x F x T
        noisy_multiframe = self.get_binaural_multiframe_vector(
            noisy_valid_frequencies
        )  # B x F x T x 2MN x 1

        if self.binaural:
            features = self.get_features(
                noisy_valid_frequencies, auxiliary_input
            )  # B x 2 * 2 * (1 + binaural) x F x T
        else:
            # concatenate features across batch dimension to obtain independent processing for left and right
            features_left = self.get_features(
                noisy_valid_frequencies[:, : self.num_channels], auxiliary_input
            )  # B x 2M x F x T
            features_right = self.get_features(
                noisy_valid_frequencies[:, self.num_channels :], auxiliary_input
            )  # B x 2M x F x T
            features = torch.cat(
                [features_left, features_right], dim=0
            )  # 2B x 2M x F x T

        batch_size, num_frames = noisy.shape[0], noisy.shape[-1]

        output_estimator = self.estimator_mask_speech(
            features
        )  # B * (2 - binaural) x 2 * (1 + binaural) x F x T

        output_estimator = torch.sigmoid(output_estimator)

        if self.binaural:
            output_estimator = torch.stack(
                [
                    output_estimator[:, :1],
                    output_estimator[:, 1:],
                ],
                dim=1,
            )  # B x 2 x 1 x F x T
        output_estimator = output_estimator.squeeze(2)

        if not self.binaural:
            # extract left and right channels from batch dimension
            output_estimator = torch.stack(
                [
                    output_estimator[:batch_size],
                    output_estimator[batch_size:],
                ],
                dim=1,
            )  # B x 2 x F x T

        # apply estimator output to reference channels to obtain target estimates
        target_estimate_preliminary = output_estimator * noisy

        try:
            training = self.trainer.training
        except RuntimeError:
            training = False
        if training or not self.use_mwf:
            return {
                "input_proc_stft": target_estimate_preliminary,
                "input_proc": 2
                * self.stft.get_istft(target_estimate_preliminary, length=num_samples),
            }

        noisy_outer_products = noisy_multiframe @ noisy_multiframe.mH
        covariance_matrix_noisy = noisy_outer_products.new_zeros(
            size=noisy_outer_products.shape
        )
        cross_covariance_vector_noisy_speech = noisy_multiframe.new_zeros(
            size=(noisy_multiframe.shape[0], 2) + noisy_multiframe.shape[1:]
        )

        cross_covariance_vector_noisy_speech[..., 0, :, :] = (
            noisy_multiframe[:, None, ..., 0, :, :]
            * target_estimate_preliminary[..., 0].conj()[..., None, None]
        )
        num_frames = noisy_outer_products.shape[-3]
        num_frames_warmup = int(2 * noisy_outer_products.shape[-1])

        for frame in torch.arange(num_frames):
            covariance_matrix_noisy[..., frame, :, :] = (
                self.smoothing_constant_recursive_smoothing
                * covariance_matrix_noisy[..., frame - 1, :, :]
                + (1.0 - self.smoothing_constant_recursive_smoothing)
                * noisy_outer_products[..., frame, :, :]
            )
            cross_covariance_vector_noisy_speech[..., frame, :, :] = (
                self.smoothing_constant_recursive_smoothing
                * cross_covariance_vector_noisy_speech[..., frame - 1, :, :]
                + (1.0 - self.smoothing_constant_recursive_smoothing)
                * noisy_multiframe[:, None, ..., frame, :, :]
                * target_estimate_preliminary[..., frame].conj()[..., None, None]
            )

        # compute BMFWF filter
        filters = torch.linalg.solve(
            utils.tik_reg(
                covariance_matrix_noisy[:, None, ..., num_frames_warmup:, :, :],
                self.reg,
            ),
            cross_covariance_vector_noisy_speech[..., num_frames_warmup:, :, :],
        )

        # filtering and retransforming
        target_estimate = (
            filters.mH @ noisy_multiframe[:, None, ..., num_frames_warmup:, :, :]
        )[..., 0, 0]

        target_estimate = torch.cat(
            [
                noisy[:, self.ref_channels, ..., :num_frames_warmup],
                target_estimate,
            ],
            dim=-1,
        )

        out = {
            "input_proc_stft": target_estimate,
            "input_proc": 2 * self.stft.get_istft(target_estimate, length=num_samples),
        }
        return out

    def get_binaural_multiframe_vector(self, singleframe):
        noisy_multiframe = self.get_multiframe_vector(singleframe)
        noisy_multiframe = torch.cat(
            [noisy_multiframe[:, x] for x in torch.arange(noisy_multiframe.shape[1])],
            dim=-2,
        )
        return noisy_multiframe

    def get_multiframe_vector(self, singleframe):
        return F.pad(singleframe, pad=[self.filter_length - 1, 0]).unfold(
            dimension=-1, size=self.filter_length, step=1
        )[..., None]

    def get_features(self, noisy, auxiliary_input: Union[None, torch.Tensor] = None):
        if self.feature_representation == "mag_phase":
            noisy_mag = noisy.abs()
            if self.use_log:
                noisy_mag = (1 + noisy_mag).log10()
            noisy_phase = noisy.angle()
            noisy_phase_cos = noisy_phase.cos()
            noisy_phase_sin = noisy_phase.sin()

            if auxiliary_input is not None:
                aux_mag = auxiliary_input.abs()
                if self.use_log:
                    aux_mag = (1 + aux_mag).log10()
                aux_phase = auxiliary_input.angle()
                aux_phase_cos = aux_phase.cos()
                aux_phase_sin = aux_phase.sin()
                list_features = [
                    noisy_mag,
                    noisy_phase_cos,
                    noisy_phase_sin,
                    aux_phase,
                    aux_phase_cos,
                    aux_phase_sin,
                ]
            else:
                list_features = [
                    noisy_mag,
                    noisy_phase_cos,
                    noisy_phase_sin,
                ]
            features_cat = torch.cat(list_features, dim=1)

        elif self.feature_representation == "real_imag":  # just use real & imag
            if auxiliary_input is not None:
                list_features = [
                    noisy.real,
                    noisy.imag,
                    auxiliary_input.real,
                    auxiliary_input.imag,
                ]
                features_cat = torch.cat(list_features, dim=1)
            else:
                features_cat = torch.cat([noisy.real, noisy.imag], dim=1)
        elif self.feature_representation == "mag":  # just use real & imag
            noisy_mag = noisy.abs()
            if self.use_log:
                noisy_mag = (1 + noisy_mag).log10()

            if auxiliary_input is not None:
                aux_mag = auxiliary_input.abs()
                if self.use_log:
                    aux_mag = (1 + aux_mag).log10()
                list_features = [
                    noisy_mag,
                    aux_mag,
                ]
            else:
                list_features = [
                    noisy_mag,
                ]
            features_cat = torch.cat(list_features, dim=1)
        else:
            raise ValueError("unknown feature representation!")
        return features_cat


class RealTimeBMWF(BMWF):
    def __init__(
        self,
        learning_rate: float = 0.001,
        batch_size: int = 16,
        loss: str = "MagnitudeAbsoluteError",
        metrics_test: Union[tuple, str] = "PESQWB,PESQNB,PESQNBRAW,STOI,ESTOI,SISDR",
        metrics_val: Union[tuple, str] = "",
        frame_length: int = 64,
        fft_length: int = 128,
        shift_length: int = 32,
        filter_length: int = 4,
        fs: int = 16000,
        num_channels: int = 1,
        reg: float = 1e-3,
        time_constant_recursive_smoothing: float = 0.002,
        window_type: str = "hann",
        feature_representation: str = "real_imag",
        auxiliary_input: str = "matched_filter",
        binaural: bool = True,  # binaural or bilateral
        use_mwf: bool = True,
        D: int = 32,
        E: int = 8,
        I: int = 8,  # noqa: E741
        J: int = 4,
        Q: int = None,
        H: int = 128,
        E_prime: int = 64,
        I_prime: int = 5,
        J_prime: int = 5,
        Q_prime: int = None,
        H_prime: int = 32,
        B: int = 2,
        nonlin: str = "Mish",
        use_log: bool = True,
        normalization_type: str = "cGLN",
        use_first_norm: bool = True,
        use_bias: bool = True,
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            batch_size=batch_size,
            loss=loss,
            metrics_test=metrics_test,
            metrics_val=metrics_val,
            frame_length=frame_length,
            fft_length=fft_length,
            shift_length=shift_length,
            filter_length=filter_length,
            fs=fs,
            num_channels=num_channels,
            reg=reg,
            time_constant_recursive_smoothing=time_constant_recursive_smoothing,
            window_type=window_type,
            feature_representation=feature_representation,
            auxiliary_input=auxiliary_input,
            binaural=binaural,
            use_mwf=use_mwf,
            D=D,
            E=E,
            I=I,
            J=J,
            Q=Q,
            H=H,
            E_prime=E_prime,
            I_prime=I_prime,
            J_prime=J_prime,
            Q_prime=Q_prime,
            H_prime=H_prime,
            B=B,
            nonlin=nonlin,
            use_log=use_log,
            normalization_type=normalization_type,
            use_first_norm=use_first_norm,
            use_bias=use_bias,
            **kwargs,
        )

        self.stft = utils.STFTTorchScript(
            frame_length=self.frame_length,
            overlap_length=self.frame_length - self.shift_length,
            window=torch.hann_window(self.frame_length, periodic=True) * 2.3094,
            synthesis_window=torch.ones(self.frame_length),
            sqrt=False,
            fft_length=self.fft_length,
            fft_length_synth=self.frame_length,
        )

        self.states = self.get_initial_states()

    def get_estimators(self):
        super().get_estimators()

        self.estimator_mask_speech = bb.RealTimeFSB_LSTMEstimator(
            input_dim=self.input_size,
            output_dim=self.output_size,
            num_frequencies=self.frequency_bins,
            **self.dnn_params,
            separate_encoders=not self.binaural,
            normalization_type=self.normalization_type,
            use_bias=self.use_bias,
        )

    def get_initial_states(self):
        # restructured version by GPT
        def create_zero_tensor(shape, device=self.device, batch_size=None):
            return torch.zeros(shape, device=device)

        def clone_zero_tensor(shape):
            return utils.clone_tensors(create_zero_tensor(shape))

        def stack_zero_tensors(shape, count):
            return torch.stack([clone_zero_tensor(shape) for _ in range(count)])

        # Calculate extended batch size
        extended_batch_size = int(
            self.batch_size
            * (
                (self.estimator_mask_speech.Q_prime - self.dnn_params["I_prime"])
                / self.dnn_params["J_prime"]
                + 1
            )
        )

        # Shapes and counts for full-band tensors
        hidden_state_shape = (1, self.batch_size, self.dnn_params["H"])
        cum_sum_shape = (self.batch_size, 1)
        entry_cnt_shape = (1, 1)
        B = self.dnn_params["B"]

        # Create full-band tensors
        full_band_hidden_state = stack_zero_tensors(hidden_state_shape, B)
        full_band_cum_sum_1 = stack_zero_tensors(cum_sum_shape, B)
        full_band_cum_pow_sum_1 = stack_zero_tensors(cum_sum_shape, B)
        full_band_entry_cnt_1 = stack_zero_tensors(entry_cnt_shape, B)
        full_band_cum_sum_2 = stack_zero_tensors(cum_sum_shape, B)
        full_band_cum_pow_sum_2 = stack_zero_tensors(cum_sum_shape, B)
        full_band_entry_cnt_2 = stack_zero_tensors(entry_cnt_shape, B)

        # Create sub-band tensors
        sub_band_hidden_state_shape = (
            1,
            extended_batch_size,
            self.dnn_params["H_prime"],
        )
        sub_band_hidden_state = stack_zero_tensors(sub_band_hidden_state_shape, B)
        sub_band_cum_sum = stack_zero_tensors(cum_sum_shape, B)
        sub_band_cum_pow_sum = stack_zero_tensors(cum_sum_shape, B)
        sub_band_entry_cnt = stack_zero_tensors(entry_cnt_shape, B)

        # Common cum_sum, pow_sum, and entry_cnt
        cum_sum = create_zero_tensor(cum_sum_shape)
        cum_pow_sum = create_zero_tensor(cum_sum_shape)
        entry_cnt = create_zero_tensor(entry_cnt_shape)

        # Covariance matrices and vectors
        covariance_matrix_shape = (
            self.batch_size,
            self.frequency_bins,
            1,
            2 * self.num_channels * self.filter_length,
            2 * self.num_channels * self.filter_length,
        )
        cross_covariance_vector_shape = (
            self.batch_size,
            2,
            self.frequency_bins,
            1,
            2 * self.num_channels * self.filter_length,
            1,
        )
        noisy_multiframe_buffer_shape = (
            self.batch_size,
            2 * self.num_channels,
            self.frequency_bins,
            self.filter_length,
        )

        # Initialize covariance-related tensors
        covariance_matrix_noisy_prev = create_zero_tensor(covariance_matrix_shape)
        cross_covariance_vector_noisy_speech_prev = create_zero_tensor(
            cross_covariance_vector_shape
        )
        noisy_multiframe_buffer = create_zero_tensor(noisy_multiframe_buffer_shape)

        # Return all initialized states
        return {
            "full_band_hidden_state": full_band_hidden_state,
            "full_band_cum_sum_1": full_band_cum_sum_1,
            "full_band_cum_pow_sum_1": full_band_cum_pow_sum_1,
            "full_band_entry_cnt_1": full_band_entry_cnt_1,
            "full_band_cum_sum_2": full_band_cum_sum_2,
            "full_band_cum_pow_sum_2": full_band_cum_pow_sum_2,
            "full_band_entry_cnt_2": full_band_entry_cnt_2,
            "sub_band_hidden_state": sub_band_hidden_state,
            "sub_band_cum_sum": sub_band_cum_sum,
            "sub_band_cum_pow_sum": sub_band_cum_pow_sum,
            "sub_band_entry_cnt": sub_band_entry_cnt,
            "cum_sum": cum_sum,
            "cum_pow_sum": cum_pow_sum,
            "entry_cnt": entry_cnt,
            "covariance_matrix_noisy_prev": covariance_matrix_noisy_prev,
            "cross_covariance_vector_noisy_speech_prev": cross_covariance_vector_noisy_speech_prev,
            "noisy_multiframe_buffer": noisy_multiframe_buffer,
        }

    @torch.jit.export
    def forward_utterance_stft(
        self,
        x,
    ):
        # B x 2M x L; 2M: first all left channels, then all right channels
        target_estimate = []
        for frame_index in torch.arange(x.shape[-1]):
            target_estimate.append(
                torch.view_as_complex(
                    self.forward(frame_index, x[:, :, :, frame_index], 0.0, 1.0)[0]
                )
            )  # B x 2M x F

        target_estimate = torch.stack(target_estimate, dim=-1)  # B x 2M x F x T
        return {"input_proc": target_estimate}

    def forward(
        self,
        frame_index: int,
        frame: torch.Tensor,
        mix_back: float = 0.0,
        calib_factor_lin: float = 1.0,
    ):
        # frame: B x 4 x F
        noisy = frame[..., None]  # B x 4 x F x T
        noisy = noisy[:, [0, 2, 1, 3]]
        # l,l,r,r

        if self.auxiliary_input == "matched_filter":
            auxiliary_input = (
                (
                    self.hrtf_front.unsqueeze(-1).unsqueeze(-3).mH
                    @ noisy.permute(0, 2, 3, 1).unsqueeze(-1).unsqueeze(1)
                )
                .squeeze(-1)
                .squeeze(-1)
            )  # B x 2 x F x T
        elif self.auxiliary_input == "none":
            auxiliary_input = None
        else:
            raise ValueError(f"unknown auxiliary input {self.auxiliary_input}!")

        if self.num_channels == 1:
            noisy = noisy[:, [0, 2]]

        # put new frame into buffer
        self.states["noisy_multiframe_buffer"] = torch.cat(
            [self.states["noisy_multiframe_buffer"][..., 1:], noisy], dim=-1
        )  # B x 2M x F x N

        # extract multi-frame vector from buffer
        noisy_multiframe = (
            self.states["noisy_multiframe_buffer"]
            .transpose(-3, -2)
            .reshape(
                self.batch_size,
                self.frequency_bins,
                2 * self.num_channels * self.filter_length,
            )
            .unsqueeze(-2)
            .unsqueeze(-1)
        )  # B x F x T x 2MN x 1

        features = self.get_features(
            noisy, auxiliary_input
        )  # B x 2 * 2 * (1 + binaural) x F x T

        (
            output_estimator,
            self.states["full_band_hidden_state"],
            self.states["full_band_cum_sum_1"],
            self.states["full_band_cum_pow_sum_1"],
            self.states["full_band_entry_cnt_1"],
            self.states["full_band_cum_sum_2"],
            self.states["full_band_cum_pow_sum_2"],
            self.states["full_band_entry_cnt_2"],
            self.states["sub_band_hidden_state"],
            self.states["sub_band_cum_sum"],
            self.states["sub_band_cum_pow_sum"],
            self.states["sub_band_entry_cnt"],
            self.states["cum_sum"],
            self.states["cum_pow_sum"],
            self.states["entry_cnt"],
        ) = self.estimator_mask_speech(
            features,
            self.states["full_band_hidden_state"],
            self.states["full_band_cum_sum_1"],
            self.states["full_band_cum_pow_sum_1"],
            self.states["full_band_entry_cnt_1"],
            self.states["full_band_cum_sum_2"],
            self.states["full_band_cum_pow_sum_2"],
            self.states["full_band_entry_cnt_2"],
            self.states["sub_band_hidden_state"],
            self.states["sub_band_cum_sum"],
            self.states["sub_band_cum_pow_sum"],
            self.states["sub_band_entry_cnt"],
            self.states["cum_sum"],
            self.states["cum_pow_sum"],
            self.states["entry_cnt"],
        )  # B * (2 - binaural) x 2 * (1 + binaural) x F x T

        output_estimator = torch.sigmoid(output_estimator)

        if self.binaural:
            output_estimator = torch.stack(
                [
                    output_estimator[:, :1],
                    output_estimator[:, 1:],
                ],
                dim=1,
            )  # B x 2 x 1 x F x T
        output_estimator = output_estimator.squeeze(2)

        # apply estimator output to reference channels to obtain target estimates
        target_estimate_preliminary = output_estimator * noisy

        if not self.use_mwf:
            target_estimate_preliminary = target_estimate_preliminary.squeeze(-1)
            return torch.stack(
                [target_estimate_preliminary.real, target_estimate_preliminary.imag],
                dim=-1,
            )

        # update second-order statistics
        covariance_matrix_noisy = (
            self.smoothing_constant_recursive_smoothing
            * self.states["covariance_matrix_noisy_prev"]
            + (1.0 - self.smoothing_constant_recursive_smoothing)
            * (noisy_multiframe @ noisy_multiframe.mH)[..., :1, :, :]
        )
        self.states["covariance_matrix_noisy_prev"] = covariance_matrix_noisy

        cross_covariance_vector_noisy_speech = (
            self.smoothing_constant_recursive_smoothing
            * self.states["cross_covariance_vector_noisy_speech_prev"]
            + (1.0 - self.smoothing_constant_recursive_smoothing)
            * noisy_multiframe[:, None, ..., :1, :, :]
            * target_estimate_preliminary[..., :1].conj()[..., None, None]
        )
        self.states["cross_covariance_vector_noisy_speech_prev"] = (
            cross_covariance_vector_noisy_speech
        )

        filters = torch.linalg.solve(
            utils.tik_reg(
                covariance_matrix_noisy[:, None],
                self.reg,
            ),
            cross_covariance_vector_noisy_speech,
        )
        if frame_index < 2 * covariance_matrix_noisy.shape[-1]:
            # replace BMFWF filter with identity filter because STCM is not warmed up yet
            filters = torch.zeros_like(filters)
            filters[:, 0, ..., self.ref_indices[0], :] = 1.0
            filters[:, 1, ..., self.ref_indices[1], :] = 1.0

        target_estimate = (filters.mH @ noisy_multiframe[:, None])[..., 0, 0, 0]
        if mix_back != 0.0:
            if self.num_channels == 2:
                noisy_mix_back = noisy[:, [0, 2]]
            elif self.num_channels == 1:
                noisy_mix_back = noisy
            else:
                raise ValueError("only 1 or 2 channels supported!")
            target_estimate = (
                1.0 - mix_back
            ) * calib_factor_lin * target_estimate + mix_back * noisy_mix_back[..., 0]
        else:
            target_estimate = calib_factor_lin * target_estimate
        return torch.stack([target_estimate.real, target_estimate.imag], dim=-1)
