from math import sqrt
import os
import lightning as pl
import soundfile as sf
import torch
import torchaudio as ta
from . import utils
from torch.utils.data import DataLoader


EPS = torch.as_tensor(torch.finfo(torch.get_default_dtype()).eps)


class H4a2RLFixedDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset class for loading and processing audio data for training.

    Args:
        root_dir (str): The root directory containing the dataset.
        target_type (str, optional): The type of target data. Defaults to "target_2".

    Attributes:
        root_dir (str): The root directory containing the dataset.
        target_type (str): The type of target data.
        length (int): The number of files in the "mix" directory.
        dirs (dict): A dictionary containing paths to the "mix" and target directories.
        channels (tuple): The channels to be used from the mix data (in order [l_front,l_back,r_front,r_back]).
        ref_channels (tuple): The reference channels to be used for evaluation.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Loads and returns the input, target signals, and metadata for a given index.

    Example:
        dataset = H4a2RLFixedDataset(root_dir="/path/to/data", target_type="target_2")
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    """

    def __init__(self, root_dir: str, target_type: str = "target_2"):
        self.root_dir = root_dir
        self.target_type = target_type

        self.length = len(os.listdir(os.path.join(root_dir, "mix")))
        self.dirs = {
            "mix": os.path.join(root_dir, "mix"),
            "target": os.path.join(root_dir, self.target_type),
        }
        if self.target_type == "reverberant":
            # required to compute reverberant target
            self.dirs["interference"] = os.path.join(root_dir, "interference")

        self.channels = (0, 2, 1, 3)
        self.ref_channels = (0, 2)

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        """
        Retrieves the audio data and corresponding metadata for a given index.

        Args:
            idx (int): The index of the audio file to retrieve.

        Returns:
            tuple: A tuple containing:
                - signals (dict): A dictionary with the following keys:
                    - "input" (torch.Tensor): The mixed audio signal. Dimensions are (n_channels==4, n_samples).
                    - "input_eval" (torch.Tensor): The mixed audio signal for evaluation. Dimensions are (2, n_samples).
                    - "target" (torch.Tensor): The target audio signal. Dimensions are (2, n_samples).
                - meta (dict): A dictionary with metadata, including:
                    - "idx" (int): The index of the audio file.
                    - "filename_mix" (str): The file path of the mixed audio file.
        """
        mix, _ = sf.read(os.path.join(self.dirs["mix"], f"file_{idx}.wav"))

        if self.target_type == "reverberant":
            interference, _ = sf.read(
                os.path.join(self.dirs["interference"], f"file_{idx}.wav")
            )
            target = (
                mix[:, self.channels][:, self.ref_channels]
                - interference[:, self.channels][:, self.ref_channels]
            )
        else:
            target, _ = sf.read(os.path.join(self.dirs["target"], f"file_{idx}.wav"))

        mix = mix[:, self.channels]
        signals = {
            "input": torch.Tensor(mix).T,
            "input_eval": torch.Tensor(mix[:, self.ref_channels]).T,
            "target": torch.Tensor(target).T,
        }
        meta = {
            "idx": idx,
            "filename_mix": os.path.join(self.dirs["mix"], f"file_{idx}.wav"),
        }
        return signals, meta


class H4a2RLFixedDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        target_type: str = "target_2",
        batch_size: int = 32,
        num_workers: int = 8,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.target_type = target_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = utils.collate_fn_signals_meta

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = H4a2RLFixedDataset(
                os.path.join(self.data_dir, "train"), self.target_type
            )
            self.val_dataset = H4a2RLFixedDataset(
                os.path.join(self.data_dir, "validation"), self.target_type
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, length: int = 32):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        # seed one generator for the inputs using idx and the one of the target using idx+1
        signals = {
            "input": torch.normal(
                0.0, sqrt(0.1), (4, 64000), generator=torch.Generator().manual_seed(idx)
            ),  # HA has 2 channels per ear
            "input_eval": torch.normal(
                0, sqrt(0.1), (4, 64000), generator=torch.Generator().manual_seed(idx)
            ),
            "target": torch.normal(
                0,
                sqrt(0.1),
                (2, 64000),
                generator=torch.Generator().manual_seed(idx + 1),
            ),  # binaural --> 2 channels
        }
        return signals, {"idx": idx}


class DummyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 8,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = utils.collate_fn_signals_meta

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = DummyDataset()
            self.val_dataset = DummyDataset()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
