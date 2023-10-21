import shutil
import re
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

import numpy as np
import scipy
import torch.utils.data
import torchvision
from sklearn.model_selection import train_test_split


DATA_URL = "https://github.com/XiongMeijing/CWRU-1/archive/refs/heads/master.zip"
ZIPFILE_PATH = Path("CWRU-1-master.zip")
DOWNLOAD_DIR = Path("CWRU-1-master")


def make_dataloaders(
    data_dir: Path, subset: str, batch_size: int, random_state: int | None = None
) -> tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    if not data_dir.exists():
        download_data(data_dir)
    train_ds, test_ds, val_ds = make_datasets(data_dir, subset, random_state)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size)
    return train_dl, test_dl, val_dl


def download_data(data_dir: Path) -> None:
    urllib.request.urlretrieve(DATA_URL, ZIPFILE_PATH)
    with zipfile.ZipFile(ZIPFILE_PATH) as zip_ref:
        zip_ref.extractall()
    (DOWNLOAD_DIR / "Data").rename(data_dir)
    (data_dir / ".gitignore").write_text("*")
    ZIPFILE_PATH.unlink()
    shutil.rmtree(DOWNLOAD_DIR)


def make_datasets(
    data_dir: Path, subset: str, random_state: int | None = None
) -> tuple[
    torch.utils.data.ConcatDataset,
    torch.utils.data.ConcatDataset,
    torch.utils.data.ConcatDataset,
]:
    data_paths = list_data_paths(data_dir, subset)
    train_paths, test_paths, val_paths = split_data_paths(
        data_paths, test_size=0.1, val_size=0.1, random_state=random_state
    )
    transforms = [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((32, 32), antialias=True),
    ]
    transform = torchvision.transforms.Compose(transforms)
    train_ds = make_dataset(train_paths, subset, transform)
    test_ds = make_dataset(test_paths, subset, transform)
    val_ds = make_dataset(val_paths, subset, transform)
    return train_ds, test_ds, val_ds


def list_data_paths(data_dir: Path, subset: str) -> list[Path]:
    normal_data_paths = list((data_dir / "Normal").glob("*.mat"))
    fault_data_paths = list((data_dir / subset).glob("*.mat"))
    data_paths = normal_data_paths + fault_data_paths
    return data_paths


def split_data_paths(
    data_paths: list[Path],
    test_size: float,
    val_size: float,
    random_state: int | None = None,
) -> tuple[list[Path], list[Path], list[Path]]:
    eval_size = test_size + val_size
    train_paths, eval_paths = train_test_split(
        data_paths, test_size=eval_size, random_state=random_state
    )
    test_paths, val_paths = train_test_split(
        train_paths, test_size=val_size / eval_size, random_state=random_state
    )
    return train_paths, test_paths, val_paths


def make_dataset(
    data_paths: list[Path],
    subset: str,
    transform: Callable[[np.ndarray], torch.FloatTensor],
) -> torch.utils.data.ConcatDataset:
    dataset = torch.utils.data.ConcatDataset(
        CWRUBearing(data_path, subset, transform) for data_path in data_paths
    )
    min_value, max_value = find_min_max(dataset)
    transforms = [
        transform,
        torchvision.transforms.Normalize(min_value, max_value - min_value),
    ]
    transform = torchvision.transforms.Compose(transforms)
    dataset = torch.utils.data.ConcatDataset(
        CWRUBearing(data_path, subset, transform) for data_path in data_paths
    )
    return dataset


def find_min_max(dataset: torch.utils.data.Dataset) -> tuple[float, float]:
    min_value = float("inf")
    max_value = float("-inf")
    for spectrogram, _ in dataset:
        min_value = min(min_value, spectrogram.min())
        max_value = max(max_value, spectrogram.max())
    return min_value, max_value


class CWRUBearing(torch.utils.data.Dataset):
    filename_regex = re.compile(
        r"""
        ([a-zA-Z]+)  # Fault
        (\d{3})?  # Fault diameter
        (@\d+)?  # Fault location
        _
        (\d+)  # Load
        \.mat
        """,
        re.VERBOSE,
    )
    faults = ["Normal", "B", "IR", "OR"]
    motor_speeds = [1797, 1772, 1750, 1730]
    segment_lengths = {"12k": 1024, "48k": 4096}

    def __init__(
        self,
        data_path: Path,
        subset: str,
        transform: Callable[[np.ndarray], torch.FloatTensor],
    ) -> None:
        super().__init__()
        self.data_path = data_path
        sampling_freq, end = subset.split("_")
        data = scipy.io.loadmat(str(data_path))
        *_, self.signal_key = [
            key for key in data.keys() if key.endswith(f"{end}_time")
        ]
        signal = data[self.signal_key].squeeze()
        self.segment_length = self.segment_lengths[sampling_freq]
        self.num_segments = len(signal) // self.segment_length
        match = self.filename_regex.fullmatch(data_path.name)
        self.fault = match.group(1)
        load = int(match.group(4))
        motor_speed = self.motor_speeds[load]
        sampling_freq = 12000 if sampling_freq == "12k" else 48000
        self.nperseg = sampling_freq * 60 // motor_speed
        self.noverlap = self.nperseg - self.segment_length // 30
        self.transform = transform

    def __len__(self) -> int:
        return self.num_segments

    def __getitem__(self, idx: int) -> tuple[torch.FloatTensor, torch.LongTensor]:
        data = scipy.io.loadmat(str(self.data_path))
        signal = data[self.signal_key].squeeze()
        segment = signal[self.segment_length * idx : self.segment_length * (idx + 1)]
        *_, spectrogram = scipy.signal.stft(
            segment, nperseg=self.nperseg, noverlap=self.noverlap
        )
        spectrogram = np.abs(spectrogram)
        spectrogram = self.transform(spectrogram)
        target = torch.tensor(self.faults.index(self.fault))
        return spectrogram, target  # type: ignore
