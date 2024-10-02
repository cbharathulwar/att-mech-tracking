import itertools
import os
from pathlib import Path
from typing import Any

import torch
from pytorch_lightning import LightningDataModule, Trainer
from torch.utils.data import RandomSampler
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

# Assuming 'logger' is defined in your 'gnn_tracking.utils.log' module
from gnn_tracking.utils.log import logger


# TrackingDataset class (from your provided code)
class TrackingDataset(Dataset):
    def __init__(
        self,
        in_dir: str | os.PathLike | list[str] | list[os.PathLike],
        *,
        start=0,
        stop=None,
        sector: int | None = None,
    ):
        super().__init__()
        self._processed_paths = self._get_paths(
            in_dir, start=start, stop=stop, sector=sector
        )

    def _get_paths(
        self,
        in_dir: str | os.PathLike | list[str] | list[os.PathLike],
        *,
        start=0,
        stop: int | None = None,
        sector: int | None = None,
    ) -> list[Path]:
        if start == stop:
            return []

        glob_pattern = "*.pt" if sector is None else f"*_s{sector}.pt"

        if not isinstance(in_dir, list):
            in_dir = [in_dir]
        for d in in_dir:
            if not Path(d).exists():
                msg = f"Directory {d} does not exist."
                raise FileNotFoundError(msg)

        available_files = sorted(
            itertools.chain.from_iterable([Path(d).glob(glob_pattern) for d in in_dir])
        )

        if stop is not None and stop > len(available_files):
            msg = f"stop={stop} is larger than the number of files ({len(available_files)})"
            raise ValueError(msg)
        considered_files = available_files[start:stop]
        logger.info(
            "DataLoader will load %d graphs (out of %d available).",
            len(considered_files),
            len(available_files),
        )
        if considered_files:
            logger.debug(
                "First graph is %s, last graph is %s",
                considered_files[0],
                considered_files[-1],
            )
        return considered_files

    def len(self) -> int:
        return len(self._processed_paths)

    def get(self, idx: int) -> Data:
        return torch.load(self._processed_paths[idx])


# TrackingDataModule class (from your provided code)
class TrackingDataModule(LightningDataModule):
    def __init__(
        self,
        *,
        identifier: str,  # noqa: ARG002
        train: dict | None = None,
        val: dict | None = None,
        test: dict | None = None,
        cpus: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._configs = {
            "train": self._fix_datatypes(train),
            "val": self._fix_datatypes(val),
            "test": self._fix_datatypes(test),
        }
        self._datasets = {}
        self._cpus = cpus

    @property
    def datasets(self) -> dict[str, TrackingDataset]:
        if not self._datasets:
            logger.error(
                "Datasets have not been loaded yet. Make sure to call the setup method."
            )
        return self._datasets

    @staticmethod
    def _fix_datatypes(dct: dict[str, Any] | None) -> dict[str, Any] | None:
        if dct is None:
            return {}
        for key in ["start", "stop", "sector", "batch_size", "sample_size"]:
            if key in dct:
                dct[key] = int(dct[key])
        return dct

    def _get_dataset(self, key: str) -> TrackingDataset:
        config = self._configs[key]
        if not config:
            msg = f"DataLoaderConfig for key '{key}' is None."
            raise ValueError(msg)
        in_dir = config["dirs"]
        return TrackingDataset(
            in_dir=in_dir,
            start=config.get("start", 0),
            stop=config.get("stop", None),
            sector=config.get("sector", None),
        )

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self._datasets["train"] = self._get_dataset("train")
            self.setup("validate")
        elif stage == "validate":
            self._datasets["val"] = self._get_dataset("val")
        elif stage == "test":
            self._datasets["test"] = self._get_dataset("test")
        else:
            msg = f"Unknown stage '{stage}'"
            raise ValueError(msg)

    def _get_dataloader(self, key: str) -> DataLoader:
        sampler = None
        dataset = self._datasets[key]
        n_samples = len(dataset)
        if key == "train" and n_samples:
            if "max_sample_size" in self._configs[key]:
                msg = "max_sample_size has been replaced by sample_size"
                raise ValueError(msg)
            n_samples = self._configs[key].get("sample_size", n_samples)
            sampler = RandomSampler(
                dataset,
                replacement=n_samples > len(dataset),
                num_samples=n_samples,
            )
        return DataLoader(
            dataset,
            batch_size=self._configs[key].get("batch_size", 1),
            num_workers=max(1, min(n_samples, self._cpus)),
            sampler=sampler,
            pin_memory=True,
        )

    def train_dataloader(self):
        return self._get_dataloader("train")

    def val_dataloader(self):
        return self._get_dataloader("val")

    def test_dataloader(self):
        return self._get_dataloader("test")


# Step 2: Define the configuration dictionaries
train_config = {
    "dirs": [
        "/home/cbharathulwar/higgsgnn/main-code/gnn_tracking/test-data/data/point_clouds/train/"
    ],
    "batch_size": 32,
    "start": 0,
    # 'stop': None,
}

val_config = {
    "dirs": [
        "/home/cbharathulwar/higgsgnn/main-code/gnn_tracking/test-data/data/point_clouds/val/"
    ],
    "batch_size": 32,
    "start": 0,
}

test_config = {
    "dirs": [
        "/home/cbharathulwar/higgsgnn/main-code/gnn_tracking/test-data/data/point_clouds/test/"
    ],
    "batch_size": 32,
    "start": 0,
}

# Step 3: Instantiate the TrackingDataModule
data_module = TrackingDataModule(
    identifier="your_dataset_identifier",
    train=train_config,
    val=val_config,
    test=test_config,
    cpus=4,
)

# Step 4: Set up the data module
data_module.setup("fit")

# Step 5: Access the DataLoaders
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()
test_loader = data_module.test_dataloader()

# Step 6: Create your model
# Replace 'YourModel' with your actual model class
model = YourModel()

# Step 7: Set up the trainer
trainer = Trainer(max_epochs=10)

# Step 8: Start training
trainer.fit(model, datamodule=data_module)
