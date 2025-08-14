from pathlib import Path
import os
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm
import numpy as np
from rich.table import Table
from rich.console import Console
import warnings
import argparse
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

from dataset_loader.dataset_tools import show_dataset_stats


class Dataset_Pump(Dataset):
    def __init__(
        self,
        args: argparse.Namespace,
        mode: str = "train",
        use_one_file: bool = False,
        use_to_the_end: bool = False,
    ) -> None:
        super().__init__()
        self.args = args
        self.x = []
        self.y = []
        self.data_folder = Path(args.root_folder / "dataset" / args.data_name)

        # Get label encoder (from all data)
        self.get_label_encoder()

        # Read all dataframes
        csv_files = sorted(self.data_folder.glob("*.csv"))
        if mode == "train":
            selected_files = csv_files[:24]  # First 24 files for training
        elif mode == "val":
            selected_files = csv_files[24:32]  # Next 8 files for validation
        elif mode == "test":
            selected_files = csv_files[32:]  # Last 8 files for testing
        else:
            raise ValueError(
                "Invalid mode specified. Choose between 'train', 'val', 'test'."
            )
        if use_one_file:
            selected_files = [selected_files[0]]
        dfs = self.read_dfs(selected_files)

        # Create x (features) and y (labels) for each df
        for df in dfs:
            if self.args.downsample_rate:
                df = df.iloc[:: self.args.downsample_rate, :]  # select every nth row
                assert len(df) >= self.args.seq_len, "Downsampled df is too short"
            features = df.iloc[:, :-1].values
            labels = df.iloc[:, -1].values
            if use_to_the_end:
                self.x.append(features)
                self.y.append(labels)
            else:
                for i in range(
                    0, len(df) - self.args.seq_len + 1, self.args.window_stride
                ):
                    self.x.append(features[i : i + self.args.seq_len])
                    self.y.append(
                        labels[i : i + self.args.seq_len][-self.args.pred_len :]
                    )

        # Flatten self.y and encode the labels
        flattened_y = np.concatenate(self.y)
        encoded_y = self.label_encoder.transform(flattened_y)

        # Remap self.y
        current_idx = 0
        for i in range(len(self.y)):
            seq_length = len(self.y[i])
            self.y[i] = encoded_y[current_idx : current_idx + seq_length]  # type: ignore
            current_idx += seq_length

    def get_label_encoder(self) -> None:
        # Read all dataframes
        dfs = self.read_dfs(sorted(self.data_folder.glob("*.csv")))

        # Make sure all dfs have the same columns
        assert all(
            list(df.columns) == list(dfs[0].columns) for df in dfs
        ), "All dataframes must have the same columns"

        # Get all labels
        y_all = np.concatenate([df.iloc[:, -1].values for df in dfs])  # type: ignore

        # Encode the labels
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(y_all)
        # print(f"K: {len(self.label_encoder.classes_)}")
        # print(f"Classes: {self.label_encoder.classes_}")

    def read_dfs(self, selected_files: list[Path]) -> list[pd.DataFrame]:
        dfs = []
        for csv_file in selected_files:
            df = pd.read_csv(csv_file)
            df.drop(columns=["t"], inplace=True)
            dfs.append(df)
        assert len(dfs) > 0, "No dataframes read"
        return dfs

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.x[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y
