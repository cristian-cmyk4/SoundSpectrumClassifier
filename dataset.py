from typing import Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import torch
import torch.utils.data as tdata
import torchaudio


class data(tdata.Dataset):

    def __init__(self, root_path: str, transforms=None):
        meta_path = Path(root_path) / "meta" / "meta.csv"
        audio_path = Path(root_path) / "audio"
        df = pd.read_csv(meta_path)
        df["Label"] = df["Label"].replace(["keysdrop"], "keys")
        file_names = df["Audio_File"]
        labels_str = df["Label"]
        self.categories = list(labels_str.unique())
        self.le = LabelEncoder().fit(self.categories)
        self.labels = self.le.transform(labels_str)
        self.file_paths = file_names.apply(lambda file_name: audio_path / file_name)
        self.waveforms = []
        for file in tqdm(self.file_paths):
            waveform, sample_rate = torchaudio.load(file)
            if transforms is not None:
                waveform = transforms(waveform)
            self.waveforms.append(waveform)


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        return self.waveforms[idx], self.labels[idx], self.file_paths[idx]

    def label_int2str(self, label: np.ndarray) -> np.ndarray:
        return self.le.inverse_transform(label)

    def __len__(self) -> int:
        return len(self.labels)
