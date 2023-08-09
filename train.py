import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchaudio import transforms
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
import numpy as np
from dataset import data
from model import MyConvModel


def remove_paths(data):
    logmels = [sample[0] for sample in data] 
    labels = [sample[1] for sample in data]
    logmels = torch.cat(logmels).unsqueeze(1)
    labels = torch.tensor(labels).type(torch.int64)
    return logmels, labels

def padd(dataset, max_length=None, min_value=None):
    logmels = [x for x, _, _ in dataset]
    labels = [y for _, y, _ in dataset]
    audio = [z for _, _, z in dataset]
    
    if max_length is None:
        max_length = max([len(x[0][0]) for x in logmels])
        
    if min_value is None:
        min_value = torch.min(torch.stack([torch.min(torch.flatten(x)) for x in logmels])).item()

    padded_data = []
    for x, y, audio in dataset:
        pad_width = max_length - len(x[0][0])
        x_padded = torch.nn.functional.pad(x, (0, pad_width), value=min_value)
        padded_data.append((torch.tensor(x_padded), y, audio))
    
    print("Valor maximo:", max_length)
    print("Relleno:", min_value)
        
    return padded_data


class featurizer_pipeline(nn.Module):
    def __init__(self, params):
        super().__init__()
        sample_rate = params["sampling_rate"]
        win_length = int(0.05*sample_rate)
        n_mels = params["n_mel_coefs"]
        self.resampler = transforms.Resample(44100, sample_rate)
        self.mel_transform = transforms.MelSpectrogram(sample_rate=sample_rate,
                                                       win_length=win_length,
                                                       n_mels=n_mels)

    def forward(self, x, eps=1e-10):
        mel_spectrogram = self.mel_transform(self.resampler(x))
        return (mel_spectrogram + eps).log()
    


if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml"))
    print("Preparando datos")
    pipeline = featurizer_pipeline(params["features"])
    data = data('data_train', transforms=pipeline)
    DATA = padd(data)
    seed = params['training']['dataset_split_seed']
    train_set, valid_set, _ = random_split(DATA, [334,84, 0],generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_set, shuffle=True,batch_size=params['training']['batch_size'],
                              num_workers=4, collate_fn=remove_paths)
    valid_loader = DataLoader(valid_set, batch_size=60, num_workers=4, collate_fn=remove_paths)

    print("Iniciando el entrenamiento")
    
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode='min', save_top_k=1, filename='best_model')
    early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=100)
    trainer = L.Trainer(accelerator="auto", max_epochs=params['training']['max_epochs'], 
                    check_val_every_n_epoch=1, logger=[CSVLogger("logs"), TensorBoardLogger("logs")],
                    callbacks=[checkpoint_callback, early_stopping])

    trainer.fit(MyConvModel(params['training']), train_dataloaders=train_loader, val_dataloaders=valid_loader)
