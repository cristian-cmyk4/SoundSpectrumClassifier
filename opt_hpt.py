import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from train import featurizer_pipeline, remove_paths, padd
from torchaudio import transforms
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
import numpy as np
from dataset import data
from model import MyConvModel
import optuna
import yaml

def objective(trial):
    params = yaml.safe_load(open("params.yaml"))
    
    sampling_rate = trial.suggest_int('sampling_rate', 2000, 8000, step=1000)
    pipeline = featurizer_pipeline({'sampling_rate': sampling_rate, 'n_mel_coefs': 64})
 
    DATA = data('data_train', transforms=pipeline)
    padd_data = padd(DATA)
    
    seed = params['training']['dataset_split_seed']
    train_set, valid_set, _ = random_split(padd_data, [334,84, 0],generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_set, shuffle=True,batch_size=params['training']['batch_size'],
                              num_workers=4, collate_fn=remove_paths)
    valid_loader = DataLoader(valid_set, batch_size=60, num_workers=4, collate_fn=remove_paths)
    model = MyConvModel(params['training']) 
    trainer = L.Trainer(max_epochs=100, logger=False)
    trainer.fit(model,train_dataloaders=train_loader, val_dataloaders=valid_loader)
    
    correct = 0
    total = 0
    for batch in valid_loader:
        x, y = batch
        y_pred = model(x)
        predicted = torch.argmax(y_pred, dim=1)
        correct += (predicted == y).sum().item()
        total += len(y)
    accuracy = correct / total
   
    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)

    best_params = study.best_params
    best_accuracy = study.best_value
    print(best_params)
