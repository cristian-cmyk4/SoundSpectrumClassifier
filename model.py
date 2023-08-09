import torch.optim as optim
import torch.nn as nn
import lightning as L
import torch
from torchmetrics import Accuracy


class MyConvModel(L.LightningModule):
    
    

    def __init__(self, params: dict):
        super(type(self), self).__init__()
        self.learning_rate = params['learning_rate']
        self.criterion = nn.CrossEntropyLoss(reduction='mean')  
        self.save_hyperparameters()
        self.features = nn.Sequential(
                            nn.Conv2d(1, 16, 5, padding=2),
                            nn.BatchNorm2d(16),
                            nn.ReLU(),
                            nn.MaxPool2d(2),
                            nn.Dropout2d(p=params['drop_rate']),  
                            nn.Conv2d(16, 32, 5, padding=2),
                            nn.BatchNorm2d(32),
                            nn.ReLU(),
                            nn.MaxPool2d(2),
                            nn.Dropout2d(p=params['drop_rate']),  
                            nn.Conv2d(32, 64, 5, padding=2),
                            nn.BatchNorm2d(64),
                            nn.ReLU(),
                            nn.MaxPool2d(2),
                            nn.Dropout2d(p=params['drop_rate'])) 
        
        
        self.classifier = nn.Sequential(
                            nn.Linear(64*8*16, 256),
                            nn.ReLU(),
                            nn.Dropout(params['drop_rate']),
                            nn.Linear(256, 128),
                            nn.ReLU(),
                            nn.Dropout(params['drop_rate']),
                            nn.Linear(128, 11))
                                     

    def forward(self, x):
        z = self.features(x)
        #print(z)
        z = z.view(-1, 64*8*16)
        return self.classifier(z)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), 
                                lr=self.learning_rate)
    
    def training_step(self, batch, batch_idx):
        data, labels = batch
        preds = self.forward(data)
        #print(preds,labels)
        loss = self.criterion(preds, labels)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        data, labels = batch
        preds = self.forward(data)
        loss = self.criterion(preds, labels)
        self.log("val_loss", loss) 
       

