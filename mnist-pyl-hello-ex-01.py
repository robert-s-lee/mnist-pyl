# from https://colab.research.google.com/github/PytorchLightning/pytorch-lightning/blob/master/notebooks/01-mnist-hello-world.ipynb

import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.loggers import TensorBoardLogger

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--epochs', default=2, type=int)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--lr', default=0.02, type=float)

parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--data_dir', default=os.getcwd(), type=str)    # should be datastore
parser.add_argument('--tb_dir', default="TB", type=str)
parser.add_argument('--tb_name', default="MNINST/ex_01", type=str)

if __name__ == "__main__.py":    
    args = parser.parse_args()      
else:
    args = parser.parse_args("")    # take defaults in Jupyter 

class MNISTModel(pl.LightningModule):

    def __init__(self):
        super(MNISTModel, self).__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=args.lr)    

# Init our model
mnist_model = MNISTModel()

# Init DataLoader from MNIST Dataset in MINST dir
train_ds = MNIST(args.data_dir, train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_ds, batch_size=args.batch_size)

# Initialize a trainer
trainer = pl.Trainer(gpus=args.gpu, max_epochs=args.epochs, progress_bar_refresh_rate=20, logger=TensorBoardLogger(args.tb_dir, name=args.tb_name)) 

# Train the model ⚡
trainer.fit(mnist_model, train_loader)