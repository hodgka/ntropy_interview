from argparse import ArgumentParser
import os

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

from torchvision.datasets.mnist import MNIST
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin

import torchmetrics

os.environ["NCCL_NSOCKS_PERTHREAD"] = "4"
os.environ["NCCL_SOCKET_NTHREADS"] = "5"
os.environ["NCCL_MIN_NCHANNELS"] = "32"

class Backbone(torch.nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, hidden_dim)
        self.l2 = torch.nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return x


class LitClassifier(pl.LightningModule):
    def __init__(self, backbone, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        self.backbone = backbone
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        # self.precision = Precision(num_classes=10, average='macro'),
        # self.recall = Recall(num_classes=10, average='macro')


    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.backbone(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.train_acc(y_hat, y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.val_acc(y_hat, y)
        self.log('val_acc', self.val_acc, on_step=True, on_epoch=True)
        self.log('val_loss', loss, on_step=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.test_acc(y_hat, y)
        self.log('test_acc', self.test_acc, on_step=True, on_epoch=True)
        self.log('test_loss', loss)
    
    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters(), lr=(self.hparams.learning_rate))

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_optimizer_args(torch.optim.Adam)
        parser.add_lr_scheduler_args(torch.optim.lr_scheduler.ExponentialLR)
        parser.add_argument('--learning_rate', type=float, default=0.001)
        return parser

def prepare_data(args):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
    dataset = MNIST('', train=True, download=True, transform=transform)
    mnist_test = MNIST('', train=False, download=True, transform=transform)
    mnist_train, mnist_val = random_split(dataset, [55000, 5000])

    train_loader = DataLoader(mnist_train, batch_size=args.batch_size, num_workers=12)
    val_loader = DataLoader(mnist_val, batch_size=args.batch_size, num_workers=12)
    test_loader = DataLoader(mnist_test, batch_size=args.batch_size, num_workers=12)
    return train_loader, val_loader, test_loader


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser(conflict_handler='resolve')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--gpus', default=2)
    parser.add_argument('--auto_lr_find', action="store_true")
    parser.add_argument('--accelerator', default="ddp", type=str)
    
    # parser.add_argument('--output_path', type=str, default="")
    parser = LitClassifier.add_model_specific_args(parser)
    args = parser.parse_args()
    print(' '.join(f'{k}={v}\n' for k, v in sorted(vars(args).items())))

    # ------------
    # data
    # ------------
    train_loader, val_loader, test_loader = prepare_data(args)

    # ------------
    # model
    # ------------
    model = LitClassifier(Backbone(hidden_dim=args.hidden_dim), args.learning_rate)

    # ------------
    # training
    # ------------
    callbacks = [
        EarlyStopping(monitor='val_loss',),
        ModelCheckpoint(monitor='val_loss', save_top_k=5),
        LearningRateMonitor(logging_interval='epoch', log_momentum=True),
        # StochasticWeightAveraging()
    ]
    plugins = [
        DDPPlugin(find_unused_parameters=False),
    ]
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, plugins=plugins)
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    result = trainer.test(test_dataloaders=test_loader)
    print(result)


if __name__ == '__main__':
    cli_main()
