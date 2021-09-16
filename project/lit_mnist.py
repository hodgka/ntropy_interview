'''

'''
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt
import torch
# import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
import pickle

mnist_data = MNIST('.', download=True, transform=ToTensor())
dataloader = DataLoader(mnist_data, shuffle=True, batch_size=60000)
X, y = next(iter(dataloader))
X = X.detach().cpu().numpy()
y = y.detach().cpu().numpy()

# read training data
X_train, X_test, y_train, y_test = X[:50000], X[50000:], y[:50000], y[50000:]
X_train = X_train.reshape(50000, 1, 28, 28)
X_test = X_test.reshape(10000, 1, 28, 28)

# assemble initial data
n_initial = 0
initial_idx = np.random.choice(range(len(X_train)), size=n_initial, replace=False)
X_initial = X_train[initial_idx]
y_initial = y_train[initial_idx]

# generate the pool
# remove the initial data from the training dataset
X_pool = np.delete(X_train, initial_idx, axis=0)[:5000]
y_pool = np.delete(y_train, initial_idx, axis=0)[:5000]


from modAL.models import ActiveLearner
from skorch import NeuralNetClassifier



class Classifier(torch.nn.Module):
    def __init__(self, hidden_dim=100):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(28 * 28, self.hidden_dim),
            torch.nn.Linear(self.hidden_dim, 10),
        ])
    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = torch.relu(layer(x))
        # x = torch.relu(self.l1(x))
        # x = torch.relu(self.l2(x))

        return x


device = "cuda" if torch.cuda.is_available() else "cpu"
classifier = NeuralNetClassifier(Classifier,
                                 criterion=torch.nn.CrossEntropyLoss,
                                 optimizer=torch.optim.Adam,
                                 train_split=None,
                                 verbose=1,
                                 device=device)
# classifier = Classifier()
# initialize ActiveLearner
learner = ActiveLearner(
    estimator=classifier,
    X_training=X_initial, y_training=y_initial,
)

# the active learning loop
n_queries = 10
for idx in range(n_queries):
    print('Query no. %d' % (idx + 1))
    query_idx, query_instance = learner.query(X_pool, n_instances=100)
    learner.teach(
        X=X_pool[query_idx], y=y_pool[query_idx], only_new=True,
    )
    # remove queried instance from pool
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx, axis=0)

    # print(dir(learner))
    estimator = learner.estimator
    estimator.save_params(f_params=f'weights/{idx}.pkl')

def eval












# from torch.nn import functional as F
# from torch.utils.data import DataLoader, random_split

# from torchvision.datasets.mnist import MNIST
# from torchvision import transforms


# class Classifier(pl.LightningModule):
#     def __init__(self, hidden_dim=128, learning_rate=1e-3):
#         super().__init__()
#         self.save_hyperparameters()

#         self.l1 = torch.nn.Linear(28 * 28, self.hparams.hidden_dim)
#         self.l2 = torch.nn.Linear(self.hparams.hidden_dim, 10)

    # def forward(self, x):
    #     x = x.view(x.size(0), -1)
    #     x = torch.relu(self.l1(x))
    #     x = torch.relu(self.l2(x))
    #     return x

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self(x)
#         loss = F.cross_entropy(y_hat, y)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self(x)
#         loss = F.cross_entropy(y_hat, y)
#         self.log('valid_loss', loss)

#     def test_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self(x)
#         loss = F.cross_entropy(y_hat, y)
#         self.log('test_loss', loss)

#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

#     @staticmethod
#     def add_model_specific_args(parent_parser):
#         parser = ArgumentParser(parents=[parent_parser], add_help=False)
#         parser.add_argument('--hidden_dim', type=int, default=128)
#         parser.add_argument('--learning_rate', type=float, default=0.0001)
#         return parser


# def cli_main():
#     pl.seed_everything(1234)

#     # ------------
#     # args
#     # ------------
#     parser = ArgumentParser()
#     parser.add_argument('--batch_size', default=32, type=int)
#     parser = pl.Trainer.add_argparse_args(parser)
#     parser = LitClassifier.add_model_specific_args(parser)
#     args = parser.parse_args()

#     # ------------
#     # data
#     # ------------
#     dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
#     mnist_test = MNIST('', train=False, download=True, transform=transforms.ToTensor())
#     mnist_train, mnist_val = random_split(dataset, [55000, 5000])

#     train_loader = DataLoader(mnist_train, batch_size=args.batch_size)
#     val_loader = DataLoader(mnist_val, batch_size=args.batch_size)
#     test_loader = DataLoader(mnist_test, batch_size=args.batch_size)

#     # ------------
#     # model
#     # ------------
#     model = LitClassifier(args.hidden_dim, args.learning_rate)

#     # ------------
#     # training
#     # ------------
#     trainer = pl.Trainer.from_argparse_args(args)
#     trainer.fit(model, train_loader, val_loader)

#     # ------------
#     # testing
#     # ------------
#     trainer.test(test_dataloaders=test_loader)


# if __name__ == '__main__':
#     cli_main()
