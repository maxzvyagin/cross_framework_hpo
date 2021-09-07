import pdb

from torch import nn
import pytorch_lightning as pl
import torchvision
import torch
import statistics
import tensorflow as tf
import sys

class CIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        self.split = split
        cifar = tf.keras.datasets.cifar10
        (x_train, y_train), (x_test, y_test) = cifar.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        if split == "train":
            self.labels = x_train[:49000]
            self.targets = y_train[:49000]
        elif split == "val":
            self.labels = x_train[49000:]
            self.targets = y_train[49000:]
        elif split == "test":
            self.labels = x_test
            self.targets = y_test
        else:
            sys.exit("Invalid split argument provided.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.labels[i], self.targets[i]

class BasePytorchModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = None
        # this is meant to operate on logits
        self.criterion = nn.CrossEntropyLoss()
        self.config = config
        self.test_loss = None
        self.test_accuracy = None
        self.accuracy = pl.metrics.Accuracy()
        self.training_loss_history = []
        self.avg_training_loss_history = []
        self.latest_training_loss_history = []
        self.training_loss_history = []

    def train_dataloader(self):
        return torch.utils.data.DataLoader(CIFAR10Dataset(split="train"),
                                           batch_size=int(self.config['batch_size']), num_workers=0, shuffle=False)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(CIFAR10Dataset(split="train"),
                                           batch_size=int(self.config['batch_size']), num_workers=0, shuffle=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(CIFAR10Dataset(split="test"),
                                           batch_size=int(self.config['batch_size']), num_workers=0, shuffle=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['learning_rate'],
                                     eps=self.config['adam_epsilon'])
        return optimizer

    def forward(self, x):
        return self.model(x)

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        out = self.forward(x)
        loss = self.criterion(out, y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        out = self.forward(x)
        loss = self.criterion(out, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        out = self.forward(x)
        loss = self.criterion(out, y)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    # def training_step_end(self, outputs):
    #     loss = self.criterion(outputs['forward'], outputs['expected'])
    #     logs = {'train_loss': loss}
    #     # pdb.set_trace()
    #     return {'loss': loss, 'logs': logs}
    #
    # def training_epoch_end(self, outputs):
    #     # pdb.set_trace()
    #     loss = []
    #     for x in outputs:
    #         loss.append(float(x['loss']))
    #     avg_loss = statistics.mean(loss)
    #     # tensorboard_logs = {'train_loss': avg_loss}
    #     self.avg_training_loss_history.append(avg_loss)
    #     self.latest_training_loss_history.append(loss[-1])
    #     # return {'avg_train_loss': avg_loss, 'log': tensorboard_logs}

    # def test_step(self, test_batch, batch_idx):
    #     x, y = test_batch
    #     return {'forward': self.forward(x), 'expected': y}

    # def test_step_end(self, outputs):
    #     loss = self.criterion(outputs['forward'], outputs['expected'])
    #     accuracy = self.accuracy(outputs['forward'], outputs['expected'])
    #     logs = {'test_loss': loss, 'test_accuracy': accuracy}
    #     return {'test_loss': loss, 'logs': logs, 'test_accuracy': accuracy}
    #
    # def test_epoch_end(self, outputs):
    #     loss = []
    #     for x in outputs:
    #         loss.append(float(x['test_loss']))
    #     avg_loss = statistics.mean(loss)
    #     # tensorboard_logs = {'test_loss': avg_loss}
    #     self.test_loss = avg_loss
    #     accuracy = []
    #     for x in outputs:
    #         accuracy.append(float(x['test_accuracy']))
    #     avg_accuracy = statistics.mean(accuracy)
    #     self.test_accuracy = avg_accuracy

def base_pytorch_function(config, supplied_model, seed):
    torch.manual_seed(seed)
    model_class = BasePytorchModel(config)
    model_class.model = supplied_model
    model_class.model.train()
    try:
        trainer = pl.Trainer(max_epochs=config['epochs'], gpus=[0])
    except:
        print("WARNING: training on CPU only, GPU[0] not found.")
        trainer = pl.Trainer(max_epochs=config['epochs'])
    trainer.fit(model_class)
    trainer.test(model_class)
    pdb.set_trace()
    return model_class.test_accuracy, model_class.model, model_class.avg_training_loss_history