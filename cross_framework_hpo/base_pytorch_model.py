import pdb

from torch import nn
import pytorch_lightning as pl
import torchvision
import torch
import statistics
import tensorflow as tf
import sys
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class CIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        self.split = split
        cifar = tf.keras.datasets.cifar10
        (x_train, y_train), (x_test, y_test) = cifar.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        if split == "train":
            self.images = x_train[:49000]
            self.targets = y_train[:49000]
        elif split == "val":
            self.images = x_train[49000:]
            self.targets = y_train[49000:]
        elif split == "test":
            self.images = x_test
            self.targets = y_test
        else:
            sys.exit("Invalid split argument provided.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.images[i].astype("float32"), self.targets[i]


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
        self.validation_loss_history = []
        self.validation_acc_history = []

    def train_dataloader(self):
        return torch.utils.data.DataLoader(CIFAR10Dataset(split="train"),
                                           batch_size=int(self.config['batch_size']), num_workers=0, shuffle=False)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(CIFAR10Dataset(split="val"),
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

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        out = self.forward(x)
        loss = self.criterion(out, y.long().flatten())
        self.log("train_loss", loss.detach(), on_epoch=True, prog_bar=False, logger=True)
        return {"loss": loss, "logs": {"train_loss": loss.detach()}}

    def validation_step(self, val_batch, batch_idx):
        # pdb.set_trace()
        x, y = val_batch
        out = self.forward(x)
        y = y.long().flatten()
        loss = self.criterion(out, y)
        acc = self.accuracy(out, y)
        self.log("val_loss", loss.detach(), on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc.detach(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "val_acc": acc, "logs": {"val_loss": loss.detach(), 'val_acc': acc.detach()}}

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        out = self.forward(x)
        y = y.long().flatten()
        loss = self.criterion(out, y)
        acc = self.accuracy(out, y)
        self.log("test_loss", loss.detach(), on_epoch=True, prog_bar=True, logger=True)
        self.log("test_acc", acc.detach(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "test_acc": acc, "logs": {"test_loss": loss.detach(), 'test_acc': acc.detach()}}

    def training_epoch_end(self, outputs):
        loss = []
        for x in outputs:
            loss.append(float(x['loss']))
        avg_loss = statistics.mean(loss)
        self.training_loss_history.append(avg_loss)

    def validation_epoch_end(self, outputs):
        loss = []
        for x in outputs:
            loss.append(float(x['loss']))
        avg_loss = statistics.mean(loss)
        self.validation_loss_history.append(avg_loss)
        accuracy = []
        for x in outputs:
            accuracy.append(float(x['val_acc']))
        avg_accuracy = statistics.mean(accuracy)
        self.validation_acc_history.append(avg_accuracy)

    def test_epoch_end(self, outputs):
        loss = []
        for x in outputs:
            loss.append(float(x['loss']))
        avg_loss = statistics.mean(loss)
        self.test_loss = avg_loss
        accuracy = []
        for x in outputs:
            accuracy.append(float(x['test_acc']))
        avg_accuracy = statistics.mean(accuracy)
        self.test_accuracy = avg_accuracy


def base_pytorch_function(config, supplied_model, seed):
    torch.manual_seed(seed)
    model_class = BasePytorchModel(config)
    model_class.model = supplied_model
    model_class.model.train()
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=3)
    try:
        trainer = pl.Trainer(max_epochs=config['epochs'], gpus=[0], callbacks=[early_stop_callback])
    except:
        print("WARNING: training on CPU only, GPU[0] not found.")
        trainer = pl.Trainer(max_epochs=config['epochs'], callbacks=[early_stop_callback])
    trainer.fit(model_class)
    trainer.test(model_class)
    return model_class.test_accuracy, model_class.model, model_class.training_loss_history, \
           model_class.validation_loss_history, model_class.validation_acc_history, \
           len(model_class.training_loss_history)
