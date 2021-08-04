from cross_framework_hpo.base_pytorch_model import BasePytorchModel
import torchvision.models as models
import pytorch_lightning as pl
import torch


def mnist_pt_objective(config):
    torch.manual_seed(0)
    model_class = BasePytorchModel(config)
    model_class.model = models.resnet50(pretrained=False)
    model_class.model.train()
    try:
        trainer = pl.Trainer(max_epochs=config['epochs'], gpus=[0])
    except:
        print("WARNING: training on CPU only, GPU[0] not found.")
        trainer = pl.Trainer(max_epochs=config['epochs'])
    trainer.fit(model_class)
    trainer.test(model_class)
    return (model_class.test_accuracy, model_class.model, model_class.avg_training_loss_history, model_class.latest_training_loss_history)


if __name__ == "__main__":
    test_config = {'batch_size': 64, 'learning_rate': .001, 'epochs': 1, 'dropout': 0.5, 'adam_epsilon': 1e-7}
    # test_config = {'batch_size': 800, 'learning_rate': 0.0001955, 'epochs': 15, 'dropout': 0.8869, 'adam_epsilon': 0.1182}
    res = mnist_pt_objective(test_config)
