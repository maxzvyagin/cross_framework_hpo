from cross_framework_hpo.base_pytorch_model import base_pytorch_function
import torchvision.models as models


def vgg_pt_objective(config):
    model = models.vgg16(pretrained=False)
    return base_pytorch_function(config, supplied_model=model)


if __name__ == "__main__":
    test_config = {'batch_size': 64, 'learning_rate': .001, 'epochs': 1, 'dropout': 0.5, 'adam_epsilon': 1e-7}
    # test_config = {'batch_size': 800, 'learning_rate': 0.0001955, 'epochs': 15, 'dropout': 0.8869, 'adam_epsilon': 0.1182}
    res = vgg_pt_objective(test_config)
    print("Accuracy is {}".format(res[0]))
