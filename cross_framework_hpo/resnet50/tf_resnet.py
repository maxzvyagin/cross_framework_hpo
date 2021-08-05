import tensorflow as tf
from cross_framework_hpo.base_tensorflow_model import base_tensorflow_function

def resnet_tf_objective(config):
    model = tf.keras.applications.resnet50.ResNet50(weights=None, input_shape=(3, 32, 32), classes=100)
    return base_tensorflow_function(config=config, model=model)

if __name__ == "__main__":
    test_config = {'batch_size': 50, 'learning_rate': .001, 'epochs': 10, 'adam_epsilon': 10**-9}
    res = resnet_tf_objective(test_config)