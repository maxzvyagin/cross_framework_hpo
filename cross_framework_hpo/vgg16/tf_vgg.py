import tensorflow as tf
from cross_framework_hpo.base_tensorflow_model import base_tensorflow_function

def vgg_tf_objective(config, seed):
    model = tf.keras.applications.vgg16.VGG16(weights=None, input_shape=(3, 32, 32), classes=10)
    return base_tensorflow_function(config=config, model=model, seed=seed)

if __name__ == "__main__":
    test_config = {'batch_size': 532, 'learning_rate': 0.074552791, 'epochs': 26, 'adam_epsilon': 0.536216016}
    tf_test_acc, tf_model, tf_training_history = vgg_tf_objective(test_config, seed=0)
    print("Accuracy is {}".format(tf_test_acc))

    tf_test_acc, tf_model, tf_training_history = vgg_tf_objective(test_config, seed=1234)
    print("Accuracy is {}".format(tf_test_acc))