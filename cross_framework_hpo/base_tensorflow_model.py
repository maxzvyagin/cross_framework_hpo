import tensorflow as tf


def base_tensorflow_function(config, model, seed):
    tf.random.set_seed(seed)
    cifar = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_val = x_train[49000:]
    y_val = y_train[49000:]
    x_train = x_train[:49000]
    y_train = y_train[:49000]
    opt = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'], epsilon=config['adam_epsilon'])

    model.compile(optimizer=opt,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    res = model.fit(x_train, y_train, epochs=config['epochs'], batch_size=int(config['batch_size']), shuffle=False,
                    validation_data=(x_val, y_val), callbacks=[callback])
    training_loss_history = res.history['loss']
    res_test = model.evaluate(x_test, y_test)
    test_accuracy = res_test[1]
    return test_accuracy, model, training_loss_history, len(training_loss_history)
