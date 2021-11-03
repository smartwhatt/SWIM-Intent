from keras.layers import Dense
from keras.models import Sequential
import keras
import tensorflow as tf

def create_model(training_shape:int, output_shape:int):
    model = Sequential([
        Dense(8, input_shape=(training_shape, ), activation="relu"),
        Dense(8, activation="relu"),
        Dense(output_shape, activation="softmax")
    ])

    loss = keras.losses.BinaryCrossentropy(from_logits=False)
    optim = tf.optimizers.Adam(lr=0.0083)
    metrics = ["accuracy"]

    # print("\nCompiling Model...\n")
    model.compile(loss=loss, optimizer=optim, metrics=metrics)
    return model

def train_model(model:Sequential, training, output, epoch:int = 50, filename="model.h5", save=False, callbacks=None):
    model.fit(training, output, epochs=epoch,  verbose=2, validation_split=0.2, callbacks=callbacks)
    if save:
        model.save(filename) 
    return model