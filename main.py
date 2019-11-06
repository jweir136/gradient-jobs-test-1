import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.datasets.mnist import load_data

class DataManager:
    def __init__(self):
        (self.train_x, self.train_y), (self.test_x, self.test_y) = load_data()
        self.train_x = np.divide(self.train_x, 255.)
        self.test_x = np.divide(self.test_x, 255.)
        self.train_y = keras.utils.to_categorical(self.train_y)
        self.test_y = keras.utils.to_categorical(self.test_y)
        self.train_x = self.train_x.reshape(-1, 28, 28, 1)
        self.test_x = self.test_x.reshape(-1, 28, 28, 1)

class MyModel:
    def create_model(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(32, activation='relu'))
        model.add(keras.layers.Dense(10, activation='softmax'))
        return model

if __name__ == "__main__":
    train_x, train_y = DataManager().train_x, DataManager().train_y
    test_x, test_y = DataManager().test_x, DataManager().test_y
    model = MyModel().create_model()
    model.compile(optimizer='adam', metrics=['acc'], loss='categorical_crossentropy')
    model.fit(train_x, train_y, epochs=10, batch_size=2056, validation_data=[test_x, test_y])
