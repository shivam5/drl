from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
import numpy as np
from config import config
import keras

class DQN_network:

    def __init__ (self, output_n):
        self.model = Sequential()
        self.model.add(Conv2D(32, 8, strides=(4, 4), padding='valid', activation='relu', input_shape=config.network_input_shape, data_format='channels_first'))
        self.model.add(Conv2D(filters=64, kernel_size=4, strides=(2, 2), padding='valid', activation='relu', data_format='channels_first'))
        self.model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='valid', activation='relu', data_format='channels_first'))
        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(output_n))
        rms = keras.optimizers.RMSprop(lr=config.learning_rate, epsilon=0.01, decay=0.0)
        self.model.compile(loss='mean_squared_error', optimizer=rms, metrics=['accuracy'])

    def predict_action(self, state):
        state = np.reshape(state, config.network_batch_shape)
        return np.argmax(self.model.predict(state))

    def predict(self, state):
        state = np.reshape(state, config.network_batch_shape)
        return self.model.predict(state)

    def output(self, state):
        state = np.reshape(state, config.network_batch_shape)
        return self.model.predict(state)

    def train(self, state, true_target):
        state = np.reshape(state, config.network_batch_shape)
        self.model.fit(state, true_target, epochs=1, verbose=0)

    def save_weights(self, file_name):
        self.model.save_weights(file_name)

    def load_weights(self, file_name):
        self.model.load_weights(file_name)

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)
