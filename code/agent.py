import numpy as np
from network import DQN_network
from collections import deque
from config import config
import random
from PIL import Image

class DQN_agent:

    def __init__(self, output_n):
        self.network = DQN_network(output_n)
        self.target_network = DQN_network(output_n)
        self.experience_memory = deque(maxlen=config.mem_length)
        self.output_n = output_n
        self.epsilon = config.epsilon

    def add_experience(self, curr_state, action, reward, next_state, game_complete):
        e = (curr_state, action, reward, next_state, game_complete)
        self.experience_memory.append(e)

    def next_action(self, state):
        r = np.random.rand()
        if r <= config.epsilon:
            return random.randrange(self.output_n)
        else:
            return self.network.predict_action(state)

    def sample_batch(self):
        return random.sample(self.experience_memory, config.minibatch_size)

    def update_epsilon(self):
        if self.epsilon > config.epsilon_min:
            self.epsilon -= config.epsilon_decay_step

    def train(self):
        if len(self.experience_memory) < config.train_start:
            return

        training_batch = self.sample_batch()
        for tuple in training_batch:
            (curr_state, action, reward, next_state, game_complete) = tuple

            true_target = self.target_network.output(curr_state)

            if game_complete:
                true_target[0][action] = reward
            else:
                true_target[0][action] = (reward + config.discount_factor * self.target_network.predict_action(next_state))

            self.network.train(curr_state, true_target)
        self.update_epsilon()

    def update_target(self):
        self.target_network.set_weights(self.network.get_weights())

    def load(self, filename):
        self.network.load_weights(filename)

    def save(self, filename):
        self.network.save_weights(filename)

    def preprocess_game_image(self, image):
        image = Image.fromarray(image, 'RGB')
        image = image.convert('L')
        image = image.resize(config.image_size)
        im_size = (image.size[1], image.size[0])
        converted_image = np.asarray(image.getdata(), dtype=np.uint8).reshape(im_size)
        converted_image = np.float32(converted_image / 255.0)
        return converted_image

    def construct_new_state(self, curr_state, new_observation):
        last_three = curr_state[1:]
        new_state = np.append(last_three, [new_observation], axis=0)
        return new_state
