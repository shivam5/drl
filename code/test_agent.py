import numpy as np
import gym
from config import config
from agent import DQN_agent
from network import DQN_network
import random

# Defining the game environment
environment = gym.make("BreakoutDeterministic-v4")

output_n = environment.action_space.n
output_shape = [1, output_n]

agent = DQN_agent(output_n)

if config.load_weight_file is None:
    print "Please specify weights file for loading"
    exit()

agent.load(config.load_weight_file)



game_complete = False
#Resetting the environment
observation = environment.reset()


observation = agent.preprocess_game_image(observation)
curr_state = np.array([observation, observation, observation, observation])

score = 0
while (not game_complete):
    environment.render()


    action = agent.next_action(curr_state)
    # real_action = action+1

    observation, reward, game_complete, info = environment.step(action)
    processed_obs = agent.preprocess_game_image(observation)
    next_state = agent.construct_new_state(curr_state, processed_obs)

    score += reward
    curr_state = next_state

    if game_complete:
        print("Score: {}".format(score))
