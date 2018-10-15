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

global_step = 0
average_q_max = 0

if config.load_weight_file is not None:
    agent.load(config.load_weight_file)


for episode_no in range(config.initial_episode, config.num_episodes):

    game_complete = False
    starting_lives = 5
    score = 0

    #Resetting the environment
    initial_observation = environment.reset()

    # Do not do anything for some number of steps
    for _ in range(random.randint(1, config.no_operation_steps)):
        observation, _, _, _ = environment.step(1)


    observation = agent.preprocess_game_image(observation)
    curr_state = np.array([observation, observation, observation, observation])

    frame_no = 0
    while ((frame_no <= config.max_time) and (not game_complete)):
        if config.see_video:
            environment.render()


        action = agent.next_action(curr_state)
        # real_action = action+1

        observation, reward, game_complete, info = environment.step(action)
        processed_obs = agent.preprocess_game_image(observation)
        next_state = agent.construct_new_state(curr_state, processed_obs)


        average_q_max += np.amax(agent.network.predict(curr_state))


        agent.add_experience(curr_state, action, np.clip(reward, -1, 1), next_state, game_complete)

        agent.train()

        if (frame_no%config.update_freq==0):
            agent.update_target()

        score += reward

        curr_state = next_state

        if game_complete:
            print("episode: {}/{}, score: {}, Memory length: {}, e: {:.3}, Global step: {}, Average q: {}"
                  .format(episode_no, config.num_episodes, score, len(agent.experience_memory), agent.epsilon, global_step,
                  average_q_max/float(frame_no)))
            average_q_max = 0
            break

        frame_no += 1
        global_step += 1

    # print("episode: {}/{}, score: {}, player score: {}, e: {:.2}"
    #       .format(episode_no, config.num_episodes, score, pos_score, agent.epsilon))
    # agent.train()

    if episode_no % 100 == 0:
        agent.save("weights/breakoutweight"+str(episode_no)+".h5")
