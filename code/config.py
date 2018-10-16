import numpy as np

class config:
    num_episodes = 50000
    initial_episode = 0

    max_time = np.inf
    update_freq = 10000
    mem_length = 400000
    train_start = 50000
    no_operation_steps = 30

    discount_factor = 0.99
    epsilon_min = 0.1
    initial_epsilon = 1.0
    epsilon_decay_step =  (initial_epsilon - epsilon_min) / 1000000.

    epsilon = max(1.0 - (epsilon_decay_step * initial_episode), epsilon_min)
    learning_rate = 0.00025
    minibatch_size = 32
    dropout_prob = 0.1

    # load_weight_file = "weights/breakoutweight"+str(initial_episode)+".h5"
    load_weight_file = None
    see_video = True
    network_input_shape = (4, 110, 84)
    network_batch_shape = (1, 4, 110, 84)
    image_size = (110, 84)
