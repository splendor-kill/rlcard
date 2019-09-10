"""
A toy example of learning a Deep-Q Agent on Blackjack
"""

import rlcard
from rlcard.agents.deep_cfr import DeepCFR
from rlcard.utils.utils import *
import tensorflow as tf
import numpy as np

# make environment
set_global_seed(0)
evaluate_every = 100
evaluate_num = 1000
num_iteration = 1000
i = 0
rewards = 0
train_env = rlcard.make('blackjack') 
test_env = rlcard.make('blackjack') 
with tf.Session() as sess:
    deep_cfr = DeepCFR(sess, #
                train_env, 
                policy_network_layers=(32,32),
                advantage_network_layers=(32,32),
                num_traversals=20,
                learning_rate=1e-4,
                batch_size_advantage=None,
                batch_size_strategy=16,
                memory_capacity=1e7)

    for i in range(num_iteration):
        # Train the agent in training environment
        _, adv_loss, policy_loss = deep_cfr.train()

        # Evaluate the agent
        if i % evaluate_every == 0:
            rewards = 0
            for j in range(evaluate_num):
                state, player = test_env.init_game()
                while True:
                    action_prob = deep_cfr.action_probabilities(state)
                    action_prob /= action_prob.sum()
                    action = np.random.choice(np.arange(len(action_prob)), p=action_prob)
                    #action_prob = list(action_prob)
                    #action = action_prob.index(max(action_prob))
                    #print("Play:", state, action)
                    state, player = test_env.step(action)
                    if test_env.is_over():
                        payoffs = test_env.get_payoffs()
                        rewards += payoffs[0]
                        break
            print('############## Iteration '+str(i)+' #################')
            print('Reward: ', float(rewards)/evaluate_num)
            print('Advantage Loss: ', adv_loss)
            print('Policy Loss: ', policy_loss)
