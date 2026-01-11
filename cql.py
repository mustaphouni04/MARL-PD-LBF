from collections import defaultdict
import random
from typing import List, DefaultDict
from itertools import product 

import numpy as np
from gymnasium.spaces import Space
from gymnasium.spaces.utils import flatdim


class CQL:
    """
    Agent using the Centralized Q-Learning algorithm
    """
    def __init__(
        self,
        num_agents: int,
        action_spaces: List[Space],
        gamma: float,
        learning_rate: float = 0.5,
        epsilon: float = 1.0,
        **kwargs,
    ):
        """
        Constructor of CQL

        Initializes variables for centralized Q-learning agents

        :param num_agents (int): number of agents
        :param action_spaces (List[Space]): action spaces of the environment for each agent
        :param gamma (float): discount factor (gamma)
        :param learning_rate (float): learning rate for Q-learning updates
        :param epsilon (float): epsilon value for all agents

        :attr n_acts (List[int]): number of actions for each agent
        :attr q_tables (List[DefaultDict]): tables for Q-values mapping actions ACTs
            to respective Q-values for all agents
        """
        self.num_agents = num_agents
        self.action_spaces = action_spaces
        self.n_acts = [flatdim(action_space) for action_space in action_spaces]

        self.gamma: float = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        # [(0, 0), (0, 1), (1, 0), (1, 1)]
        # Source: https://www.geeksforgeeks.org/python/python-itertools-product/
        self.permutation = list(product([0, 1], repeat=2))

        # Single Q-Table with joint actions
        self.q_table: DefaultDict = defaultdict(lambda: (0.0)) 

    def act(self, obss) -> List[int]:
        """
        Implement the epsilon-greedy action selection here for stateless task

        **IMPLEMENT THIS FUNCTION**

        :param obss (List): List of observations for each agent
        :return (List[int]): index of selected action for each agent
        """
        actions = []

        # obss = [0,0] ALWAYS --> STATELESS PD, we use obs[0] to make this practical for the joint action

        best = float('-inf')
        best_idx = None

        for j, (a1, a2) in enumerate(self.permutation):
            q_val = self.q_table[str((obss[0], a1, a2))]

            if q_val > best:
                best = q_val 
                best_idx = j

        probs = (np.ones(len(self.permutation)) * self.epsilon / len(self.permutation))
        probs[best_idx] += 1.0 - self.epsilon

        draw = np.random.choice(np.arange(len(probs)), p=probs)

        sampled_actions = self.permutation[draw]

        actions = list(sampled_actions)
            
        return actions

    def learn(
        self,
        obss: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        n_obss: List[np.ndarray],
        done: bool,
    ):
        """
        Updates the Q-tables based on agents' experience

        **IMPLEMENT THIS FUNCTION**

        :param obss (List[np.ndarray]): list of observations for each agent
        :param action (List[int]): index of applied action of each agent
        :param rewards (List[float]): received reward for each agent
        :param n_obss (List[np.ndarray]): list of observations after taking the action for each agent
        :param done (bool): flag indicating whether a terminal state has been reached
        :return (List[float]): updated Q-values for current actions of each agent
        """

        a1, a2 = actions
        
        obs = obss[0] # stateless obs
        total_reward = np.mean(rewards) #rewards[0] + rewards[1]
        next_state = n_obss[0]

        if done:
            td_target = total_reward
        else:
            td_target = total_reward + self.gamma * max([self.q_table[str((next_state, action1, action2))]
                                                         for action1, action2 in self.permutation])

        td_error = td_target - self.q_table[str((obs, a1, a2))]

        self.q_table[str((obs, a1, a2))] += self.learning_rate * td_error

        # match expected signature
        return [self.q_table[str((obs, a1, a2))]] * 2

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """
        Updates the hyperparameters

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        self.epsilon = 1.0 - (min(1.0, timestep / (0.8 * max_timestep))) * 0.99
