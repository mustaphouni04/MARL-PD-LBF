"""
This module implements the Independent Q-Learning (IQL) algorithm.

IQL is a multi-agent reinforcement learning algorithm where each agent
independently learns its own Q-table, treating other agents as part of the
environment.
"""
from collections import defaultdict
import random
from typing import List, DefaultDict

import numpy as np
from gymnasium.spaces import Space
from gymnasium.spaces.utils import flatdim


class IQL:
    """
    Agent using the Independent Q-Learning algorithm
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
        Constructor of IQL

        Initializes variables for independent Q-learning agents

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

        # initialise Q-tables for all agents
        # access value of Q_i(o, a) with self.q_tables[i][str((o, a))] (str conversion for hashable obs)
        self.q_tables: List[DefaultDict] = [
            defaultdict(lambda: 0) for _ in range(self.num_agents)
        ]

    def act(self, obss) -> List[int]:
        """
        Implement the epsilon-greedy action selection here for stateless task

        :param obss (List): list of observations for each agent
        :return (List[int]): index of selected action for each agent
        """
        actions = []
        for i, agent_obs in enumerate(obss):
            # look up Q value for that obs
            best = float('-inf')
            best_action = None
            
            n_acts_i = self.n_acts[i]

            for a in range(n_acts_i):
                q_val = self.q_tables[i][str((agent_obs,a))]

                if q_val > best:
                    best = q_val
                    best_action = a

            # epsilon greedy selection
            probs = (np.ones(n_acts_i) * self.epsilon / n_acts_i)
            probs[best_action] += 1.0 - self.epsilon

            draw = np.random.choice(np.arange(len(probs)), p=probs)

            actions.append(draw)
        
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

        :param obss (List[np.ndarray]): list of observations for each agent
        :param action (List[int]): index of applied action of each agent
        :param rewards (List[float]): received reward for each agent
        :param n_obss (List[np.ndarray]): list of observations after taking the action for each agent
        :param done (bool): flag indicating whether a terminal state has been reached
        :return (List[float]): updated Q-values for current actions of each agent
        """

        q_values = []
        
        for i, obs in enumerate(obss):
            next_state_i, reward_i, action_i = n_obss[i], rewards[i], actions[i]
            
            if done:
                td_target = reward_i
            else:
                td_target = reward_i + self.gamma * max([self.q_tables[i][str((next_state_i, a))] 
                                                         for a in range(self.n_acts[i])])

            td_error = td_target - self.q_tables[i][str((obs, action_i))]

            self.q_tables[i][str((obs, action_i))] += self.learning_rate * td_error

            q_values.append(self.q_tables[i][str((obs, action_i))])

        return q_values

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """
        Updates the hyperparameters

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        self.epsilon = 1.0 - (min(1.0, timestep / (0.8 * max_timestep))) * 0.99
