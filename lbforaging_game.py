"""
This script implements and compares two multi-agent reinforcement learning algorithms,
Independent Q-Learning (IQL) and Conservative Q-Learning (CQL), on the Level-Based
Foraging (LBF) environment.

The script trains both algorithms sequentially and generates plots to visualize their
performance, including convergence of returns and evolution of Q-values. It also
records a video of the learned policies.
"""
import numpy as np
import gymnasium as gym
import imageio

from lbforaging.foraging.rendering import Viewer
import matplotlib.pyplot as plt
from itertools import product
from collections import defaultdict
from typing import List, DefaultDict
from enum import Enum
import re

class Action(Enum):
    """
    An enumeration of the possible actions in the Level-Based Foraging environment.
    """
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4
    LOAD = 5

def plot_convergence(returns: list, algorithm: str):
    """
    Plots the convergence of the average return per episode.

    :param returns: A list of average returns per episode.
    :param algorithm: The name of the algorithm being plotted.
    """
    plt.figure(figsize=(8,5))
    plt.plot(returns, label=f"{algorithm} Return")
    plt.xlabel("Episode Round")
    plt.ylabel("Average Return")
    plt.title(f"{algorithm} Convergence")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_q_value_evolution(q_value_history, eval_steps, algorithm, k=5):
    """
    Plots the evolution of the top-k Q-values over training episodes.

    :param q_value_history: A list of lists, where each inner list contains the top-k Q-values at a given evaluation step.
    :param eval_steps: A list of the episode numbers at which the Q-values were evaluated.
    :param algorithm: The name of the algorithm being plotted.
    :param k: The number of top Q-values to plot.
    """
    q_arr = np.array(q_value_history)  # shape: [T, K]

    plt.figure(figsize=(9, 6))
    for i in range(k):
        plt.plot(eval_steps, q_arr[:, i], label=f"Top-{i+1} Q")

    plt.xlabel("Episode")
    plt.ylabel("Q value")
    plt.title(f"{algorithm}: Top-{k} Q-value evolution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def get_topk_q_values(agent, k=5):
    """
    Gets the top-k Q-values from the agent's Q-table.

    :param agent: The agent object.
    :param k: The number of top Q-values to retrieve.
    :return: A list of the top-k Q-values.
    """

    if agent.algorithm == "IQL":
        # flatten all agent Q-tables
        all_qs = []
        for qtab in agent.q_table:
            all_qs.extend(qtab.values())
    else:  # CQL
        all_qs = list(agent.q_table.values())

    if len(all_qs) == 0:
        return [0.0] * k

    topk = sorted(all_qs, reverse=True)[:k]

    # pad if not enough entries yet
    if len(topk) < k:
        topk += [topk[-1]] * (k - len(topk))

    return topk


def iql_act(obss, q_table, n_acts, epsilon) -> List[int]:
    """
    Selects actions for each agent using an epsilon-greedy policy based on the IQL algorithm.

    :param obss: A list of observations for each agent.
    :param q_table: The Q-table for the IQL agent.
    :param n_acts: A list of the number of actions for each agent.
    :param epsilon: The exploration rate.
    :return: A list of the selected actions for each agent.
    """
    actions = []
    for i, agent_obs in enumerate(obss):
        # look up Q value for that obs
        best = float('-inf')
        best_action = None
        
        for a in range(n_acts[0]):
            state_action = str(tuple(agent_obs) + (a,))
            q_val = q_table[i][state_action]

            if q_val > best:
                best = q_val
                best_action = a

        # epsilon greedy selection
        probs = (np.ones(n_acts[0]) * epsilon / n_acts[0])
        probs[best_action] += 1.0 - epsilon

        draw = np.random.choice(np.arange(len(probs)), p=probs)

        actions.append(draw)
    
    return actions

def iql_learn(
        obss: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        n_obss: List[np.ndarray],
        done: bool,
        gamma: float,
        learning_rate: float,
        q_table: DefaultDict,
        n_acts: List
        ):
    """
    Updates the Q-table for the IQL agent based on the agent's experience.

    :param obss: A list of observations for each agent.
    :param actions: A list of the actions taken by each agent.
    :param rewards: A list of the rewards received by each agent.
    :param n_obss: A list of the next observations for each agent.
    :param done: A boolean indicating whether the episode has ended.
    :param gamma: The discount factor.
    :param learning_rate: The learning rate.
    :param q_table: The Q-table for the IQL agent.
    :param n_acts: A list of the number of actions for each agent.
    :return: A list of the updated Q-values for the current state-action pairs.
    """

    q_values = []

    for i, obs in enumerate(obss):
        next_state_i, reward_i, action_i = n_obss[i], rewards[i], actions[i]
        
        if done:
            td_target = reward_i
        else:
            td_target = reward_i + gamma * max([q_table[i][str((tuple(next_state_i) + (a,)))] 
                                                     for a in range(n_acts[i])])

        td_error = td_target - q_table[i][str(tuple(obs) + (action_i,))]

        q_table[i][str(tuple(obs) + (action_i,))] += learning_rate * td_error

        q_values.append(q_table[i][str(tuple(obs) + (action_i,))])

    return q_values

def cql_act(obss, q_table, epsilon, permutation):
    """
    Selects actions for each agent using an epsilon-greedy policy based on the CQL algorithm.

    :param obss: A list of observations for each agent.
    :param q_table: The Q-table for the CQL agent.
    :param epsilon: The exploration rate.
    :param permutation: A list of all possible joint actions.
    :return: A list of the selected actions for each agent.
    """
    actions = []

    best = float('-inf')
    best_idx = None

    for j, (a1, a2) in enumerate(permutation):
        state_action = str(tuple(obss[0]) + tuple(obss[1]) + (a1,a2))
        q_val = q_table[state_action]

        if q_val > best:
            best = q_val 
            best_idx = j

    probs = (np.ones(len(permutation)) * epsilon / len(permutation))
    probs[best_idx] += 1.0 - epsilon

    draw = np.random.choice(np.arange(len(probs)), p=probs)

    sampled_actions = permutation[draw]
    actions = list(sampled_actions)
        
    return actions

def cql_learn(
        obss: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        n_obss: List[np.ndarray],
        done: bool,
        gamma: float,
        learning_rate: float,
        q_table: DefaultDict,
        permutation: List[List],
        ):
    """
    Updates the Q-table for the CQL agent based on the agent's experience.

    :param obss: A list of observations for each agent.
    :param actions: A list of the actions taken by each agent.
    :param rewards: A list of the rewards received by each agent.
    :param n_obss: A list of the next observations for each agent.
    :param done: A boolean indicating whether the episode has ended.
    :param gamma: The discount factor.
    :param learning_rate: The learning rate.
    :param q_table: The Q-table for the CQL agent.
    :param permutation: A list of all possible joint actions.
    :return: A list of the updated Q-values for the current state-action pair.
    """

    a1, a2 = actions
    
    total_reward = np.mean(rewards) 
    next_state = n_obss

    if done:
        td_target = total_reward
    else:
        state_tuple = tuple(next_state[0]) + tuple(next_state[1])
        td_target = total_reward + gamma * max([
                                            q_table[str(state_tuple + (action1, action2))]
                                                     for action1, action2 in permutation])

    state_action = tuple(obss[0]) + tuple(obss[1]) + (a1, a2)
    td_error = td_target - q_table[str(state_action)]

    q_table[str(state_action)] += learning_rate * td_error

    # match expected signature
    return [q_table[str(state_action)]] * 2


class LearningModule:
    """
    A wrapper class for the learning algorithms.
    """

    def __init__(self, env: gym.Env,
                 algorithm: str,
                 learning_rate: float,
                 gamma: float,
                 epsilon: float) -> None:
        """
        Initializes the LearningModule.

        :param env: The environment.
        :param algorithm: The name of the algorithm to use ('IQL' or 'CQL').
        :param learning_rate: The learning rate.
        :param gamma: The discount factor.
        :param epsilon: The exploration rate.
        """

        assert algorithm in ["IQL", "CQL"], "Please provide a valid algorithm name: either 'IQL' or 'CQL'"
        self.env = env
        self.algorithm = algorithm

        env_name: str = env.unwrapped.spec.id

        # https://www.reddit.com/r/learnpython/comments/7ood1y/how_to_extract_numbers_within_a_string/
        parsed = re.findall("(\d+)", env_name)
        self.env_name = env_name
        self.num_agents, self.food_locations = int(parsed[2:-1][0]), int(parsed[2:-1][1])

        if algorithm == "IQL":
            q_table = [
                defaultdict(lambda: 0.0) for _ in range(self.num_agents)
            ]
        elif algorithm == "CQL":
            q_table = defaultdict(lambda: (0.0))

        else:
            q_table = None

        self.q_table = q_table

        self.gamma: float = gamma 
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        
        print(type(self.num_agents))
        self.n_acts = [len(Action)] * self.num_agents

        self.permutation = list(product(range(self.n_acts[0]), repeat=2))
        
    def act(self, obss) -> List[int]:
        """
        Selects actions for each agent based on the current observations and the selected algorithm.

        :param obss: A list of observations for each agent.
        :return: A list of the selected actions for each agent.
        """
        if self.algorithm == "IQL":
            return iql_act(obss, self.q_table, self.n_acts, self.epsilon)
        elif self.algorithm == "CQL":
            return cql_act(obss, self.q_table, self.epsilon, self.permutation)
        else:
            raise ValueError

    def learn(self,
        obss: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        n_obss: List[np.ndarray],
        done: bool):
        """
        Updates the Q-table based on the agent's experience.

        :param obss: A list of observations for each agent.
        :param actions: A list of the actions taken by each agent.
        :param rewards: A list of the rewards received by each agent.
        :param n_obss: A list of the next observations for each agent.
        :param done: A boolean indicating whether the episode has ended.
        """
        
        if self.algorithm == "IQL":
            return iql_learn(obss, actions, rewards, n_obss, done, 
                             self.gamma, self.learning_rate, self.q_table, self.n_acts)
        elif self.algorithm == "CQL":
            return cql_learn(obss, actions, rewards, n_obss, done, 
                             self.gamma, self.learning_rate, self.q_table, self.permutation)
        else:
            raise ValueError
                           
    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """
        Schedules the hyperparameters for the learning algorithm.

        :param timestep: The current timestep.
        :param max_timestep: The maximum number of timesteps.
        """
        self.epsilon = 1.0 - (min(1.0, timestep / (0.8 * max_timestep))) * 0.99 

def record_video_with_viewer(agent, filename: str, episodes: int = 3):
    """Record video using the 2D Viewer from lbforaging."""
    env = agent.env.unwrapped  
    world_size = (env.rows, env.cols)
    viewer = Viewer(world_size)
    frames = []

    for ep in range(episodes):
        obss, _ = env.reset()
        done = False
        while not done:
            frame = viewer.render(env, return_rgb_array=True)  # pass raw env
            frames.append(frame)
            actions = agent.act(obss)
            obss, rewards, done, truncated, info = env.step(actions)

    viewer.close()
    imageio.mimsave(filename, frames, fps=5)
    print(f"Video saved to {filename}")

def evaluate_policy(agent, env, n_eval_episodes=50, max_steps=50):
    """
    Evaluates the policy of the agent in the given environment.

    :param agent: The agent to evaluate.
    :param env: The environment to evaluate the agent in.
    :param n_eval_episodes: The number of episodes to evaluate the agent for.
    :param max_steps: The maximum number of steps per episode.
    :return: The average return over the evaluation episodes.
    """
    eval_returns = []

    for _ in range(n_eval_episodes):
        obss, _ = env.reset()
        done = False
        step = 0
        ep_return = 0.0

        while not done and step < max_steps:
            actions = agent.act(obss)
            obss, rewards, done, truncated, info = env.step(actions)
            ep_return += np.mean(rewards)
            step += 1

        eval_returns.append(ep_return)

    return np.mean(eval_returns)

def demo():
    """
    Runs a demonstration of the IQL and CQL algorithms on the Level-Based Foraging environment.
    """
    TOP_K = 10
    EVAL_EVERY = 100  # episodes

    q_value_history = []  # list of [q1, q2, ..., qK]
    q_eval_steps = []

    env = gym.make("Foraging-5x5-2p-1f-coop-v3", render_mode="rgb_array")

    for algo in ["IQL", "CQL"]:
        agent = LearningModule(
            env=env,
            algorithm=algo,
            learning_rate=0.1,
            gamma=0.99,
            epsilon=1.0
        )

        num_episodes = 100000
        max_steps = 50
        mean_returns = []
        eval_steps = []

        for ep in range(num_episodes):
            obss, _ = env.reset()
            done = False
            step = 0
            ep_return = 0.0

            while not done and step < max_steps:
                actions = agent.act(obss)
                n_obss, rewards, done, truncated, info = env.step(actions)

                agent.learn(
                    obss=obss,
                    actions=actions,
                    rewards=rewards,
                    n_obss=n_obss,
                    done=done
                )

                obss = n_obss
                ep_return += np.mean(rewards)
                step += 1

            agent.schedule_hyperparameters(ep, num_episodes)

            if ep % EVAL_EVERY == 0:
                # --- policy tracking ---
                mean_ret = evaluate_policy(agent, env)
                mean_returns.append(mean_ret)
                eval_steps.append(ep)

                # --- Q-value tracking ---
                topk = get_topk_q_values(agent, k=TOP_K)
                q_value_history.append(topk)
                q_eval_steps.append(ep)

                print(f"[{algo}] Episode {ep} | Return: {ep_return:.2f}")

        #plt.plot(eval_steps, mean_returns)
        plot_convergence(mean_returns, algorithm=algo)
        plot_q_value_evolution(q_value_history, q_eval_steps, algo, k=TOP_K)
       
        record_video_with_viewer(agent, filename=f"{algo}_demo.gif", episodes=20)

    env.close()

if __name__ == "__main__":
    demo()



