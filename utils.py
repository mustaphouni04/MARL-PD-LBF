import matplotlib.pyplot as plt
from itertools import product
import numpy as np

FIG_WIDTH=5
FIG_HEIGHT=2
FIG_ALPHA=0.2
FIG_WSPACE=0.3
FIG_HSPACE=0.2


def visualise_q_tables(q_tables):
    for i, q_table in enumerate(q_tables):
        print(f"Q-table for Agent {i + 1}:")
        for a in range(2):
            print(f"Q({a + 1}) = {q_table[str((0, a))]:.2f}")
        print()


def visualise_q_tables_cql(q_table):
    print("Actual keys:")
    for k in list(q_table.keys())[:5]:
        print(repr(k))

    print("Centralized Q-table (obs=0):")
    for j, (a1, a2) in enumerate(product([0,1], repeat=2)):
        print(f"Q((0), {a1}, {a2}) = {q_table[str((0, a1, a2))]:.2f}") 


def visualise_evaluation_returns(means, stds):
    """
    Plot evaluation returns

    :param means (List[List[float]]): mean evaluation returns for each agent
    :param stds (List[List[float]]): standard deviation of evaluation returns for each agent
    """
    n_agents = len(means[0])
    n_evals = len(means)

    fig, ax = plt.subplots(nrows=1, ncols=n_agents, figsize=(FIG_WIDTH, FIG_HEIGHT * n_agents))

    colors = ["b", "r"]
    for i, color in enumerate(colors):
        ax[i].plot(range(n_evals), [mean[i] for mean in means], label=f"Agent {i+1}", color=color)
        ax[i].fill_between(range(n_evals), [mean[i] - std[i] for mean, std in zip(means, stds)],
                           [mean[i] + std[i] for mean, std in zip(means, stds)], alpha=FIG_ALPHA, color=color)
        ax[i].set_xlabel("Evaluations")
        ax[i].set_ylabel("Evaluation return")
    fig.legend()
    fig.subplots_adjust(hspace=FIG_HSPACE)

    plt.show()

def visualise_q_convergence(eval_q_tables, env, savefig=None):
    """
    Plot q_table convergence
    :param eval_q_tables (List[List[Dict[Act, float]]]): q_tables of both agents for each evaluation
    :param env (gym.Env): gym matrix environment with `payoff` attribute
    :param savefig (str): path to save figure
    """
    assert hasattr(env, "payoff")
    payoff = np.array(env.payoff)
    n_actions = 2
    n_agents = 2
    assert payoff.shape == (n_actions, n_actions, n_agents), "Payoff matrix must be 2x2x2 for 2x2 PD game"
    # (n_evals, n_agents, n_actions)
    q_tables = np.array(
            [[[q_table[str((0, act))] for act in range(n_actions)] for q_table in q_tables] for q_tables in eval_q_tables]
    )

    fig, ax = plt.subplots(nrows=n_agents, ncols=n_actions, figsize=(n_actions * FIG_WIDTH, FIG_HEIGHT * n_agents))

    for i in range(n_agents):
        max_payoff = payoff[:, :, i].max()
        min_payoff = payoff[:, :, i].min()
    
        for act in range(n_actions):
            # plot max Q-values
            if i == 0:
                max_r = payoff[act, :, i].max()
                max_label = rf"$max_b Q(a, b)$"
                q_label = rf"$Q(a_{act}, \cdot)$"
            else:
                max_r = payoff[:, act, i].max()
                max_label = rf"$max_a Q(a, b_{act})$"
                q_label = rf"$Q(\cdot, b_{act})$"
            ax[i, act].axhline(max_r, ls='--', color='r', alpha=0.5, label=max_label)

            # plot respective Q-values
            q_values = q_tables[:, i, act]
            ax[i, act].plot(q_values, label=q_label)

            # axes labels and limits
            ax[i, act].set_ylim([min_payoff - 0.05, max_payoff + 0.05])
            ax[i, act].set_xlabel(f"Evaluations")
            if i == 0:
                ax[i, act].set_ylabel(fr"$Q(a_{act})$")
            else:
                ax[i, act].set_ylabel(fr"$Q(b_{act})$")

            ax[i, act].legend(loc="upper center")

    fig.subplots_adjust(wspace=FIG_WSPACE)

    if savefig is not None:
        plt.savefig(f"{savefig}.pdf", format="pdf")

    plt.show()

def visualise_q_convergence_cql(eval_q_tables, env, savefig=None):
    """
    Plot all 4 joint Q-values convergence for CQL
    :param eval_q_tables (List[DefaultDict]): list of single q_tables for each evaluation
    :param env (gym.Env): gym matrix environment with `payoff` attribute
    :param savefig (str): path to save figure
    """
    assert hasattr(env, "payoff")
    payoff = np.array(env.payoff)
    n_actions = 2
    n_agents = 2
    
    n_evals = len(eval_q_tables)
    
    # Extract Q-values for each joint action: (n_evals, 4)
    joint_qs = np.zeros((n_evals, 4))
    permutation = list(product([0, 1], repeat=2))
    
    for eval_idx, q_table in enumerate(eval_q_tables):
        for joint_idx, (a1, a2) in enumerate(permutation):
            joint_qs[eval_idx, joint_idx] = q_table[str((0, a1, a2))]
    
    # 1x4 plot for all joint actions
    fig, ax = plt.subplots(1, 4, figsize=(4*FIG_WIDTH, FIG_HEIGHT))
    joint_names = ["(0,0)", "(0,1)", "(1,0)", "(1,1)"]
    
    min_payoff = payoff.min()
    max_payoff = payoff.max()
    
    for j in range(4):
        a1, a2 = permutation[j]
        # Plot corresponding payoff lines for reference
        payoff_a1_a2 = payoff[a1, a2, 0]  # Agent 1 payoff for this joint action
        ax[j].axhline(payoff_a1_a2, ls='--', color='r', alpha=0.7, label=f"R1={payoff_a1_a2}")
        ax[j].axhline(payoff[a1, a2, 1], ls='--', color='b', alpha=0.7, label=f"R2={payoff[a1, a2, 1]}")
        
        ax[j].plot(joint_qs[:, j], 'k-', linewidth=2, label=f"Q({a1},{a2})")
        
        ax[j].set_ylim([min_payoff - 0.5, max_payoff + 0.5])
        ax[j].set_title(f"Q({a1},{a2})")
        ax[j].set_xlabel("Evaluations")
        ax[j].legend()
    
    fig.suptitle("CQL: All Joint Action Q-values Convergence")
    plt.tight_layout()
    
    if savefig is not None:
        plt.savefig(f"{savefig}.pdf", format="pdf")
    plt.show()
