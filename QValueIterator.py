import sys
from mdp import MDP
from collections import defaultdict


def max_qvalue(util_dict, state):
    return max(util_dict[state])


def qvalue_iteration(mdp_in, epsilon):

    # Setup Variables
    qval_list = defaultdict(dict)

    # Set initial utility for all (s, a) combinations to 0
    for state in mdp_in.state_reward_dict:
        for action in mdp_in.get_actions():
            qval_list[state][action] = 0

    while True:
        conv_dist = 0
        adj_qval_list = qval_list.copy()

        for state in mdp_in.state_reward_dict:
            for a in mdp_in.get_actions():

                # List to hold calculated probabilities of each action
                curr_sum = 0

                # Get Q'[s | a]
                for probability in mdp_in.get_state_transition_prob_list(state, a):
                    res_state = probability[0]
                    if res_state is not 0:
                        # Multiply by U[s'] (max Q-value of next state / Utility of next state)
                        curr_sum = curr_sum + (probability[1] * max_qvalue(adj_qval_list, res_state))

                # Add summed Q' to qval list ( Q'[s | a] = R(s) + Gamma *
                qval_list[state][a] = mdp_in.get_reward(state) + mdp_in.get_gamma() * curr_sum

            # Find the distance between the two values on a number line
            conv_dist = max(conv_dist, abs(current_util[state] - adjusted_util[state]))

        if conv_dist < epsilon:
            return adj_qval_list


# Get file names from command line args
state_file = sys.argv[1]
transition_file = sys.argv[2]

# Get epsilon value
eps = float(input("Please enter an epsilon value that you would like to use: "))

# Generate MDP
mdp = MDP(state_file, transition_file)
utility = qvalue_iteration(mdp, eps)

for s in utility:
    print("State", s, ":", " Utility:", utility[s], "  Policy: ", mdp.get_policy(s))
