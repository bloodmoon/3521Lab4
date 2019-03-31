import sys
from mdp import MDP


def value_iteration(mdp_in, epsilon):

    # Setup Variables
    current_util = {s: 0 for s in mdp_in.state_reward_dict}

    while True:
        adjusted_util = current_util.copy()
        conv_dist = 0

        for s in mdp_in.state_reward_dict:
            maximum = 0
            for a in mdp_in.get_actions():
                # TODO: Figure out how to multiply probability by utility
                for probability in mdp_in.get_state_transition_prob_list(s, a):
                    # maximum = max(sum(probability * current))
                    print(probability)

            current_util[s] = mdp_in.get_reward(s) + mdp_in.get_gamma() * maximum

            conv_dist = max(conv_dist, abs(current_util[s] - adjusted_util[s]))

        if conv_dist < epsilon:
            return adjusted_util


# Get file names from command line args
state_file = sys.argv[1]
transition_file = sys.argv[2]

# Generate MDP
mdp = MDP(state_file, transition_file)
value_iteration(mdp, 0.000006)


