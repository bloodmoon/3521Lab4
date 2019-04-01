import sys
from mdp import MDP


def value_iteration(mdp_in, epsilon):

    # Setup Variables
    current_util = {st: 0 for st in mdp_in.state_reward_dict}

    while True:
        adjusted_util = current_util.copy()
        conv_dist = 0

        for state in mdp_in.state_reward_dict:
            maximum = []
            action_sum_list = []
            for a in mdp_in.get_actions():
                # List to hold calculated probabilities of each action
                curr_sum = 0

                for probability in mdp_in.get_state_transition_prob_list(state, a):
                    res_state = probability[0]
                    if res_state is not 0:
                        curr_sum = curr_sum + (probability[1] * adjusted_util[res_state])

                # Add to sum list for finding max and store resultant state for finding policy
                action_sum_list.append(tuple((curr_sum, a)))

            maximum.append(max(action_sum_list))

            # Update lists/dicts/policy
            current_util[state] = mdp_in.get_reward(state) + mdp_in.get_gamma() * maximum[0][0]
            mdp_in.set_policy(state, maximum[0][1])

            # Find the distance between the two values on a number line
            conv_dist = max(conv_dist, abs(current_util[state] - adjusted_util[state]))

        if conv_dist < epsilon:
            return adjusted_util


# Get file names from command line args
state_file = sys.argv[1]
transition_file = sys.argv[2]

# Get epsilon value
eps = float(input("Please enter an epsilon value that you would like to use: "))

# Generate MDP
mdp = MDP(state_file, transition_file)
utility = value_iteration(mdp, eps)

# Heading
print("===================")
print("=     Results     =")
print("===================")

# Output data to stdout
for s in utility:
    print("State", s, ":", " Utility:", utility[s], "  Policy: ", mdp.get_policy(s))
