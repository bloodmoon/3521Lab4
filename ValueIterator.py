from collections import defaultdict
import IterAgentUtils
import sys

# Setup Variables
transition_matrix = defaultdict(dict)
action_list = []
state_reward_dict = {}

state_file = sys.argv[1]
transition_file = sys.argv[2]

IterAgentUtils.parse_state_file(state_file, state_reward_dict)
IterAgentUtils.parse_actions_probs_file(transition_file, action_list, transition_matrix)

print(state_reward_dict)
