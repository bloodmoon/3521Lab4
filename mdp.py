"""A Markov Decision Process Module"""

from collections import defaultdict
import IterAgentUtils


class MDP:

    def __init__(self, state_file, transition_file):
        self.transition_matrix = defaultdict(dict)
        self.action_list = []
        self.state_reward_dict = {}
        IterAgentUtils.parse_state_file(state_file, self.state_reward_dict)
        IterAgentUtils.parse_actions_probs_file(transition_file, self.
                                                action_list, self.transition_matrix)
        self.discount = 0
        self.gamma = 1

    def get_reward(self, state):
        return self.state_reward_dict[state]

    def get_actions(self):
        return self.action_list

    def get_discount(self):
        return self.discount

    def get_transition_matrix(self):
        return self.transition_matrix

    def get_state_transition_prob_list(self, state, action):
        if self.is_terminal(state):
            # TODO : PROB  SHOULD JUST RETURN ZERO TO MAKE UNCHANGED
            tuplist = list()
            tuplist.append(tuple((0, 0)))
            return tuplist
        else:
            return self.transition_matrix[state][action]

    def is_terminal(self, state):
        if abs(self.state_reward_dict[state]) == 1:
            return True
        else:
            return False

    def get_gamma(self):
        return self.gamma
