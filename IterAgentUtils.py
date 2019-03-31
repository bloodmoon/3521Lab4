import csv


def parse_state_file(file, state_dict):
    state_dict_out = {}

    with open(file) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')

        for current_line in reader:
            state = int(current_line[0])
            reward = float(current_line[3])

            if state not in state_dict_out:
                state_dict[state] = reward


def parse_actions_probs_file(file, action_list, trans_matrix):

    with open(file) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')

        for current_line in reader:
            start_state = int(current_line[0])
            action = current_line[1]

            # Add action to list of possible actions if it does not exist
            if action not in action_list:
                action_list.append(action)

            # Parse the destinations and probabilities
            counter = 2

            # Create list to hold possible destinations and probabilities
            curr_prob_list = []

            while counter + 1 <= len(current_line):

                # Get destination State and prob of the choice
                dest_state = int(current_line[counter])
                prob_state = float(current_line[counter+1])

                # Add a tuple containing the info to the prob list
                curr_prob_list.append((dest_state, prob_state))

                # Increment counter to check for next probability
                counter = counter + 2

            # Add the results to the Transition Matrix
            trans_matrix[start_state][action] = curr_prob_list
