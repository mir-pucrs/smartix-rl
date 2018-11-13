from database import Database
from state import State

import random, pprint, copy, math

class Agent:


    MAX_TRAINING_EPISODES = 25
    MAX_STEPS_PER_EPISODE = 100


    def __init__(self):
        # Stats attributes
        self.episode_reward = dict()

        # Agent attributes
        self.state = None
        self.next_state = None
        self.reward = 0.0
        self.action = None
        
        self.alpha = 0.05 # Learning rate
        self.gamma = 0.8 # Discount factor
        self.epsilon = 1.0 # Epsilon-greedy value

        self.action_weights = dict()
        self.prev_action_weights = dict()



    def weights_difference(self):
        prev = 0
        for weights in self.prev_action_weights.values():
            prev += sum(weights.values())
        
        curr = 0
        for weights in self.action_weights.values():
            curr += sum(weights.values())

        print("Weights difference")
        print("Prev:", prev)
        print("Curr:", curr)
        print("Diff:", math.sqrt(abs(curr - prev)))

        return math.sqrt(abs(curr - prev))



    def argmax_a(self, state):
        a = None
        max_value = float('-inf')

        q_values = self.predict(state)

        for action in self.env.get_available_actions(state):
            q_sa = q_values[action]
            if q_sa > max_value:
                max_value = q_sa
                a = action

        return a



    def max_a(self, state):
        max_value = float('-inf')

        q_values = self.predict(state)

        for action in self.env.get_available_actions(state):
            q_sa = q_values[action]
            if q_sa > max_value:
                max_value = q_sa

        if max_value == float('-inf'): 
            max_value = 0.0
        
        return max_value



    def get_random_action(self, state):
        actions = list()

        for action in self.env.get_available_actions(state):
            actions.append(action)

        return random.choice(actions)



    def get_action_epsilon_greedy(self, state):
        # Epsilon-greedily choose action
        rand = random.random()

        if rand > self.epsilon: # EXPLOIT
            print("Random %.2f > %.2f Epsilon (Get argmax action)" % (rand, self.epsilon))
            action = self.argmax_a(state)
            action_type = 'ARGMAX'
            # print("Action:", action)
        else: # EXPLORE
            print("Random %.2f < %.2f Epsilon (Get random action)" % (rand, self.epsilon))
            action = self.get_random_action(state)
            action_type = 'RANDOM'
            # print("Action:", action)

        return action, action_type



    def initialize_weights(self, state):
        state_features = self.env.get_state_features(self.state)
        action_space = self.env.get_action_space(self.state)

        for a in action_space:
            self.action_weights[a] = dict()
            self.prev_action_weights[a] = dict()
            for f in state_features.keys():
                self.action_weights[a][f] = random.random()
                self.prev_action_weights[a][f] = 0.0



    def predict(self, state, action = None):
        state_features = self.env.get_state_features(state)

        if action == None:
            prediction = dict()
            for action, weights in self.action_weights.items():
                prediction[action] = 0.0
                for feature, weight in weights.items():
                    prediction[action] += weight * state_features[feature]
        else:
            prediction = 0.0
            for feature, value in self.action_weights[action].items():
                prediction += value * state_features[feature]

        return prediction



    def update(self, state, action, td_target, q_value):
        state_features = self.env.get_state_features(state)

        for weight in self.action_weights[action].keys():
            partial_derivative = state_features[weight]
            self.action_weights[self.action][weight] += self.alpha * (td_target - q_value) * partial_derivative
    


    def train(self, env):
        # Reset environment
        self.env = env
        self.state = self.env.reset()

        # Initialize features' weights vector
        self.initialize_weights(self.state)

        pprint.pprint(self.action_weights)

        # Episodes loop
        for episode in range(self.MAX_TRAINING_EPISODES):

            # Update statistics
            self.episode_reward[episode] = 0

            # Update epsilon value
            self.epsilon = 1 / (episode + 1)

            # Steps in each episode
            for step in range(self.MAX_STEPS_PER_EPISODE):

                print("\n\nEpisode {}/{} @ Step {}".format(episode, self.MAX_TRAINING_EPISODES, step))

                # Get action
                self.action, action_type = self.get_action_epsilon_greedy(self.state)
                print("Chosen action: ", self.action)

                # Write for every state which utility values are for each possible action
                with open('data/state_utilities.csv', 'a+') as outfile:
                    outfile.write(str(episode) + ' | ' + str(step) + ' | ' + str(repr(self.state)) + ' | ' + action_type + ' | ' + str(repr(self.action)))
                    prediction = self.predict(self.state)
                    for action in self.env.get_available_actions(self.state):
                        outfile.write(' | ' + str(repr(action)) + ': ' + str(prediction[action]))
                    outfile.write('\n')
                # Write epsilon value to file
                with open('data/epsilon_history.dat', 'a+') as outfile:
                    outfile.write(str(self.epsilon) + '\n')

                # Execute action in the environment
                self.next_state, self.reward = self.env.step(self.action)
                print("Resulting state: ", self.next_state)
                print("Resulting reward: ", self.reward)

                # Predict Q-Value for previous state-action
                q_value = self.predict(self.state, self.action)

                # TD target (what really happened)
                td_target = self.reward + self.gamma * self.max_a(self.state)

                error = td_target - q_value
                print("TD target:", td_target, '| Q-value', q_value, '| Error:', error)

                # Update action weights
                self.update(self.state, self.action, td_target, q_value)

                # Update current state
                self.state = self.next_state



                # Update statistics
                self.episode_reward[episode] += self.reward
                with open('data/history.dat', 'a+') as outfile:
                    outfile.write(str(episode) + ', ' + str(step) + ', ' + str(self.reward) + ', ' + str(error) + '\n')
                with open('data/episode_error_plot.dat', 'a+') as outfile:
                    outfile.write(str(error) + '\n')

                # If episode's last execution
                if step+1 == self.MAX_STEPS_PER_EPISODE:
                    # Save current state-rewards and plot graphics
                    self.env.post_episode(episode, self.episode_reward[episode], self.weights_difference())
                    self.prev_action_weights = copy.deepcopy(self.action_weights)
                    
                    print("Total reward in episode {}: {}".format(episode, self.episode_reward[episode]))

                    # Decrease epsilon value by half
                    # self.epsilon = (80 - math.log(episode+1) * 15) * 0.01
                    print("Epsilon =", self.epsilon)

                    # Reset environment and attributes
                    self.state = self.env.reset()
                    self.next_state = None
                    self.action = None
                    self.reward = None