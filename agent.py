from database import Database
from state import State
from action import Action

import random, pprint, copy, math, time

class Agent:


    MAX_TRAINING_EPISODES = 50
    MAX_STEPS_PER_EPISODE = 100


    def __init__(self):
        # Stats attributes
        self.episode_reward = dict()
        self.episode_duration = dict()

        # Agent attributes
        self.state = None
        self.next_state = None
        self.reward = 0.0
        self.action = None
        
        self.alpha = 0.1 # Learning rate
        self.gamma = 0.9 # Discount factor
        self.epsilon = 0.9 # Epsilon value

        self.action_weights = dict()
        # self.features_exploration = dict()
        # self.exploration_threshold = 2


    def argmax_a(self, state):
        a = None
        max_value = float('-inf')

        q_values = self.predict(state)

        for action in self.env.get_available_actions(state):
            q_value = q_values[action]
            if q_value > max_value:
                max_value = q_value
                a = action

        return a


    def max_a(self, state):
        max_value = float('-inf')

        q_values = self.predict(state)

        for action in self.env.get_available_actions(state):
            q_value = q_values[action]
            if q_value > max_value:
                max_value = q_value

        if max_value == float('-inf'): 
            max_value = 0.0
        
        return max_value


    def get_random_action(self, state):
        actions = self.env.get_available_actions(state)
        return random.choice(actions)


    def get_action_epsilon_greedy(self, state):
        # Epsilon-greedily choose action
        rand = random.random()

        if rand > self.epsilon: # EXPLOIT
            print("Random %.2f > %.2f Epsilon (Get argmax action)" % (rand, self.epsilon))
            action = self.argmax_a(state)
        else: # EXPLORE
            print("Random %.2f < %.2f Epsilon (Get random action)" % (rand, self.epsilon))
            action = self.get_random_action(state)

        return action


    def weights_initialization(self, state):
        state_features = self.env.get_state_features(self.state)
        action_space = self.env.get_action_space(self.state)

        for action in action_space:
            self.action_weights[action] = dict()
            for feature in state_features.keys():
                self.action_weights[action][feature] = random.random()


    # def weights_optimistic_initialization(self, state):
    #     state_features = self.env.get_state_features(self.state)
    #     action_space = self.env.get_action_space(self.state)

    #     for a in action_space:
    #         self.action_weights[a] = dict()
    #         self.prev_action_weights[a] = dict()
    #         for f in state_features.keys():
    #             self.action_weights[a][f] = 1000.0
    #             self.prev_action_weights[a][f] = 0.0
        
    #     for f in state_features.keys():
    #             self.features_exploration[f] = 0


    # def predict_optimistic(self, state, action = None):
    #     state_features = self.env.get_state_features(state)

    #     if action == None:
    #         prediction = dict()
    #         for action, weights in self.action_weights.items():
    #             prediction[action] = 0.0
    #             for feature, weight in weights.items():
    #                 prediction[action] += weight * state_features[feature]
    #     else:
    #         if action.type == 'CREATE':
    #             self.features_exploration[action.column] += 1
    #         prediction = 0.0
    #         for feature, value in self.action_weights[action].items():
    #             prediction += value * state_features[feature]

    #     return prediction


    # def update_optimistic(self, state, action, td_target, q_value):
    #     state_features = self.env.get_state_features(state)

    #     for weight in self.action_weights[action].keys():
    #         if self.features_exploration[weight] >= self.exploration_threshold or weight == 'Bias':
    #             if weight == 'Bias':
    #                 print('Updating bias...')
    #             if self.features_exploration[weight] >= self.exploration_threshold:
    #                 print('Feature explored:', weight)
    #             partial_derivative = state_features[weight]
    #             self.action_weights[self.action][weight] += self.alpha * (td_target - q_value) * partial_derivative
    #         else:
    #             print('Feature not sufficiently explored:', weight)


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
            feature = state_features[weight]
            self.action_weights[action][weight] += self.alpha * (td_target - q_value) * feature


    def train(self, env):
        # Reset environment
        self.env = env
        self.state = self.env.reset()

        # Initialize features' weights vector
        self.weights_initialization(self.state)

        # Episodes loop
        for episode in range(self.MAX_TRAINING_EPISODES):

            # Update statistics
            self.episode_reward[episode] = 0
            episode_start_time = time.time()

            # Steps in each episode
            for step in range(self.MAX_STEPS_PER_EPISODE):

                print("\n\nEpisode {}/{} @ Step {}".format(episode, self.MAX_TRAINING_EPISODES, step))

                # Get action
                self.action = self.get_action_epsilon_greedy(self.state)

                # Execute action in the environment
                self.next_state, self.reward = self.env.step(self.action)
                print("Resulting reward: ", self.reward)
                print("Resulting state: ", self.next_state)

                # Predict Q-Value for previous state-action
                q_value = self.predict(self.state, self.action)

                best_next_state = self.max_a(self.next_state)
                # TD target (what really happened)
                td_target = self.reward + self.gamma * best_next_state

                # Calculate and print TD error
                td_error = td_target - q_value
                print("TD target:", td_target, '| Q-value', q_value, '| Error:', td_error, "| Max_a:", best_next_state)
                with open('data/errors.dat', 'a+') as f:
                    f.write(str(td_error) + '\n')

                # Update action weights
                self.update(self.state, self.action, td_target, q_value)

                # Update current state
                self.state = self.next_state

                # Update episode stats
                self.episode_reward[episode] += self.reward

                # If episode's last execution
                if step+1 == self.MAX_STEPS_PER_EPISODE:

                    # Calculate episode duration
                    self.episode_duration[episode] = time.time() - episode_start_time

                    print("\n\n\n### FINISHED EPISODE %s ###" % episode)
                    print("Epsilon:", self.epsilon)
                    print("Reward:", self.episode_reward[episode])
                    print("Duration:", self.episode_duration[episode])

                    # Save data
                    self.env.post_episode(episode, self.episode_reward[episode], self.episode_duration[episode])

                    # Decrease epsilon value by 20%
                    self.epsilon -= self.epsilon * 0.2

                    # Reset environment and attributes
                    self.state = self.env.reset()
                    self.next_state = None
                    self.action = None
                    self.reward = None
