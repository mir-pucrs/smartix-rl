from database import Database
from q_value import Q_Value
from state import State

import random
import pprint

class Agent:


    MAX_TRAINING_EPISODES = 15
    MAX_STEPS_PER_EPISODE = 20


    def __init__(self):
        # Stats attributes
        self.episode_rewards = dict()

        # Agent attributes
        self.state = None
        self.reward = None
        self.action = None
        self.p_state = None
        self.p_reward = None
        self.p_action = None
        
        self.alpha = 0.9 # Learning rate
        self.gamma = 0.9 # Discount factor
        self.epsilon = 0.5 # Epsilon-greedy value

        self.r_plus = 9999
        self.exploration = 1

        self.q_values = dict()
        self.frequency = dict()

        self.action_weights = dict()



    def f(self, qv):
        if qv in self.q_values and self.frequency[qv] >= self.exploration:
            return self.q_values[qv]
        else:
            return self.r_plus



    def argmax_a(self, state):
        a = None
        max_value = float('-inf')
        
        if state == None:
            return a

        for action in self.env.get_available_actions(state):
            qv = Q_Value(state, action)
            f_value = self.f(qv)
            if f_value > max_value:
                max_value = f_value
                a = action

        return a



    def max_a(self, state, q_values):
        max_value = float('-inf')

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
            qv = Q_Value(state, action)
            if qv not in self.q_values:
                actions.append(action)

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



    def predict_q_values(self, state, action = None):
        state_features = self.env.get_state_features(state)
        if not action:
            prediction = dict()
            for a in self.action_weights.keys():
                prediction[a] = 1 # Start with 1 (bias term)
                for i in range(len(a)):
                    prediction[a] += self.action_weights[a][i] * state_features[i]
        else:
            prediction = 1 # Start with 1 (bias term)
            for i in range(len(self.action_weights[action])):
                prediction += self.action_weights[action][i] * state_features[i]
        return prediction



    def train(self, env):
        # Reset environment
        self.env = env
        self.state = self.env.reset()

        # Initialize features' weights vector
        features = self.env.get_state_features(self.state)
        action_space = self.env.get_action_space(self.state)
        for a in action_space:
            self.action_weights[a] = dict()
            for f in features.keys():
                self.action_weights[a][f] = random.random()
        
        # Print weights vector
        pprint.pprint(self.action_weights, width=2)

        # Episodes loop
        for episode in range(self.MAX_TRAINING_EPISODES):

            # Update statistics
            self.episode_rewards[episode] = 0

            # Steps in each episode
            for step in range(self.MAX_STEPS_PER_EPISODE):

                print("\n\nEpisode {}/{} @ Step {}".format(episode, self.MAX_TRAINING_EPISODES, step))
                print("Previous state: ", self.p_state)
                print("Previous action: ", self.p_action)
                print("Previous reward: ", self.p_reward)

                # Update frequencies
                if self.p_state != None:
                    # Update frequencies for state–action pairs
                    if Q_Value(self.p_state, self.p_action) in self.frequency and self.frequency[Q_Value(self.p_state, self.p_action)] > 0:
                        self.frequency[Q_Value(self.p_state, self.p_action)] += 1
                    else:
                        self.frequency[Q_Value(self.p_state, self.p_action)] = 1
                
                # Get action
                self.action = self.get_action_epsilon_greedy(self.state)
                print("Chosen action: ", self.action)

                # Save previous state
                self.p_state = self.state

                # Execute action in the environment
                self.state, self.reward = self.env.step(self.action)
                print("Resulting state: ", self.state)
                print("Resulting reward: ", self.reward)

                # Predict Q-Value for previous state-action
                previous_q_sa = self.predict_q_values(self.p_state, self.action)
                print("Previous Q(s,a):", previous_q_sa)

                # Predict Q-Value for possible actions in next state
                state_q_values = self.predict_q_values(self.state)

                # Update action weights
                for i in range(len(self.action_weights[self.action])):
                    self.action_weights[self.action][i] += self.alpha * (self.reward + self.gamma * self.max_a(self.state, state_q_values) - previous_q_sa) * 10**-5

                # Save previous action and reward
                self.p_action = self.action
                self.p_reward = self.reward

                # - # - # - # - # - #

                # Update statistics
                self.episode_rewards[episode] += self.reward

                # Write Q-Table to file
                with open('data/qtfr.txt', 'w+') as f:
                    f.write(repr(self.q_values) + '\n\n')

                # If episode's last execution
                if step+1 == self.MAX_STEPS_PER_EPISODE:
                    # Save current state-rewards and plot graphics
                    self.env.post_episode(self.q_values, episode)
                    print("Total reward in episode {}: {}".format(episode, self.episode_rewards[episode]))

                    # Decrease epsilon value by half
                    self.epsilon = self.epsilon / 2

                    # Reset environment and attributes
                    self.state = self.env.reset()
                    self.action = None
                    self.reward = None
                    self.p_state = None
                    self.p_action = None
                    self.p_reward = None