from database import Database
from q_value import Q_Value
from state import State

import random


class Agent:


    MAX_TRAINING_EPISODES = 25 # TRY 100


    def __init__(self):
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

        self.weights = list()


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



    def max_a(self, state):
        max_value = float('-inf')

        for action in self.env.get_available_actions(state):
            qv = Q_Value(state, action)
            if qv in self.q_values:
                q_sa = self.q_values[qv]
                if q_sa > max_value:
                    max_value = q_sa

        if max_value == float('-inf'): 
            max_value = 0.0
        
        return max_value



    def get_random_action(self, state):
        return random.choice(self.env.get_available_actions(state))



    def get_action(self, state):
        # Get reward for current state
        reward_prime = self.env.get_reward(state)
        print("Reward prime:", reward_prime)
        
        if self.p_state != None:
            # Calculate predicted q_sa
            features = state.get_features()
            predicted_q = 0
            for i in range(len(self.weights)):
                predicted_q += features[i] * self.weights[i]

            print("\n\nPrevious weights:", self.weights)

            print("\n\nPredicted Q_sa:", predicted_q)
            
            # Adjust weights according to actual reward
            for i in range(len(self.weights)):
                self.weights[i] = self.weights[i] + self.alpha * (self.p_reward + self.gamma * self.max_a(state) - predicted_q) # * derivadas!!!

            print("\n\nUpdated weights:", self.weights)

            # Adjust table frequencies and zero new Q-Values
            if Q_Value(self.p_state, self.p_action) in self.frequency and self.frequency[Q_Value(self.p_state, self.p_action)] > 0:
                print("Already executed this Q(s,a), increment 1 in frequency")
                self.frequency[Q_Value(self.p_state, self.p_action)] += 1
            else:
                print("Never executed this Q(s,a), zero Q-Value and 1 to frequency")
                self.frequency[Q_Value(self.p_state, self.p_action)] = 1
                self.q_values[Q_Value(self.p_state, self.p_action)] = 0

            # Bellman equation
            q_sa = self.q_values[Q_Value(self.p_state, self.p_action)]
            self.q_values[Q_Value(self.p_state, self.p_action)] = q_sa + self.alpha * (self.p_reward + self.gamma * self.max_a(state) - q_sa)
            print("Calculated Bellman equation")
        
        # Update previous state and reward
        self.p_state = state
        self.p_reward = reward_prime

        # Epsilon-greedily choose action
        rand = random.random()
        if rand > self.epsilon: # EXPLOIT
            print("Random %.2f > %.2f Epsilon" % (rand, self.epsilon))
            self.p_action = self.argmax_a(self.p_state)
            print("Argmax action:", self.p_action)
        else: # EXPLORE
            print("Random %.2f < %.2f Epsilon" % (rand, self.epsilon))
            self.p_action = self.get_random_action(self.p_state)
            print("Random action:", self.p_action)

        # Return chosen action
        return self.p_action



    def train(self, env):
        # Get and reset environment
        self.env = env
        self.env.reset()

        # Initialize weights vector with random numbers
        self.weights.append(1.0) # First element is bias
        for _ in range(len(State().indexes_map)):
            self.weights.append(random.random())

        # Episodes loop
        episode = 0
        while episode < self.MAX_TRAINING_EPISODES:

            # Runs for each episode
            for execution in range(2):

                print("\n\nEP. #%d | EXEC. #%d" % (episode, execution))

                # Get state, action and execute action in the environment
                self.state = State()
                self.action = self.get_action(self.state)
                self.env.execute(self.action)
                
                # Write Q-Tables to file
                with open('data/qtfr.txt', 'w+') as f:
                    f.write(repr(self.q_values) + '\n\n')

                # If episode's last execution
                if execution == 1:
                    # Save current state-rewards and plot graphics
                    self.env.post_episode(self.q_values, episode)

                    # Decrease epsilon value by half
                    self.epsilon = self.epsilon / 2

                    # Reset attributes and environment
                    self.p_state = None
                    self.p_action = None
                    self.p_reward = None
                    self.state = None
                    self.action = None
                    self.reward = None
                    self.env.reset()

                    # Loop management
                    episode += 1
                    break