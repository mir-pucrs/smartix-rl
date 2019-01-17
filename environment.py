from database import Database
from benchmark import Benchmark
from state import State
from action import Action
from agent import Agent

import json
# import matplotlib.pyplot as plt
from pathlib import Path
from random import randint


class Environment:


    def __init__(self):
        # Database instance
        self.db = Database()

        # Benchmark
        self.benchmark = Benchmark('TPCH')

        # Current rewards dictionary
        self.rewards = dict()

        # State-rewards file records to dict
        self.rewards_list = list()
        self.rewards_archive = dict()
        self.visited_states = list()


    def step(self, action):
        action.execute()
        state = State()
        reward = self.get_reward(state)
        return state, reward


    def get_available_actions(self, state):
        available_actions = list()
        available_actions.append(Action('PASS', 'PASS', 'PASS'))

        for table, columns in state.indexes_map.items():
            for column in columns.keys():
                if state.indexes_map[table][column] == 0:
                    available_actions.append(Action(table, column, 'CREATE'))
                else:
                    available_actions.append(Action(table, column, 'DROP'))
        
        return available_actions


    def get_reward(self, state):
        if repr(state) in self.rewards_archive.keys():
            print("State-reward in dictionary!")
            self.rewards[state] = self.rewards_archive[repr(state)]
        else:
            print("State-reward not in dictionary")
            # self.rewards[state] = randint(2500, 2550)
            self.rewards[state] = self.benchmark.run()
            self.rewards_archive[repr(state)] = self.rewards[state]

        # Save reward to list for plotting
        self.rewards_list.append(self.rewards[state])

        # Save visited state
        if state not in self.visited_states:
            self.visited_states.append(state)

        return self.rewards[state]


    def get_action_space(self, state):
        action_space = list()
        action_space.append(Action('PASS', 'PASS', 'PASS'))
        for table, columns in state.indexes_map.items():
            for column in columns.keys():
                action_space.append(Action(table, column, 'CREATE'))
                action_space.append(Action(table, column, 'DROP'))
        return action_space


    def get_state_features(self, state):
        state_features = dict()
        state_features['Bias'] = 1.0
        for table, columns in state.indexes_map.items():
            for column in columns.keys():
                state_features[column] = state.indexes_map[table][column] + 1.0
        return state_features


    def reset(self):
        self.db.reset_indexes()
        return State()


    '''
        Data files and plots
    '''
    def dump_rewards_archive(self):
        with open('data/rewards_archive.json', 'w+') as outfile:
            json.dump(self.rewards_archive, outfile)

    def dump_rewards_history(self, rewards):
        with open('data/rewards_history_plot.dat', 'w+') as outfile:
            for value in rewards:
                outfile.write(str(value) + '\n')

    def post_episode(self, episode, episode_reward):
        # Dump rewards archive
        self.dump_rewards_archive()

        # Dump computed state-rewards up to now
        self.dump_rewards_history(self.rewards_list)

        # Write episode rewards to file
        with open('data/episode_reward.dat', 'a+') as f:
            f.write(str(episode) + ', ' + str(episode_reward) + '\n')
        
        # Write number of visited distinct states
        with open('data/visited_distinct_states.dat', 'a+') as f:
            f.write(str(episode) + ', ' + str(len(self.visited_states)) + '\n')

        # Write highest state reward to file
        max_reward = max(self.rewards, key = lambda x: self.rewards.get(x))
        with open('data/max_reward.dat', 'a+') as outfile:
            outfile.write(str(episode) + ', ' + repr(max_reward) + ', ' + str(self.rewards[max_reward]) + '\n')



if __name__ == "__main__":
    agent = Agent()
    env = Environment()
    agent.train(env)
