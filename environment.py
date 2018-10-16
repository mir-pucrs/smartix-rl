from database import Database
from benchmark import Benchmark
from state import State
from action import Action
from agent import Agent

import time
import json
from random import randint
import PyGnuplot as gp


class Environment:


    def __init__(self):
        # Database instance
        self.db = Database()

        # Benchmark
        self.benchmark = Benchmark('TPCH')

        # Current rewards dictionary
        self.rewards = dict()

        # State-rewards file records to dict
        # self.rewards_archive = self.load_rewards_archive()
        self.rewards_archive = dict()



    def execute(self, action):
        action.execute()



    def get_available_actions(self, state):
        available_actions = list()
        for column in state.indexes_map.keys():
            # print(state.indexes_map[column])
            if state.indexes_map[column] == 0:
                available_actions.append(Action(column, 'CREATE'))
            else:
                available_actions.append(Action(column, 'DROP'))
        # print("\n\nAvailable actions:", available_actions, "\n\n")
        return available_actions



    def get_reward(self, state):
        # if repr(state) in self.rewards_archive.keys():
        #     self.rewards[state] = self.rewards_archive[repr(state)]
        #     print("State-reward in dictionary!")
        # else:
        #     self.rewards[state] = self.benchmark.run()
        #     self.rewards_archive[repr(state)] = self.rewards[state]
        #     print("State-reward not in dictionary")
        
        self.rewards[state] = self.benchmark.run()

        # print("\n\nSTATE IS:\n\n",state,"\n\n")
        with open('data/srrf.txt', 'w+') as f:
            f.write(repr(state) + ': ' + str(self.rewards[state]) + '\n')

        return self.rewards[state]



    def reset(self):
        return self.db.reset_indexes()



    '''
        Data files and plots
    '''
    def load_rewards_archive(self):
        with open('data/rewards_archive.json', 'r') as infile:
            return json.load(infile)

    def dump_rewards_archive(self):
        with open('data/rewards_archive.json', 'w+') as outfile:
            json.dump(self.rewards_archive, outfile)

    def dump_rewards_to_plot(self):
        with open('data/rewards.dat', 'w+') as outfile:
            for value in self.rewards.values():
                outfile.write(str(value) + '\n')

    def plot_rewards(self, episode):
        with open("plots/averages.gnu") as f: 
            gp.c(f.read())
            gp.pdf('plots/rewards_plot_%d.pdf' % episode)

    def post_episode(self, q_values, episode):
        # Dump rewards archive
        self.dump_rewards_archive()

        # Dump computed state-rewards up to now
        self.dump_rewards_to_plot()
        
        # Write highest Q-value to file
        max_q_value = max(q_values, key = lambda x: q_values.get(x) )
        with open('data/max_q_values.txt', 'a+') as outfile:
            outfile.write(repr(max_q_value) + ': ' + str(q_values[max_q_value]) + '\n')

        # Write highest state reward to file
        max_reward = max(self.rewards, key = lambda x: self.rewards.get(x) )
        with open('data/max_rewards.txt', 'a+') as outfile:
            outfile.write(repr(max_reward) + ': ' + str(self.rewards[max_reward]) + '\n')

        # Plot rewards
        self.plot_rewards(episode)
        print("Plotting graphics...")


if __name__ == "__main__":
    agent = Agent()
    env = Environment()

    start_time = time.time()
    agent.train(env)
    elapsed_time = time.time() - start_time

    print("It took %.2f seconds to train.", elapsed_time)

    # env = Environment()
    # env.reset()
    # print("Resetted Environment")

    # env = Environment()
    # print(env.rewards_archive)