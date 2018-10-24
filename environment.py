from database import Database
from benchmark import Benchmark
from state import State
from action import Action
from agent import Agent

import time
import json
from pathlib import Path
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
        self.rewards_list = list()
        self.rewards_archive = self.load_rewards_archive()



    def step(self, action):
        action.execute()
        state = State()
        reward = self.get_reward(state)
        return state, reward



    def get_available_actions(self, state):
        available_actions = list()
        for column in state.indexes_map.keys():
            if state.indexes_map[column] == 0:
                available_actions.append(Action(column, 'CREATE'))
            else:
                available_actions.append(Action(column, 'DROP'))
        return available_actions



    def get_reward(self, state):
        if repr(state) in self.rewards_archive.keys():
            print("State-reward in dictionary!")
            self.rewards[state] = self.rewards_archive[repr(state)]
        else:
            print("State-reward not in dictionary")
            # self.rewards[state] = randint(2000, 2500)
            self.rewards[state] = self.benchmark.run()
            self.rewards_archive[repr(state)] = self.rewards[state]

        # Save reward to list for plotting
        self.rewards_list.append(self.rewards[state])

        return self.rewards[state]



    def get_action_space(self, state):
        action_space = list()
        for column in state.indexes_map.keys():
            action_space.append(Action(column, 'CREATE'))
            action_space.append(Action(column, 'DROP'))
        return action_space



    def get_state_features(self, state):
        state_features = dict()
        state_features['Bias'] = 1.0
        for column in state.indexes_map.keys():
            state_features[column] = state.indexes_map[column] + 1.0
        return state_features



    def reset(self):
        self.db.reset_indexes()
        return State()



    '''
        Data files and plots
    '''
    def load_rewards_archive(self):
        rewards_archive = Path("/path/to/file")
        if rewards_archive.is_file():
            with open(rewards_archive, 'r') as infile:
                return json.load(infile)
        else:
            return dict()

    def dump_rewards_archive(self):
        with open('data/rewards_archive.json', 'w+') as outfile:
            json.dump(self.rewards_archive, outfile)

    def dump_rewards_history_to_plot(self, rewards):
        with open('data/rewards_history_plot.dat', 'w+') as outfile:
            for value in rewards:
                outfile.write(str(value) + '\n')

    def dump_episode_reward_to_plot(self, episode_reward):
        with open('data/episode_reward_plot.dat', 'a+') as outfile:
            outfile.write(str(episode_reward) + '\n')

    def dump_weights_difference_to_plot(self, weights_difference):
        with open('data/weights_difference_plot.dat', 'a+') as outfile:
            outfile.write(str(weights_difference) + '\n')

    def plot_rewards_history(self, episode):
        with open("plots/averages_rewards_history.gnu") as f: 
            gp.c(f.read())
            gp.pdf('plots/rewards_history_plot_%d.pdf' % episode)
    
    def plot_episode_reward(self, episode):
        with open("plots/averages_episode_reward.gnu") as f: 
            gp.c(f.read())
            gp.pdf('plots/episode_reward_plot_%d.pdf' % episode)

    def plot_weights_difference(self, episode):
        with open("plots/averages_weights_difference.gnu") as f: 
            gp.c(f.read())
            gp.pdf('plots/weights_difference_plot_%d.pdf' % episode)

    def plot_error(self, episode):
        with open("plots/averages_episode_error.gnu") as f: 
            gp.c(f.read())
            gp.pdf('plots/episode_error_plot_%d.pdf' % episode)

    def post_episode(self, episode, episode_reward, weights_difference):
        # Dump rewards archive
        self.dump_rewards_archive()

        # Dump computed state-rewards up to now
        self.dump_rewards_history_to_plot(self.rewards_list)

        # Dump total episode reward
        self.dump_episode_reward_to_plot(episode_reward)

        # Dump weights difference to plot
        self.dump_weights_difference_to_plot(weights_difference)

        # Write episode rewards to file
        with open('data/episode_reward.csv', 'a+') as f:
            f.write(str(episode) + ', ' + str(episode_reward) + '\n')

        # Write highest state reward to file
        max_reward = max(self.rewards, key = lambda x: self.rewards.get(x))
        with open('data/max_reward.csv', 'a+') as outfile:
            outfile.write(str(episode) + ', ' + repr(max_reward) + ', ' + str(self.rewards[max_reward]) + '\n')

        # Plot rewards
        print("Plotting graphics...")
        self.plot_rewards_history(episode)
        self.plot_episode_reward(episode)
        self.plot_weights_difference(episode)
        self.plot_error(episode)


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