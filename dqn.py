import random
import math
import time
import json
import os
import collections
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F

from environment import Environment
import gym

from pprint import pprint

class ReplayMemory():
    def __init__(self, capacity):
        self.memory = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.memory)

    def add(self, s0, a, r, s1, done):
        self.memory.append((s0, [a], [r], s1, [done]))

    def sample(self, batch_size):
        s0, a, r, s1, done = zip(*random.sample(self.memory, batch_size))
        s0 = torch.tensor(s0, dtype=torch.float)
        s1 = torch.tensor(s1, dtype=torch.float)
        a = torch.tensor(a, dtype=torch.long)
        r = torch.tensor(r, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.float)
        return s0, a, r, s1, done


class QNet(nn.Module):
    def __init__(self, n_features, n_actions):
        super(QNet, self).__init__()

        # Architecture
        self.nn = nn.Sequential(
            nn.Linear(n_features, 64),
            # nn.ReLU(),
            # nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        return self.nn(x)


class Agent:
    def __init__(self, env=Environment(), output_path=None):
        # Hyperparameters
        self.gamma = 0.9
        self.alpha = 0.0001
        # self.beta = 0.01
        # self.avg_reward = 0

        # Training
        self.n_steps = 50000  # 100k
        self.memory_size = 10000  # 10k
        self.memory = ReplayMemory(self.memory_size)
        self.target_update_interval = 128
        self.batch_size = 1024

        # Log
        if output_path == None:
            path = os.path.join('output', '{}_{}_{}_{}_{}_{}'.format(
                str(time.time()), self.alpha, self.n_steps, self.memory_size, self.target_update_interval, self.batch_size))
        else:
            path = os.path.join('output', output_path)
        if not os.path.isdir(path): os.makedirs(path)
        self.output_path = path + '/'

        # Epsilon
        self.epsilon = 1  # 100%
        self.epsilon_min = 0.01  # 1%
        self.epsilon_decay = 0.03  # 1%

        # Environment
        self.env = env
        self.n_features = self.env.n_features
        self.n_actions = self.env.n_actions
        # self.env = gym.make('CartPole-v0')
        # self.n_features = self.env.observation_space.shape[0]
        # self.n_actions = self.env.action_space.n

        # Model
        self.qnet = QNet(self.n_features, self.n_actions)
        self.qnet_target = QNet(self.n_features, self.n_actions)
        self.qnet_target.load_state_dict(self.qnet.state_dict())
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=self.alpha)

    """
        Model
    """
    def load_model(self):
        self.qnet.load_state_dict(torch.load(self.output_path+'model.pkl'))

    def save_model(self):
        torch.save(self.qnet.state_dict(), self.output_path+'model.pkl')

    """
        Training
    """
    def choose_action(self, state):
        if random.random() > self.epsilon:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            q_value = self.qnet.forward(state)
            action = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.n_actions)
        return action

    def learn_batch(self):
        s0, a, r, s1, done = self.memory.sample(self.batch_size)

        q_values = self.qnet(s0).gather(1,a)
        max_q_values = self.qnet_target(s1).max(1)[0].unsqueeze(1)
        q_targets = r + self.gamma * max_q_values

        loss = (q_values - q_targets).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def learn_episode(self, episode_batch):
        s0, a, r, s1, done = zip(*episode_batch)
        s0 = torch.tensor(s0, dtype=torch.float)
        s1 = torch.tensor(s1, dtype=torch.float)
        a = torch.tensor(a, dtype=torch.long)
        r = torch.tensor(r, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.float)

        q_values = self.qnet(s0).gather(1,a)
        max_q_values = self.qnet_target(s1).max(1)[0].unsqueeze(1)
        q_targets = r + self.gamma * max_q_values

        loss = (q_values - q_targets).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self):
        # Stats
        loss_history = list()
        actions_history = list()
        states_history = list()
        rewards_history = list()

        loss = 0
        episode_rewards = [0]
        episode_reward = 0
        episode_num = 0

        # Start plot
        plt.ion()
        plt.show()
        plt.pause(0.001)

        # Reset environment
        print("Preparing environment...")
        state = self.env.reset()
        states_history.append(state.tolist())

        self.episode_batch = list()

        # Start training
        print("Started training...")
        start = time.time()
        for step in range(self.n_steps):

            # Choose action
            action = self.choose_action(state)

            # Apply action
            next_state, reward, done, _ = self.env.step(action)
            
            # Update average reward
            # q_target = np.max(self.qnet_target(torch.tensor(np.concatenate(np.expand_dims(next_state, 0)), dtype=torch.float)).detach().numpy())
            # q_value = self.qnet(torch.tensor(np.concatenate(np.expand_dims(state, 0)), dtype=torch.float)).detach().numpy()[action]
            # delta = reward - self.avg_reward + (q_target - q_value)
            # self.avg_reward += self.beta*delta

            # Add to replay memory
            self.memory.add(state, action, reward, next_state, done)
            self.episode_batch.append((state, [action], [reward], next_state, [done]))

            # Update state
            state = next_state

            # Stats
            episode_reward += reward
            actions_history.append(action)
            rewards_history.append(reward)
            states_history.append(next_state.tolist())

            # Learn
            if len(self.memory) > self.batch_size:
                loss = self.learn_batch()
                loss_history.append(loss)

            # Save interval
            if (step != 0 and step % self.target_update_interval == 0):
                # Learn episode batch
                ep_loss = self.learn_episode(self.episode_batch)
                self.episode_batch = list()

                # Update step time
                end = time.time()
                elapsed = end - start
                start = time.time()

                str_state = str(state.tolist()).replace(", ", "").replace("[", ""). replace("]", "")
                print('\n', str_state[:45], str_state[45:])

                # Print stats
                print("episode: %2d \t acc_reward: %10.3f \t loss: %8.8f \t ep_loss: %8.8f \t elapsed: %6.2f \t epsilon: %2.4f" % (episode_num, episode_reward, loss, ep_loss, float(elapsed), self.epsilon))
                
                # Save logs
                log = "%2d\t%8.3f\t%8.8f\t%8.8f\t%.2f\t%.4f\n" % (episode_num, episode_reward, loss, ep_loss, elapsed, self.epsilon)
                with open(self.output_path+'log.txt', 'a+') as f:
                    f.write(log)
                with open(self.output_path+'rewards_history.json', 'w+') as f:
                    json.dump(rewards_history, f)
                with open(self.output_path+'states_history.json', 'w+') as f:
                    json.dump(states_history, f)
                with open(self.output_path+'actions_history.json', 'w+') as f:
                    json.dump(actions_history, f)

                # Stats
                episode_rewards.append(episode_reward)
                episode_reward = 0
                episode_num += 1

                # Plot
                # plt.plot(rewards_history[(episode_num-1)*128:])
                # plt.plot(loss_history[(episode_num-1)*128:])
                # plt.draw()
                # plt.pause(0.001)

                # Epsilon decay
                if len(self.memory) > self.batch_size: self.epsilon -= self.epsilon * self.epsilon_decay
                if self.epsilon < self.epsilon_min: self.epsilon = self.epsilon_min

                # Update target weights
                self.qnet_target.load_state_dict(self.qnet.state_dict())

                # Save model checkpoint
                self.save_model()
                
                self.env.debug()

                # self.env.reset()

        # Close and finish
        self.env.close()


if __name__ == "__main__":
    import os
    print("Restarting PostgreSQL...")
    os.system('sudo systemctl restart postgresql@12-main')
    
    agent = Agent(env=Environment())
    agent.train()

    # env1 = Environment(allow_columns=False, flip=False, reward_function=1)
    # env2 = Environment(allow_columns=False, flip=False, reward_function=2)
    # env3 = Environment(allow_columns=False, flip=False, reward_function=3)
    # env4 = Environment(allow_columns=False, flip=False, reward_function=4)
    # env5 = Environment(allow_columns=False, flip=False, reward_function=5)

    # env6 = Environment(allow_columns=True, flip=False, reward_function=1)
    # env7 = Environment(allow_columns=True, flip=False, reward_function=2)
    # env8 = Environment(allow_columns=True, flip=False, reward_function=3)
    # env9 = Environment(allow_columns=True, flip=False, reward_function=4)
    # env10 = Environment(allow_columns=True, flip=False, reward_function=5)

    # env11 = Environment(allow_columns=False, flip=True, reward_function=1)
    # env12 = Environment(allow_columns=False, flip=True, reward_function=2)
    # env13 = Environment(allow_columns=False, flip=True, reward_function=3)
    # env14 = Environment(allow_columns=False, flip=True, reward_function=4)
    # env15 = Environment(allow_columns=False, flip=True, reward_function=5)

    # env16 = Environment(allow_columns=True, flip=True, reward_function=1)
    # env17 = Environment(allow_columns=True, flip=True, reward_function=2)
    # env18 = Environment(allow_columns=True, flip=True, reward_function=3)
    # env19 = Environment(allow_columns=True, flip=True, reward_function=4)
    # env20 = Environment(allow_columns=True, flip=True, reward_function=5)

    # agent = Agent(env=env1, output_path='env1')
    # agent.train()

    # agent = Agent(env=env2, output_path='env2')
    # agent.train()

    # agent = Agent(env=env3, output_path='env3')
    # agent.train()

    # agent = Agent(env=env4, output_path='env4')
    # agent.train()

    # agent = Agent(env=env5, output_path='env5')
    # agent.train()

    # agent = Agent(env=env6, output_path='env6')
    # agent.train()

    # agent = Agent(env=env7, output_path='env7')
    # agent.train()

    # agent = Agent(env=env8, output_path='env8')
    # agent.train()

    # agent = Agent(env=env9, output_path='env9')
    # agent.train()

    # agent = Agent(env=env10, output_path='env10')
    # agent.train()

    # agent = Agent(env=env11, output_path='env11')
    # agent.train()

    # agent = Agent(env=env12, output_path='env12')
    # agent.train()

    # agent = Agent(env=env13, output_path='env13')
    # agent.train()

    # agent = Agent(env=env14, output_path='env14')
    # agent.train()

    # agent = Agent(env=env15, output_path='env15')
    # agent.train()

    # agent = Agent(env=env16, output_path='env16')
    # agent.train()

    # agent = Agent(env=env17, output_path='env17')
    # agent.train()

    # agent = Agent(env=env18, output_path='env18')
    # agent.train()

    # agent = Agent(env=env19, output_path='env19')
    # agent.train()

    # agent = Agent(env=env20, output_path='env20')
    # agent.train()
