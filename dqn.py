import random
import time
import json
import os
import collections
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from environment import Environment
import gym


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
        print(n_features, n_actions, "blaaaa")
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
    def __init__(self, env=Environment(), output_path=None, tag=None):
        # Hyperparameters
        self.gamma = 0.9  # 0.9
        self.alpha = 0.0001  # 0.0001

        # Training
        self.n_steps = 100000  # 100k
        self.memory_size = 10000  # 10k
        self.memory = ReplayMemory(self.memory_size)
        self.target_update_interval = 128  # 128
        self.batch_size = 1024  # 1024

        # Epsilon
        self.epsilon = 1  # 100%
        self.epsilon_min = 0.01  # 1% 0.01
        self.epsilon_decay = 0.01  # 1% 0.01

        # Log
        if output_path == None:
            path = os.path.join('output', '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                str(time.time()), 
                self.alpha, 
                self.gamma, 
                self.n_steps, 
                self.memory_size, 
                self.target_update_interval, 
                self.batch_size, 
                self.epsilon_min, 
                self.epsilon_decay, 
                str(tag))
            )
        else:
            path = os.path.join('output', output_path)
        if not os.path.isdir(path): os.makedirs(path)
        self.output_path = path + '/'

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
    def load_model(self, path):
        self.qnet.load_state_dict(torch.load(path+'/model.pkl'))

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
        states_history = list()
        actions_history = list()
        rewards_history = list()
        episode_rewards = list()
        transitions_history = list()
        batch_loss = 0
        episode_reward = 0
        episode_num = 0

        # Start plot
        # plt.ion()
        # plt.show()
        # plt.pause(0.001)

        # Reset environment
        print("Preparing environment...")
        state = self.env.reset()
        states_history.append(state.tolist())

        # Start training
        print("Started training...")
        start = time.time()
        for step in range(self.n_steps):

            # Choose action
            action = self.choose_action(state)

            # Apply action
            next_state, reward, done, _ = self.env.step(action)

            # Add to replay memory
            self.memory.add(state, action, reward, next_state, done)

            # Update state
            state = next_state

            # Stats
            episode_reward += reward
            actions_history.append(action)
            rewards_history.append(reward)
            states_history.append(next_state.tolist())
            transitions_history.append((state.tolist(), [action], [reward], next_state.tolist(), [done]))

            # Learn
            if len(self.memory) > self.batch_size:
                batch_loss += self.learn_batch()

            # Save interval
            if (step != 0 and step % self.target_update_interval == 0):

                # Update step time
                end = time.time()
                elapsed = end - start
                start = time.time()

                # Print stats
                print("episode: %2d \t acc_reward: %10.3f \t batch_loss: %8.8f \t elapsed: %6.2f \t epsilon: %2.4f" % (episode_num, episode_reward, batch_loss, float(elapsed), self.epsilon))
                
                # Save logs
                log = "%2d\t%8.3f\t%8.8f\t%.2f\t%.4f\n" % (episode_num, episode_reward, batch_loss, elapsed, self.epsilon)

                with open(self.output_path+'log.txt', 'a+') as f:
                    f.write(log)
                with open(self.output_path+'states_history.json', 'w+') as f:
                    json.dump(states_history, f)
                with open(self.output_path+'actions_history.json', 'w+') as f:
                    json.dump(actions_history, f)
                with open(self.output_path+'rewards_history.json', 'w+') as f:
                    json.dump(rewards_history, f)
                with open(self.output_path+'transitions_history.json', 'w+') as f:
                    json.dump(transitions_history, f)

                # Stats
                episode_rewards.append(episode_reward)
                episode_reward = 0
                episode_num += 1
                batch_loss = 0

                # Plot
                # plt.plot(episode_rewards)
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
            
        # Close and finish
        self.env.close()

    def test(self, model_path=''):
        # Stats
        states_history = list()
        actions_history = list()
        rewards_history = list()
        episode_rewards = list()
        transitions_history = list()
        batch_loss = 0
        episode_reward = 0
        episode_num = 0

        # IMPORTANT!!!!!!!!!!
        self.load_model(model_path)
        self.epsilon = 0.0

        # Reset environment
        print("Preparing environment...")
        state = self.env.reset()
        states_history.append(state.tolist())

        # Start training
        print("Started training...")
        start = time.time()
        for step in range(self.n_steps):

            # Choose action
            action = self.choose_action(state)

            # Apply action
            next_state, reward, done, _ = self.env.step(action)

            # Update state
            state = next_state

            # Stats
            episode_reward += reward
            actions_history.append(action)
            rewards_history.append(reward)
            states_history.append(next_state.tolist())
            transitions_history.append((state.tolist(), [action], [reward], next_state.tolist(), [done]))

            # Save interval
            if (step != 0 and step % self.target_update_interval == 0):

                # Update step time
                end = time.time()
                elapsed = end - start
                start = time.time()

                # Print stats
                print("episode: %2d \t acc_reward: %10.3f \t batch_loss: %8.8f \t elapsed: %6.2f \t epsilon: %2.4f" % (episode_num, episode_reward, batch_loss, float(elapsed), self.epsilon))
                
                # Save logs
                log = "%2d\t%8.3f\t%8.8f\t%.2f\t%.4f\n" % (episode_num, episode_reward, batch_loss, elapsed, self.epsilon)

                with open(self.output_path+'log.txt', 'a+') as f:
                    f.write(log)
                with open(self.output_path+'states_history.json', 'w+') as f:
                    json.dump(states_history, f)
                with open(self.output_path+'actions_history.json', 'w+') as f:
                    json.dump(actions_history, f)
                with open(self.output_path+'rewards_history.json', 'w+') as f:
                    json.dump(rewards_history, f)
                with open(self.output_path+'transitions_history.json', 'w+') as f:
                    json.dump(transitions_history, f)

                # Stats
                episode_rewards.append(episode_reward)
                episode_reward = 0
                episode_num += 1
                batch_loss = 0

                self.env.debug()
            
        # Close and finish
        self.env.close()

if __name__ == "__main__":
    import os
    print("Restarting PostgreSQL...")
    os.system('sudo systemctl restart postgresql@12-main')
    
    # agent1 = Agent(env=Environment(window_size=40, shift=False), tag='winsize40_noshift')
    # agent2 = Agent(env=Environment(window_size=80, shift=False), tag='winsize80_noshift')
    # agent1.train()
    # agent2.train()

    # agent3 = Agent(env=Environment(window_size=40, reward_func=4), tag='func4_win40')
    # agent3.train()

    # agent4 = Agent(env=Environment(window_size=80, reward_func=4), tag='func4_win80')
    # agent4.train()

    # agent2 = Agent(env=Environment(reward_func=3), tag='func3')
    # agent2.train()

    # agent1 = Agent(env=Environment(reward_func=2), tag='func2')
    # agent1.train()

    agent_test = Agent(env=Environment(window_size=40, shift=False), tag='winsize40_model40_test_10gb')
    agent_test.test(model_path='results/0.0001_0.9_100000_10000_128_1024_0.01_0.01_winsize40 (BEST)')
