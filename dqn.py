import random
import math
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

from pprint import pprint

class ReplayMemory():
    def __init__(self, capacity):
        self.memory = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.memory)

    def add(self, s0, a, r, s1, done):
        self.memory.append((np.expand_dims(s0, 0), a, r, np.expand_dims(s1, 0), done))

    def sample(self, batch_size):
        s0, a, r, s1, done = zip(*random.sample(self.memory, batch_size))
        s0 = torch.tensor(np.concatenate(s0), dtype=torch.float)
        s1 = torch.tensor(np.concatenate(s1), dtype=torch.float)
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
    def __init__(self, env=Environment(), output_path=str(time.time())):
        # Log
        path = os.path.join('output', output_path)
        if not os.path.isdir(path): os.makedirs(path)
        self.output_path = path + '/'

        # Hyperparameters
        self.gamma = 0.9
        self.alpha = 1e-4
        self.beta = 0.01
        self.avg_reward = 0

        # Training
        self.n_steps = 100000  # 100k
        self.memory_size = 10000  # 10k
        self.memory = ReplayMemory(self.memory_size)
        self.target_update_interval = 128
        self.batch_size = 256

        # Epsilon
        self.epsilon = 1  # 100%
        self.epsilon_min = 0.01  # 1%
        self.epsilon_decay = 0.05  # 10%

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
            self.argmax_cnt += 1
        else:
            action = random.randrange(self.n_actions)
            self.random_cnt += 1
        return action

    def learn_batch(self):
        losses = 0
        for _ in range(3):
            s0, a, r, s1, done = self.memory.sample(self.batch_size)

            # Normal DDQN update
            q_values = self.qnet(s0)
            q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
            # Double Q-learning
            online_next_q_values = self.qnet(s1)
            _, max_indicies = torch.max(online_next_q_values, dim=1)
            target_q_values = self.qnet_target(s1)
            next_q_value = torch.gather(target_q_values, 1, max_indicies.unsqueeze(1))

            expected_q_value = r + self.gamma * next_q_value.squeeze()
            # expected_q_value = r - self.avg_reward + next_q_value.squeeze()

            loss = (q_value - expected_q_value.data).pow(2).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses += loss.item()
        return losses
    
    def train(self, pre_fr=0):
        # Stats
        loss_history = list()
        loss = 0
        states_history = list()
        rewards_history = list()
        episode_rewards = [0]
        episode_reward = 0
        episode_num = 0
        rewards = list()

        # Debug
        self.argmax_cnt = 0
        self.random_cnt = 0

        # Start plot
        plt.ion()
        plt.show()
        plt.pause(0.001)

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

            # Update average reward
            # q_target = np.max(self.qnet_target(torch.tensor(np.concatenate(np.expand_dims(next_state, 0)), dtype=torch.float)).detach().numpy())
            # q_value = self.qnet(torch.tensor(np.concatenate(np.expand_dims(state, 0)), dtype=torch.float)).detach().numpy()[action]
            # delta = reward - self.avg_reward + (q_target - q_value)
            # self.avg_reward += self.beta*delta

            # Add to replay memory
            self.memory.add(state, action, reward, next_state, done)

            # Update state
            state = next_state

            # Stats
            episode_reward += reward
            rewards_history.append(reward)
            states_history.append(next_state.tolist())

            # Learn
            if len(self.memory) > self.batch_size:
                loss = self.learn_batch()
                loss_history.append(loss)

            # Save interval
            if (step != 0 and step % self.target_update_interval == 0):
                # Update step time
                end = time.time()
                elapsed = end - start
                start = time.time()

                # Print stats
                if episode_reward > max(episode_rewards): last = '!'
                elif episode_reward >= episode_rewards[-1]: last = '+' 
                else: last = '-'
                print("episode: %2d \t acc_reward: %10.3f  %s \t loss: %8.3f \t elapsed: %6.2f \t epsilon: %2.4f" % (episode_num, episode_reward, last, loss, float(elapsed), self.epsilon))
                
                # Save logs
                log = "%2d\t%8.3f\t%s\t%8.3f\t%.2f\t%.4f\n" % (episode_num, episode_reward, last, loss, elapsed, self.epsilon)
                with open(self.output_path+'log.txt', 'a+') as f:
                    f.write(log)
                with open(self.output_path+'rewards_history.json', 'w+') as f:
                    json.dump(rewards_history, f)
                with open(self.output_path+'states_history.json', 'w+') as f:
                    json.dump(states_history, f)

                # Stats
                episode_rewards.append(episode_reward)
                episode_reward = 0
                episode_num += 1

                # Plot
                plt.plot(episode_rewards[-(len(episode_rewards)-1):])
                plt.draw()
                plt.pause(0.001)

                # Epsilon decay
                self.epsilon -= self.epsilon * self.epsilon_decay
                if self.epsilon < self.epsilon_min: self.epsilon = self.epsilon_min

                # Update target weights
                self.qnet_target.load_state_dict(self.qnet.state_dict())

                # Save model checkpoint
                self.save_model()

                # self.env.reset()
                self.env.debug()
                print('ARGMAX', self.argmax_cnt)
                print('RANDOM', self.random_cnt)
                print('')
                self.argmax_cnt = 0
                self.random_cnt = 0

                # self.env.reset()

        # Close and finish
        self.env.close()


if __name__ == "__main__":
    import os
    print("Restarting PostgreSQL...")
    os.system('sudo systemctl restart postgresql@12-main')

    agent = Agent(env=Environment(allow_columns=True, flip=False), output_path='with_columns')
    agent.train()