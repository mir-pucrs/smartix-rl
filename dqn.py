import gym
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pprint import pprint
from environment import Environment


class ReplayMemory():
    def __init__(self, memory_size):
        self.memory = collections.deque(maxlen=memory_size)

    def put(self, transition):
        self.memory.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.memory, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), \
            torch.tensor(a_lst), \
            torch.tensor(r_lst), \
            torch.tensor(s_prime_lst, dtype=torch.float), \
            torch.tensor(done_mask_lst)

    def size(self):
        return len(self.memory)


class QNetwork(nn.Module):
    """
        HAVE TO CHANGE in/out sizes
    """
    def __init__(self, input_size=45, output_size=25):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DQNAgent():
    """
        HAVE TO CHANGE THE ENVIRONMENT!!!
    """
    def __init__(self, replay_memory=ReplayMemory(50000), qnet_input_size=45, qnet_output_size=45, env=Environment()):
        self.env = env
        self.replay_memory = replay_memory
        self.q = QNetwork(qnet_input_size, qnet_output_size)
        self.q_target = QNetwork(qnet_input_size, qnet_output_size)
        self.q_target.load_state_dict(self.q.state_dict())

        # Hyperparameters
        self.learning_rate = 0.0005
        self.gamma  = 0.9
        self.epsilon = 0.08
        self.batch_size = 16
        
    def sample_action(self, state):
        out = self.q(state)
        coin = random.random()
        if coin < self.epsilon:
            return random.randint(0, 1)
        else: 
            return out.argmax().item()


    def optimize_model(self, q, q_target, memory, optimizer):
        if self.replay_memory.size() > self.batch_size:
            for _ in range(self.batch_size):
                s, a, r, s_prime, done_mask = memory.sample(self.batch_size)
                
                q_out = q(s)
                q_a = q_out.gather(1,a)
                max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
                target = r + self.gamma * max_q_prime * done_mask

                loss = F.smooth_l1_loss(q_a, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    

    def train(self):
        change_params_interval = 16
        accum_reward = 0.0  
        optimizer = optim.Adam(self.q.parameters(), lr=self.learning_rate)

        print(self.env)

        for episode in range(50):
            self.epsilon = max(0.01, 0.08 - 0.01*(episode/200)) #Linear annealing from 8% to 1%
            s = self.env.reset()

            for step in range(22):

                a = self.sample_action(torch.from_numpy(s).float())
                
                s_prime, r, done, info = self.env.step(a)

                print(step, "\tState\t", s)
                print(step, "\tAction\t", a)
                print(step, "\tReward\t", r)
                print(step, "\tS_prime\t", s_prime)

                done_mask = 0.0 if done else 1.0
                self.replay_memory.put((s,a,r/100.0,s_prime, done_mask))

                s = s_prime

                accum_reward += r

                if done:
                    break
                
                # break
                # self.env.render()
                
                # Perform one step of the optimization (on the target network)
                self.optimize_model(self.q, self.q_target, self.replay_memory, optimizer)

            # break

            if episode%change_params_interval == 0 and episode != 0:
                self.q_target.load_state_dict(self.q.state_dict())
                print("# of episode :{}, avg accum_reward : {:.1f}, buffer size : {}, epsilon : {:.1f}%".format(
                                                                episode, accum_reward/change_params_interval, self.replay_memory.size(), self.epsilon*100))
                accum_reward = 0.0
        
        self.env.close()


if __name__ == "__main__":
    agent = DQNAgent()
    agent.train()
