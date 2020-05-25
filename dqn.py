import collections
import random
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DQNAgent():
    def __init__(self, replay_memory=ReplayMemory(50000), env=Environment()):
        self.env = env
        self.replay_memory = replay_memory
        self.q = QNetwork(self.env.n_features, self.env.n_actions)
        self.q_target = QNetwork(self.env.n_features, self.env.n_actions)
        self.q_target.load_state_dict(self.q.state_dict())

        # Hyperparameters
        self.learning_rate = 0.0005
        self.gamma  = 0.99
        self.epsilon = 1.0
        self.batch_size = 16
        
    def sample_action(self, state):
        out = self.q(state)
        coin = random.random()
        if coin < self.epsilon:
            rand = random.randint(0, self.env.n_actions-1)
            print("Random action:", rand)
            return rand
        else: 
            argmax = out.argmax().item()
            print("Argmax action:", argmax)
            return argmax


    def optimize_model(self, q, q_target, memory, optimizer):
        if self.replay_memory.size() > self.batch_size:
            for _ in range(self.batch_size):
                s, a, r, s_prime, done_mask = memory.sample(self.batch_size)
                
                q_out = q.forward(s)
                q_a = q_out.gather(1,a)
                max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
                target = r + self.gamma * max_q_prime * done_mask

                loss = F.smooth_l1_loss(q_a, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    

    def train(self):
        change_params_interval = 22
        optimizer = optim.Adam(self.q.parameters(), lr=self.learning_rate)

        reward_list = list()

        s = self.env.reset()

        for step in range(10000):
            print("\n-- Step", step)

            a = self.sample_action(torch.from_numpy(s).float())

            s_prime, r, done, info = self.env.step(a)

            print("State\t", str(s).replace("\n", "").replace(" ", ""))
            print("Action\t", a)
            print("Reward\t", r)
            print("S_prime\t", str(s_prime).replace("\n", "").replace(" ", ""))
            reward_list.append(r)

            done_mask = 0.0 if done else 1.0
            self.replay_memory.put((s,a,r/100.0,s_prime, done_mask))

            s = s_prime
            
            # Perform one step of the optimization (on the target network)
            self.optimize_model(self.q, self.q_target, self.replay_memory, optimizer)

            print("contaloca", step+1%change_params_interval)
            if (step+1) % change_params_interval == 0:
                self.q_target.load_state_dict(self.q.state_dict())
            
                avg_reward = sum(reward_list[-change_params_interval:])/change_params_interval
                print("\nStep:{}, Avg. reward: {:.2f}, Memory size: {}, Epsilon: {:.2f}%".format(
                    step, avg_reward, self.replay_memory.size(), self.epsilon))
            
                with open('data.txt', 'a+') as f:
                    f.write(str(step) + '\t' + str(self.epsilon) + '\t' + str(avg_reward) + '\n')
                
                with open('reward_list.json', 'w+') as f:
                    json.dump(reward_list, f, indent=4)

                self.epsilon = self.epsilon * 0.9

        self.env.close()


if __name__ == "__main__":
    agent = DQNAgent()
    agent.train()
