import gym
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)
    

    def size(self):
        return len(self.memory)


class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, 2)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DQNAgent():
    def __init__(self, replay_memory=ReplayMemory(50000), q=QNetwork(), q_target=QNetwork(), env=gym.make('CartPole-v1')):
        self.replay_memory = replay_memory
        self.q = q
        self.q_target = q_target
        self.env = env

        # Hyperparameters
        self.learning_rate = 0.0005
        self.gamma  = 0.9
        self.batch_size = 32
        self.epsilon = 0.08

        self.q_target.load_state_dict(q.state_dict())
    

    def sample_action(self, state):
        # print("Sampling action...")
        out = self.q.forward(state)
        # print("Out:", out)
        coin = random.random()
        if coin < self.epsilon:
            # THE RANDOM SHOULD BE 0 or 1 cause there are only two actions
            # print("Random action")
            return random.randint(0,1)
        else: 
            # print("Out argmax:", out.argmax().item())
            return out.argmax().item()


    def optimize_model(self, q, q_target, memory, optimizer):
        if self.replay_memory.size() > self.batch_size:
            for _ in range(10):
                s, a, r, s_prime, done_mask = memory.sample(self.batch_size)

                # print("s", s)
                # print("a", a)
                # print("r", r)
                # print("s'", s_prime)
                # print("done_mask", done_mask)

                # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
                # columns of actions taken. These are the actions which would've been taken
                # for each batch state according to policy_net
                # state_action_values = policy_net(state_batch).gather(1, action_batch)
                q_out = q(s)
                # print("q_out", q_out)
                q_a = q_out.gather(1,a)
                # print("q_a", q_a)
                max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
                target = r + self.gamma * max_q_prime * done_mask
                loss = F.smooth_l1_loss(q_a, target)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    

    def train(self):
        print_interval = 20
        score = 0.0  
        optimizer = optim.Adam(self.q.parameters(), lr=self.learning_rate)

        print(self.env)

        for episode in range(500):
            self.epsilon = max(0.01, 0.08 - 0.01*(episode/200)) #Linear annealing from 8% to 1%
            s = self.env.reset()

            # for step in range(50):
            for step in range(10):
                
                # Action: int
                # State: numpy array
                # Reward: float
                # Info: empty dict

                a = self.sample_action(torch.from_numpy(s).float())
                
                s_prime, r, done, info = self.env.step(a)

                print(step, "\tAction\t",   type(a),    "\t\t\t",   a)
                print(step, "\tS_prime\t",  type(s),    "\t",       s, s.shape)
                print(step, "\tReward\t",   type(r),    "\t\t",     r)
                print(step, "\tInfo\t",     type(info), "\t\t",     info)

                break

                done_mask = 0.0 if done else 1.0
                self.replay_memory.put((s,a,r/100.0,s_prime, done_mask))

                s = s_prime

                score += r

                if done:
                    break
            
                # Perform one step of the optimization (on the target network)
                self.optimize_model(self.q, self.q_target, self.replay_memory, optimizer)

                self.env.render()

            break

            if episode%print_interval==0 and episode!=0:
                self.q_target.load_state_dict(self.q.state_dict())
                print("# of episode :{}, avg score : {:.1f}, buffer size : {}, epsilon : {:.1f}%".format(
                                                                episode, score/print_interval, self.replay_memory.size(), self.epsilon*100))
                score = 0.0
        
        self.env.close()


if __name__ == "__main__":
    agent = DQNAgent()
    agent.train()
