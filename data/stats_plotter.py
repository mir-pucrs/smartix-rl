import os
import json
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint

filename = 'results/0.0001_0.9_100000_10000_128_1024_0.01_0.01_winsize40 (BEST)/log.txt'

log = pd.read_table(filename, header=None, names=['episode', 'reward', 'loss', 'time', 'epsilon'])

rewards = log['reward'][:500] #.rolling(window=10).mean()
loss = log['loss'][:500] #.rolling(window=10).mean()
epsilon = log['epsilon'][:500] #.rolling(window=10).mean()

plt.plot(rewards)
plt.ylabel('Accumulated reward')
plt.xlabel('Step')
plt.tight_layout()
rewards_fig = plt.gcf()
default_size = rewards_fig.get_size_inches()
rewards_fig.set_size_inches((default_size[0]*2, default_size[1]))
# plt.show()
plt.tight_layout()
plt.savefig('rewards.pdf')
rewards_fig.set_size_inches((default_size[0], default_size[1]))
plt.clf()

plt.plot(loss)
plt.ylabel('Accumulated loss')
plt.xlabel('Step')
plt.tight_layout()
# plt.show()
plt.savefig('loss.pdf')
plt.clf()

plt.plot(epsilon)
plt.ylabel('Epsilon value')
plt.xlabel('Step')
plt.tight_layout()
# plt.show()
plt.savefig('epsilon.pdf')
plt.clf()
