import os
import json
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint

filename = 'results/0.0001_0.9_100000_10000_128_1024_0.01_0.01_winsize40 (BEST)/log.txt'

log = pd.read_table(filename, header=None, names=['episode', 'reward', 'loss', 'time', 'epsilon'])

rewards_rolling = log['reward'][:500].rolling(window=10).mean()

plt.plot(rewards_rolling)
plt.show()
