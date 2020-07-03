import os
import json
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint

# Plots
# 1 - Optimal indexes and total sum of indexes (with a ground truth line)
# 2 - Reward history while training
# 3 - Loss while training

# Optimal indexes:
# 12, 20, 22, 23, 28, 39

folder = 'results/0.0001_0.9_100000_10000_128_1024_0.01_0.01_winsize40 (BEST)'

with open(folder+'/states_history.json', 'r') as f:
    states = json.load(f)
    states = states[:64000]
    optimal = list()
    total = list()
    for state in states:
        optimal.append(sum([state[12], state[20], state[22], state[23], state[28], state[39]]))
        total.append(sum(state[:45]))
    optimal = pd.DataFrame(optimal).rolling(window=10).mean()
    total = pd.DataFrame(total).rolling(window=10).mean()
    plt.title(folder)
    plt.plot(total)
    plt.plot(optimal)
    plt.axhline(6, color='r', linestyle='--')
    plt.show()
