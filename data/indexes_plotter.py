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
    
    training_states = states
    test_states = states[63119:]

    optimal = list()
    total = list()
    for state in test_states:
        optimal.append(sum([state[12], state[20], state[22], state[23], state[28], state[39]]))
        total.append(sum(state[:45]))

    optimal = pd.DataFrame(optimal).rolling(window=5).mean()
    total = pd.DataFrame(total).rolling(window=5).mean()

    plt.plot(total, label="Total indexes")
    plt.plot(optimal, label="Total optimal indexes")
    plt.axhline(6, color='r', linestyle='--', linewidth=0.9, label="Ground truth optimal indexes")
    plt.ylim((0, 25))
    plt.xlabel('Step')
    plt.ylabel('Number of indexes')
    plt.legend(loc='center right')
    plt.tight_layout()
    # plt.show()
    # plt.savefig('training_indexes.pdf')
    plt.savefig('fixed_workload_smartix.pdf')
