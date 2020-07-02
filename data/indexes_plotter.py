import os
import json
import matplotlib.pyplot as plt
from pprint import pprint

# Plots
# 1 - Optimal indexes and total sum of indexes (with a ground truth line)
# 2 - Reward history while training
# 3 - Loss while training

# Optimal indexes:
# 12, 20, 22, 23, 28, 39

outputs = [folder[0] for folder in os.walk('output/')][1:]

for folder in outputs:
    # with open(folder+'/rewards_history.json', 'r') as f:
    #     rewards = json.load(f)
    #     plt.title(folder)
    #     plt.plot(rewards)
    #     plt.show()
    with open(folder+'/states_history.json', 'r') as f:
        states = json.load(f)
        states = states[:50000]
        optimal = list()
        total = list()
        for state in states:
            optimal.append(sum([state[12], state[20], state[22], state[23], state[28], state[39]]))
            total.append(sum(state[:45]))
        plt.title(folder)
        plt.plot(optimal)
        plt.plot(total)
        plt.axhline(6, color='r', linestyle='--')
        plt.show()
