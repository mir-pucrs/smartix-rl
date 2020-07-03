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

folder = 'output/0.0001_0.9_100000_10000_128_1024_0.01_0.05_winsize40_test'

shift_at = 220

with open(folder+'/states_history.json', 'r') as f:
    states = json.load(f)
    states = states[:880]
    optimal = list()
    total = list()

    # VER CERTO QUAIS SAO ESSES INDICES!!!!!!!!!!!!!!
    optimal_before_shift = [22, 28, 39]
    optimal_after_shift = [12, 20, 23]
    shift_count = 0

    for step, state in enumerate(states):
        
        if step % shift_at == 0:
            if shift_count % 2 == 0:
                curr_optimal = optimal_before_shift
            else:
                curr_optimal = optimal_after_shift
            shift_count += 1
        optimal.append(sum([state[idx] for idx in curr_optimal]))
        total.append(sum(state[:45]))
    
    optimal = pd.DataFrame(optimal).rolling(window=10).mean()
    total = pd.DataFrame(total).rolling(window=10).mean()

    plt.title(folder)
    plt.plot(total, label='total')
    plt.plot(optimal, label='optimal')
    plt.axhline(3, color='r', linestyle='--')
    plt.axvline(shift_at, color='c', linestyle='--')
    plt.axvline(shift_at*2, color='c', linestyle='--')
    plt.axvline(shift_at*3, color='c', linestyle='--')
    plt.legend(loc="upper left")
    plt.show()
