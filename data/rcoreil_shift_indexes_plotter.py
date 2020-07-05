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
# ['c_acctbal', 'p_brand', 'p_size', 'p_container', 'o_orderdate', 'l_shipdate']

folder = 'data'

shift_at = 220

with open(folder+'/teste_rcoreil.txt', 'r') as f:
    states = f.read().splitlines()
    states = states[991:]
    states = states[440:]
    last = states[-1]
    for _ in range(440):
        states.append(last)
    optimal = list()
    total = list()

    optimal_before_shift = ['p_size', 'o_orderdate', 'l_shipdate']
    optimal_after_shift = ['c_acctbal', 'p_brand', 'p_container']
    shift_count = 0

    for step, state in enumerate(states):
        state = json.loads(state)["state"]
        state = list(set(state))

        if step % shift_at == 0:
            if shift_count % 2 == 0:
                curr_optimal = optimal_before_shift
            else:
                curr_optimal = optimal_after_shift
            shift_count += 1

        optimal_sum = 0
        for col in curr_optimal:
            if col in str(state).lower():
                optimal_sum += 1
                continue
        
        optimal.append(optimal_sum)
        total.append(len(state))

    optimal = pd.DataFrame(optimal).rolling(window=5).mean()
    total = pd.DataFrame(total).rolling(window=5).mean()

    plt.axvline(shift_at, color='g', linestyle='--', linewidth=0.7, label="Workload shifts")
    plt.axvline(shift_at*2, color='g', linestyle='--', linewidth=0.7)
    plt.axvline(shift_at*3, color='g', linestyle='--', linewidth=0.7)
    plt.axhline(3, color='r', linestyle='--', linewidth=0.9, label="Ground truth optimal indexes")
    plt.plot(total, label="Total indexes")
    plt.plot(optimal, label="Total optimal indexes")
    plt.ylim((0, 25))
    plt.xlabel('Step')
    plt.ylabel('Number of indexes')
    plt.legend(loc='center right')
    plt.tight_layout()
    # plt.show()
    plt.savefig('shifting_workload_rcoreil.pdf')
