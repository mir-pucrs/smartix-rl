import os
import json
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint

# Optimal indexes:
# 12, 20, 22, 23, 28, 39
# ['c_acctbal', 'p_brand', 'p_size', 'p_container', 'o_orderdate', 'l_shipdate']

folder = 'output/1593741428.3491588_0.0001_0.9_100000_10000_128_1024_0.01_0.01_winsize40_test'
# folder = 'output/1593741711.5689929_0.0001_0.9_100000_10000_128_1024_0.01_0.01_winsize40_test'

shift_at = 220

with open(folder+'/states_history.json', 'r') as f:
    states = json.load(f)
    states = states[:880]
    optimal = list()
    total = list()

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
    
    optimal = pd.DataFrame(optimal).rolling(window=6).mean()
    total = pd.DataFrame(total).rolling(window=6).mean()

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
    plt.savefig("shifting_workload_smartix.pdf")
