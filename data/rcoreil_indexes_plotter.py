import os
import json
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint

# Optimal indexes:
# 12, 20, 22, 23, 28, 39
# ['c_acctbal', 'p_brand', 'p_size', 'p_container', 'o_orderdate', 'l_shipdate']

folder = 'data'

with open(folder+'/teste_rcoreil.txt', 'r') as f:
    states = f.read().splitlines()

    train_states = states[:990]
    test_states = states[991:]
    
    optimals = ['c_acctbal', 'p_brand', 'p_size', 'p_container', 'o_orderdate', 'l_shipdate']
    optimal = list()
    total = list()

    for state in train_states:
        state = json.loads(state)["state"]
        state = list(set(state))

        optimal_sum = 0
        for col in optimals:
            if col in str(state).lower():
                optimal_sum += 1
                continue
        
        optimal.append(optimal_sum)
        total.append(len(state))

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
    plt.savefig('fixed_workload_rcoreil.pdf')
