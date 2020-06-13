import os
import json
import matplotlib.pyplot as plt
from pprint import pprint

# 12
# 20
# 22
# 23
# 28
# 39

outputs = [folder[0] for folder in os.walk('output/')][1:]

for folder in outputs:
    # with open(folder+'/rewards_history.json', 'r') as f:
    #     rewards = json.load(f)
    #     plt.title(folder)
    #     plt.plot(rewards)
    #     plt.show()
    with open(folder+'/states_history.json', 'r') as f:
        states = json.load(f)
        states = states[31100:]
        total = [0] * 45
        for state in states:
            total = [sum(col) for col in zip(total, state)]
        print(str(total))
        print(folder, total[12], total[20], total[22], total[23], total[28], total[39])
        print("")