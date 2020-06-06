import json
import matplotlib.pyplot as plt

with open('../output/rewards.json', 'r') as f:
    rewards = json.load(f)

plt.plot(rewards)
plt.show()