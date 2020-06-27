import json
from pprint import pprint

with open('data/states_history.json', 'r') as f:
    states = json.load(f)

print(states[-2][:47])