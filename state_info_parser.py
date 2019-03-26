import json
import pprint

with open("data/state_info.json", "r") as read_file:
    data = json.load(read_file)

pprint.pprint(data)