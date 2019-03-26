import pprint
import json

dump = dict()

with open("output_to_parse.out", "r") as infile:
  for line in infile:
    line = line.strip()
    line = line.split(" = ")
    if (line[0] == "episode"):
      episode = line[1]
      step = 0
      dump[episode] = dict()
      dump[episode][step] = dict()
    elif (line[0] == "action_type"):
      dump[episode][step][line[0]] = line[1]
    elif (line[0] == "action"):
      dump[episode][step][line[0]] = line[1]
    elif (line[0] == "reward"):
      dump[episode][step][line[0]] = float(line[1])
    elif (line[0] == "state"):
      dump[episode][step][line[0]] = line[1]
      # print(line)
    elif (line[0] == "td_target"):
      dump[episode][step][line[0]] = float(line[1])
    elif (line[0] == "q_value"):
      dump[episode][step][line[0]] = float(line[1])
    elif (line[0] == "td_error"):
      dump[episode][step][line[0]] = float(line[1])
    elif (line[0] == "max_a"):
      dump[episode][step][line[0]] = float(line[1])
      step += 1
      if step != 100:
        dump[episode][step] = dict()

# pprint.pprint(dump)

with open('dump_agent_iterations.json', 'w') as outfile:
    json.dump(dump, outfile)