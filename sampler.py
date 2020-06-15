import random
import json
import os
import numpy as np

from environment import Environment


class Agent:
    def __init__(self, env=Environment(), output_path=None):
        self.output_path = output_path
        
        # Training
        self.n_steps = 100000  # 100k
        self.memory = list()
        self.target_update_interval = 500

        # Environment
        self.env = env
        self.n_actions = self.env.n_actions

    """
        Training
    """
    def sample(self):
        state = self.env.reset()

        import time
        start = time.time()

        for step in range(self.n_steps):
            action = random.randrange(self.n_actions)

            next_state, reward, done, _ = self.env.step(action)

            self.memory.append((state.tolist(), [action], [reward], next_state.tolist(), [done]))

            state = next_state

            if (step != 0 and step % self.target_update_interval == 0):
                with open('{}_samples.json'.format(self.output_path), 'w+') as f:
                    json.dump(self.memory, f)
                print("")
                print(time.time() - start)
                print(step)
                self.env.reset()
                break

        self.env.close()


if __name__ == "__main__":
    import os
    print("Restarting PostgreSQL...")
    os.system('sudo systemctl restart postgresql@12-main')

    agent1 = Agent(env=Environment(allow_columns=False, flip=False, window_size=40), output_path='env1')
    agent2 = Agent(env=Environment(allow_columns=False, flip=False, window_size=80), output_path='env2')

    agent3 = Agent(env=Environment(allow_columns=True, flip=False, window_size=40), output_path='env3')
    agent4 = Agent(env=Environment(allow_columns=True, flip=False, window_size=80), output_path='env4')

    agent5 = Agent(env=Environment(allow_columns=False, flip=True, window_size=40), output_path='env5')
    agent6 = Agent(env=Environment(allow_columns=False, flip=True, window_size=80), output_path='env6')
    
    agent7 = Agent(env=Environment(allow_columns=True, flip=True, window_size=40), output_path='env7')
    agent8 = Agent(env=Environment(allow_columns=True, flip=True, window_size=80), output_path='env8')

    agent1.sample()
    agent2.sample()
    agent3.sample()
    agent4.sample()
    agent5.sample()
    agent6.sample()
    agent7.sample()
    agent8.sample()
