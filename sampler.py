import random
import json
import os
import numpy as np

from environment import Environment


class Agent:
    def __init__(self, env=Environment(), output_path=None):
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

        for step in range(self.n_steps):
            action = random.randrange(self.n_actions)

            next_state, reward, done, _ = self.env.step(action)

            self.memory.append((state, [action], [reward], next_state, [done]))

            state = next_state

            if (step != 0 and step % self.target_update_interval == 0):
                with open('memory_samples.json', 'w+') as f:
                    json.dump(self.memory, f)
                print(step)
                self.env.reset()

        self.env.close()


if __name__ == "__main__":
    import os
    print("Restarting PostgreSQL...")
    os.system('sudo systemctl restart postgresql@12-main')
    
    agent = Agent(env=Environment())
    agent.sample()
