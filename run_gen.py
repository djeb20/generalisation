from GridWorld import GridEnv
import numpy as np
import sys
from DQN import DQN
from tqdm import tqdm

env = GridEnv(7, 7)
agent = DQN(env.state_dim, 
            env.action_dim,
            critic_arch=[40, 30], 
            buffer_size=10000, 
            batch_size=64,
            gamma=0.99,
            epsilon=0.05,
            step_size=1e-4,
            tau=0.001)

num_episodes = 1000000

returns = []

for ep in tqdm(range(num_episodes)):

    state = env.reset()
    ret = 0

    while True:

        if ep < 100:
            # At first only explore.

            action = np.random.randint(env.action_dim)

        else:

            action = agent.choose_action(state)

        new_state, reward, done, _ = env.step(action)

        agent.store((state, action, reward, new_state, 1 - done))

        batch = agent.get_batch()
        agent.update_critic(*batch)
        agent.update_target()

        state = new_state

        ret += reward

        if done: break

    returns.append(ret)
