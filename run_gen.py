from GridWorld import GridEnv
import numpy as np
from DQN import DQN
from tqdm import tqdm
from supervised_agent import Sup_Agent 
import matplotlib.pyplot as plt
import csv

# TRAIN THE RL AGENT FIRST

env = GridEnv(15, 15, num_hidden=20)
agent = DQN(env.state_dim, 
            env.action_dim,
            critic_arch=[40, 30], 
            buffer_size=10000, 
            batch_size=64,
            gamma=0.99,
            epsilon=0.05,
            step_size=1e-4,
            tau=0.001)

num_episodes = 10000
explore = 100

returns = []
errors = []

for ep in tqdm(range(num_episodes)):

    state = env.reset()
    ret = 0

    while True:

        if ep < explore:
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

    # if ret != env.H - np.abs(env.init - env.goal).sum():

    #     print('Best Return/Return: {}/{}'.format(ret, env.H - np.abs(env.init - env.goal).sum()))

    returns.append(ret)
    errors.append((env.H - np.abs(env.init - env.goal).sum()) - ret)

plt.plot(errors)
plt.xlabel('Episode')
plt.ylabel('Difference in return')
plt.title('Difference in return from optimum for DQN agent')
plt.savefig('errors_plot.svg')

# DISTILL INTO NEW NETWORK USING SUPERVISED LEARNING

# Do I bother with the monitoring the validation set?


state_buffer = []
action_buffer = []
value_buffer = []

all_states = np.array([np.append(s, g) for s in env.goals for g in env.visible_goals])
all_values = agent.critic(all_states).numpy()

for state, q_values in zip(all_states, all_values):

    for action in range(env.action_dim):

        state_buffer.append(state)
        action_buffer.append(action)
        value_buffer.append(q_values[action])

sup_agent = Sup_Agent(env.action_dim, batch_size=agent.batch_size, step_size=agent.step_size)
sup_agent.create_buffer(state_buffer, action_buffer, value_buffer)
sup_agent.critic = agent.make_critic(env.state_dim, env.action_dim, [40, 30])

sup_agent.train()

# TEST EPISODES

# Try each new goal
for goal in env.hidden_goals:

    # Try a bunch of start states
    for _ in range(5):

        state = env.reset_test(goal)
        start = state.copy()

        ret_DQN = 0

        # Play DQN agent
        while True:

            action = agent.choose_action(state, exp=False)
            new_state, reward, done, _ = env.step(action)
            ret_DQN += reward
            state = new_state

            if done: break

        # Play Supervised agent
        state = env.reset_test(goal, start[:2])
        ret_sup = 0

        while True:

            action = sup_agent.choose_action(state, exp=False)
            new_state, reward, done, _ = env.step(action)
            ret_sup += reward
            state = new_state

            if done: break

        results = []
        
        if ret_sup > ret_DQN:

            results.append('Supervised agent did better')
            print('Supervised agent did better')

        elif ret_sup == ret_DQN:

            results.append('There was no difference')
            print('There was no difference')

        else:

            results.append('DQN agent did better')
            print('DQN agent did better')

        print()

np.savetxt('results.csv', results, delimiter=',')



