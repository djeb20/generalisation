from GridWorld import GridEnv
import numpy as np
import sys
from DQN import DQN
from tqdm import tqdm
# import tensorflow as tf
from supervised_agent import Sup_Agent 

# TRAIN THE RL AGENT FIRST

env = GridEnv(10, 10)
agent = DQN(env.state_dim, 
            env.action_dim,
            critic_arch=[40, 30], 
            buffer_size=10000, 
            batch_size=64,
            gamma=0.99,
            epsilon=0.05,
            step_size=1e-4,
            tau=0.001)

num_episodes = 100
explore = 50

returns = []

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

    if ret != env.H - np.abs(env.init - env.goal).sum():

        print('Best Return/Return: {}/{}'.format(ret, env.H - np.abs(env.init - env.goal).sum()))

    returns.append(ret)

# DISTILL INTO NEW NETWORK USING SUPERVISED LEARNING

# # DOES THIS WORK?
# # Do I bother with the monitoring the validation set?

# critic = agent.make_critic(env.state_dim, env.action_dim, critic_arch=[40, 30])

# # Going to have an early stopping function
# callback = tf.keras.callbacks.EarlyStopping(
#     # monitor="val_loss",
#     monitor="loss",
#     patience=20,
#     verbose=1,
#     mode="auto",
#     restore_best_weights=True,
# )

# critic.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=agent.step_size))

# # Train the neural network
# train_data = np.array([np.append(s, g) for s in env.goals for g in env.visible_goals])
# targets = agent.critic(train_data).numpy()

# critic.fit(train_data, 
#         targets, 
#         epochs=10000, 
#         batch_size=agent.batch_size, 
#         # validation_split=0.2, 
#         callbacks=[callback], 
#         verbose=1)

# print(targets)
# print(critic(train_data).numpy())

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

sup_agent = Sup_Agent(batch_size=agent.batch_size, step_size=agent.step_size)
sup_agent.create_buffer(state_buffer, action_buffer, value_buffer)
sup_agent.critic = agent.make_critic(env.state_dim, env.action_dim, [40, 30])

sup_agent.train()

print(all_values)
print(sup_agent.critic(all_states).numpy())

# TEST EPISODES

state = env.reset()