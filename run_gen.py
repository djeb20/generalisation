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

num_episodes = 1000

returns = []

for ep in tqdm(range(num_episodes)):

    state = env.reset()
    ret = 0

    while True:

        if ep < 50:
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

@tf.function
def update_critic(self, batch):
    """
    Function to update the critic network
    """
    
    # Giving parts of the batch human understandable names
    old_states = batch[0]
    actions = batch[1]   
    targets = batch[2]
    
    # Everything in tape will have derivative taken
    with tf.GradientTape() as g:
                    
        Q_values = self.critic(old_states, training=True)
        Q_values_actions = tf.gather_nd(Q_values, tf.stack((tf.constant(np.arange(len(Q_values)), dtype=tf.int32), actions), -1))
        targets_actions = tf.gather_nd(targets, tf.stack((tf.constant(np.arange(len(Q_values)), dtype=tf.int32), actions), -1))

        error = Q_values_actions - targets_actions

        # Calculate loss
        loss = tf.math.reduce_mean(tf.math.square(error))

    # Calculates gradient and then optimises using it (moving in that direction)
    gradient = g.gradient(loss, self.critic.trainable_variables)
    self.optimizer_critic.apply_gradients(zip(gradient, self.critic.trainable_variables))

def get_batch(self, b):
    """
    Selects a batch from the buffer.
    Using a selected trajectory.
    """
        
    # Index of the randomly chosen transitions   
    index = np.arange(self.split_ind)[b * self.batch_size:(b+1) * self.batch_size]     

    # We have a different index for values as we need the next one too.
    batch = [self.buffer['obs'][index], 
                self.buffer['action'][index]]

    batch.append(self.Q_table_arr[batch[0][:, 0],
    batch[0][:, 1],
    batch[0][:, 2],
    batch[0][:, 3]])
    
    # Make tensors for speed.
    t_trajectory = [tf.convert_to_tensor(item) for item in batch]
    t_trajectory[1] = tf.cast(t_trajectory[1], dtype=tf.int32)
    t_trajectory[2] = tf.cast(t_trajectory[2], dtype=tf.float32)

    return t_trajectory

# @tf.function
def learn(self):
    """
    Trains the critic using supervised learning.
    """

    self.Q_table_arr = np.zeros((10, 10, 10, 10, self.action_dim))
    for s, v in self.Q_table.items():
        self.Q_table_arr[s[0], s[1], s[2], s[3]] = v

    for _ in range(self.epochs):
        if _ % 100 == 0:
            print('{}/{}'.format(int(_ / 100 + 1), int(self.epochs / 100)))
        for b in range(self.num_batches):

            batch = self.get_batch(b)
            self.update_critic(batch)

        if self.check_val():
            break

        
# @tf.function
def check_val(self):
    """
    Checks whether validation loss has changed.
    """

    # We have a different index for values as we need the next one too.
    batch = [self.buffer['obs'][self.split_ind:], 
                self.buffer['action'][self.split_ind:]]

    batch.append(self.Q_table_arr[batch[0][:, 0],
    batch[0][:, 1],
    batch[0][:, 2],
    batch[0][:, 3]])
    
    # Make tensors for speed.
    batch = [tf.convert_to_tensor(item) for item in batch]
    batch[1] = tf.cast(batch[1], dtype=tf.int32)
    batch[2] = tf.cast(batch[2], dtype=tf.float32)

    # Giving parts of the batch human understandable names
    old_states = batch[0]
    actions = batch[1]   
    targets = batch[2]
            
    Q_values = self.critic(old_states, training=True)
    Q_values_actions = tf.gather_nd(Q_values, tf.stack((tf.constant(np.arange(len(Q_values)), dtype=tf.int32), actions), -1))
    targets_actions = tf.gather_nd(targets, tf.stack((tf.constant(np.arange(len(Q_values)), dtype=tf.int32), actions), -1))

    # Calculate loss
    loss = tf.math.reduce_mean(tf.math.square(Q_values_actions - targets_actions))

    if loss < self.best:

        self.curr = 0
        self.best_weights = self.critic.get_weights()
        self.best = float(loss)

    else:

        self.curr += 1

    if self.curr == self.patience:

        print('Final loss: {:0.05f}'.format(float(self.best)))
        self.critic.set_weights(self.best_weights)
        return True

    else:

        return False


# TEST EPISODES

state = env.reset()