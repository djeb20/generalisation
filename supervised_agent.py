import tensorflow as tf
import numpy as np
import keras
from keras.layers import Dense
from keras.layers import Input

class Sup_Agent:

    def __init__(self, state_dim, action_dim, critic_arch=[40, 30], 
    buffer_size=int(1e6), batch_size=64, gamma=0.99, 
    step_size=1e-4):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.step_size = step_size

        self.epochs = 10000
        self.val_split = 0.8
        # CHECK SPLIT IS RIGHT
        self.split_ind = int(self.val_split * self.buffer_size)
        self.num_batches = (self.split_ind // self.batch_size) + 1

        # Used for validation checks
        self.best = 1e6
        # THIS SHOULD BE HIGHER
        self.patience = 25

        self.create_buffer()
        self.critic = self.make_critic()

        # Step the optimiser used and its stepsize.
        self.optimizer_critic = tf.keras.optimizers.Adam(step_size)

    def create_buffer(self):
        """
        Creates a new memory based on the number of transitions being recorded
        """
        
        # Count to keep track of size of memory
        self.count = 0
        
        # Initiate empty buffer
        # Actions are always single number in a discrete setting?
        self.buffer = {'obs': np.empty((self.buffer_size, self.state_dim)),
                       'action': np.empty((self.buffer_size, )),
                       'value': np.empty((self.buffer_size, ))}

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

    def choose_action(self, state):
        """
        Given a state returns a chosen action given by policy.
        Has epsilon greedy exploration.
        """

        values = self.critic(state.reshape(1, -1))[0].numpy()

        return np.random.choice(np.arange(self.action_dim)[values == values.max()])