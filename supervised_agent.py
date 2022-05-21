import tensorflow as tf
import numpy as np
from tqdm import tqdm

class Sup_Agent:

    def __init__(self, action_dim, batch_size=64, step_size=1e-4, epsilon=0.05):

        self.batch_size = batch_size
        self.step_size = step_size
        self.epsilon = epsilon
        self.action_dim = action_dim

        self.epochs = 10000
        self.val_split = 1

        # Used for validation checks
        self.best = 1e6
        # THIS SHOULD BE HIGHER
        self.patience = 100

        # Step the optimiser used and its stepsize.
        self.optimizer_critic = tf.keras.optimizers.Adam(step_size)

    def create_buffer(self, state_buffer, action_buffer, value_buffer):
        """
        Creates a new memory based on the number of transitions being recorded
        """
        
        # Count to keep track of size of memory
        self.count = 0
        
        # Initiate empty buffer
        # Actions are always single number in a discrete setting?
        self.buffer = {'obs': np.array(state_buffer), 
                       'action': np.array(action_buffer), 
                       'value': np.array(value_buffer)}

        self.buffer_size = len(state_buffer)

        # CHECK SPLIT IS RIGHT
        self.split_ind = int(self.val_split * self.buffer_size)
        self.num_batches = (self.split_ind // self.batch_size) + 1

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

            error = Q_values_actions - targets

            # Calculate loss
            loss = tf.math.reduce_mean(tf.math.square(error))

        # Calculates gradient and then optimises using it (moving in that direction)
        gradient = g.gradient(loss, self.critic.trainable_variables)
        self.optimizer_critic.apply_gradients(zip(gradient, self.critic.trainable_variables))

    def get_batch(self, shuffled_indexes, b):
        """
        Selects a batch from the buffer.
        Using a selected trajectory.
        """

        # SHOULD I BE SAMPLING WITH REPLACEMENT
        # DO I WORK MY WAY THROUGH THE ENTIRE DATASET EACH EPOCH?
            
        # Index of the randomly chosen transitions
        index = shuffled_indexes[b * self.batch_size:(b+1) * self.batch_size]  
        # index = np.random.choice(np.arange(self.split_ind), self.batch_size)   

        # We have a different index for values as we need the next one too.
        batch = [self.buffer['obs'][index], 
                 self.buffer['action'][index],
                 self.buffer['value'][index]]
        
        # Make tensors for speed.
        t_trajectory = [tf.convert_to_tensor(item) for item in batch]
        t_trajectory[1] = tf.cast(t_trajectory[1], dtype=tf.int32)
        t_trajectory[2] = tf.cast(t_trajectory[2], dtype=tf.float32)

        return t_trajectory

    # @tf.function
    def train(self):
        """
        Trains the critic using supervised learning.
        """

        for _ in tqdm(range(self.epochs)):

            shuffled_indexes = np.random.choice(np.arange(self.split_ind), self.split_ind, replace=False)

            for b in range(self.num_batches):

                batch = self.get_batch(shuffled_indexes, b)
                self.update_critic(batch)

            if self.check_val():
                break

            
    # @tf.function
    def check_val(self):
        """
        Checks whether validation loss has changed.
        """

        # We have a different index for values as we need the next one too.
        # batch = [self.buffer['obs'][self.split_ind:], 
        #          self.buffer['action'][self.split_ind:],
        #          self.buffer['value'][self.split_ind:]]

        # For early stopping not using validation set
        batch = [self.buffer['obs'], 
                 self.buffer['action'],
                 self.buffer['value']]

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

        # Calculate loss
        loss = tf.math.reduce_mean(tf.math.square(Q_values_actions - targets))

        if loss < self.best:

            print(float(loss))

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

    def choose_action(self, state, exp=True):
        """
        Given a state returns a chosen action given by policy.
        Has epsilon greedy exploration.
        """

        if (np.random.rand() < self.epsilon) & exp:

            return np.random.randint(self.action_dim)

        else:

            return self.critic(state.reshape(1, -1))[0].numpy().argmax()