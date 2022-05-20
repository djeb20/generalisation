"""
Maze environment.

Primitive action space:

Up, Down, Left, Right.

State Space:

Initial state is randomly sampled.
Terminal space is a randomly sampled goal.
State is the agent's position.

Reward structure:

Reward of +1 is given for reaching goal state.
"""

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from matplotlib import colors

class GridEnv:
    
    def __init__(self, H, W, num_hidden=5):
        """
        The number in the grid representes what it contains:
        
        0 is an empty space
        1 is a wall
        """
            
        # Height and width of the maze.
        self.H = H
        self.W = W
        
        # Save an action dictionary for moves
        self.action_dict = {0 : np.array([-1, 0]),
                            1 : np.array([0, 1]),
                            2 : np.array([1, 0]),
                            3 : np.array([0, -1])}
        
        # Create grid
        self.env = np.zeros((self.H, self.W))
        
        # Fill in a boarder - code is simpler this way
        for i in range(self.env.shape[0]):
            for j in range(self.env.shape[1]):
                
                if (i % (self.H - 1) == 0) or (j % (self.W - 1) == 0):
                    self.env[i][j] = 1
        
        # For rendering
        self.env_mask = np.copy(self.env)
        
        self.action_dim = len(self.action_dict)
        self.state_dim = 2

        # Every possible goal
        goals = [[i, j] for i in range(1, self.H - 1) for j in range(1, self.W -  1)]
        index = np.random.choice(np.arrange(len(goals)), len(goals), replace=False)

        # These are the goals that we restrict and ones we don't
        self.hidden_goals = np.array(goals[index[:num_hidden]])
        self.visible_goals = np.array(goals[index[num_hidden:]])
        self.goals_len = len(goals)
        self.vis_len = self.goals_len - num_hidden   
                
    def reset(self):
        """
        Resets the enviroment, usually called after an episode terminates.
        """

        self.goal = self.visible_goals[np.random.randint(self.vis_len)]

        while True:

            self.pos = self.goals[np.random.randint(self.goals_len)]

            if (self.pos != self.goal).all(): break
                
        return self.pos
        
    def step(self, action):
        """
        Takes a step in the environment
        """
        
        action_v = self.action_dict[action]
        
        new_pos = self.pos + action_v
        
        # No reward unless goal found
#         reward = -.02
        reward = 0
        
        # Assume episode has not ended
        done = False
        
        if self.env[new_pos[0], new_pos[1]] == 0:
            # Enters an empty space
            true_pos = new_pos
        
        elif self.env[new_pos[0], new_pos[1]] == 1:
            # Hits a wall
            true_pos = self.pos
                    
        # Set new position
        self.pos = true_pos
        
        return self.pos, reward, done, False     
        
    def render(self, name=''):
        """ 
        This renders the gridworld envrionment, adapted from Joshua's code 
        """
        
        # Turn interactive mode on.
        plt.ion()
        fig = plt.figure(num = "env_render", figsize=(10, 10 * self.env_mask.shape[1] / self.env_mask.shape[0]))
        ax = plt.gca()
        ax.clear()
        clear_output(wait = True)

        # Prepare the environment
        env_plot = np.copy(self.env_mask).astype(int)
        env_plot[self.goal[0], self.goal[1]] = 2
        
        colours = ['w', 'grey', 'peru']
        
        # Plot the gridworld.
        cmap = colors.ListedColormap(colours)
        bounds = [0, 1, 2]
        norm = colors.BoundaryNorm(bounds, cmap.N - 1)
        ax.imshow(env_plot, cmap = cmap, norm = norm, zorder = 0)
        
        # Plot position of agent
        ax.scatter(self.pos[1], self.pos[0], color='k', linewidth=6, label='Agent')
        
        # Set up axes.
        ax.grid(which = 'major', axis = 'both', linestyle = '-', color = 'grey', linewidth = 2, zorder = 1)
        ax.set_xticks(np.arange(-0.5, self.env_mask.shape[1] , 1))
        ax.set_xticklabels([])
        ax.set_yticks(np.arange(-0.5, self.env_mask.shape[0], 1))
        ax.set_yticklabels([])
        
#         plt.legend(bbox_to_anchor=(1.15, 1), fontsize=30)
        
        if len(name) != 0:
            plt.savefig('Images/{}'.format(name))
        
        plt.show()