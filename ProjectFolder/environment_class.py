import numpy as np
'''This script contains the Environment class. This class is used to create an environment instance'''

class Environment:
    def __init__(self, size, dimensions):
        self.radius = size
        self.dimension = dimensions

    def calculate_agent_positions(self, agents):
        '''This method calculates the positions of all agents in the environment.'''
        agent_positions = np.empty((0,2))
        agent_positions = np.append(agent_positions, [agent.position for agent in agents], axis=0)
        return agent_positions