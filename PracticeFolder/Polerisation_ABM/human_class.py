import numpy as np

class PopulationAgent:
    def __init__(self, age_range, sex, average_opinion):
        self.age = age_range
        self.sex = sex
        self.opinion = average_opinion

    def interact(self, other_agents):
        new_opinion = self.opinion
        for agent in other_agents:
            new_opinion += (0.6)*agent.opinion

    def interaction_weight(self, other_agent):
        pass

agent_matrix = np.array([[PopulationAgent(18, 'M', 0.5).opinion]]*2)

print(agent_matrix)    
