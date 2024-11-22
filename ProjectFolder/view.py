from matplotlib import pyplot as plt
import random
import numpy as np
from prey_class import Prey
from environment_class import Environment

'''This script is a alternative to matplotlib.animation.FuncAnimation for creating
 an animation of the model. It uses plt.pause between each iteration to update the plot.'''

POPULATION = 30
ARENA_RADIUS = 30
TIMESTEPS = 1000

env = Environment(ARENA_RADIUS)
all_agents = []

for _ in range(POPULATION):
    r = random.uniform(0, env.radius)
    theta = random.uniform(0, 2*np.pi)

    x = r*np.cos(theta)*0.9
    y = r*np.sin(theta)*0.9
    all_agents.append(Prey(x, y))

#Animation using plt.pause method
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,6))
ax1.set_axis_off()
ax1.set_xlim(-env.radius*1.01, env.radius*1.01)
ax1.set_ylim(-env.radius*1.01, env.radius*1.01)

ax2.set_xlim(0, TIMESTEPS)
ax2.set_ylim(0, 1)
ax2.set_xlabel('Timesteps')
ax2.set_ylabel('Parameter Value')

scatt = ax1.scatter([agent.position[0] for agent in all_agents],
            [agent.position[1] for agent in all_agents],
            c=['blue' if isinstance(agent, Prey) else 'red' for agent in all_agents],
           s=10)

rotation_line, = ax2.plot([], [], label='Rotation Parameter', color='blue')
polarisation_line, = ax2.plot([], [], label='Polarisation Parameter', color='red')
ax2.legend()


centre = [0,0]
radius = env.radius
theta = np.linspace(0, 2*np.pi, 100)
x = centre[0] + radius * np.cos(theta)
y = centre[1] + radius * np.sin(theta)
ax1.plot(x, y, c='black')

rotation_parameter_data = np.zeros(TIMESTEPS)
polarisation_data = np.zeros(TIMESTEPS)

for _ in range(TIMESTEPS):            #Update the scatter plot for each timestep
    all_steering_vectors = np.zeros((len(all_agents), 4, 2))
    total_angular_momentum = 0
    total_heading = np.zeros(2)
    centre_of_mass = sum(agent.position for agent in all_agents) / len(all_agents)
    
    for index, agent in enumerate(all_agents): 
        steering_vector = agent.calculate_steering_vector(all_agents, env)      
        all_steering_vectors[index] = steering_vector

        unit_vector_from_com = (agent.position - centre_of_mass) / np.linalg.norm(agent.position - centre_of_mass)
        angular_momentum = np.cross(agent.speed * agent.direction, unit_vector_from_com)
        total_angular_momentum += angular_momentum

        total_heading += agent.direction

    O_r = abs(total_angular_momentum)/POPULATION
    O_p = np.linalg.norm(total_heading)/POPULATION
    print(f"Total Angular Momentum at timestep {_}: {O_r}")
    print(f"Total Heading at timestep {_}: {O_p}")
    rotation_parameter_data[_] = O_r
    polarisation_data[_] = O_p

    for index, agent in enumerate(all_agents):
        agent.update_position(all_steering_vectors[index])

    scatt.set_offsets([(agent.position[0], agent.position[1]) for agent in all_agents])
    rotation_line.set_data(range(_), rotation_parameter_data[:_])
    polarisation_line.set_data(range(_), polarisation_data[:_])

    plt.pause(0.0001)
plt.show()
