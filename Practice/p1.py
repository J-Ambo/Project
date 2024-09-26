#%%

import random
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

class Bird:
    def __init__(self, x, y):
        
        self.pos = np.array([x, y])         # Position of the bird
        self.dir = np.linalg.norm(np.array([random.uniform(-1, 1), random.uniform(-1, 1)]))  # Direction of the bird
        self.speed = 1                     # Speed of the bird

        self.cohesion_factor = 0.02
        self.separation_factor = 0.1
        self.alignment_factor = 0.01
        self.wall_sep_distance = 2
        self.neighbourhood = 20
        

    def align_vector(self, birds):
        align_vector = np.zeros(2)
        num_neighbours = 0

        for bird in birds:
            if self.distance_to_birds(bird) <= self.neighbourhood and bird != self: 
                align_vector += bird.dir
                num_neighbours += 1

        if num_neighbours > 0:
            align_vector /= num_neighbours

        return align_vector

    def cohesion_vector(self, birds):
        cohesion_vector, avg_pos = np.zeros((2,2))
        num_neighbours = 0

        for bird in birds:
            if self.distance_to_birds(bird) <= self.neighbourhood and bird != self: 
                avg_pos += bird.pos
                num_neighbours += 1

        if num_neighbours > 0:
            avg_pos /= num_neighbours
            cohesion_vector = avg_pos - self.pos

        return cohesion_vector

    def separation_vector(self, birds):
        separation_vector = np.zeros(2)
        num_neighbours = 0

        for bird in birds:
            if self.distance_to_birds(bird) <= self.neighbourhood and bird != self: 
                separation_vector += (self.pos - bird.pos)/(self.distance_to_birds(bird))
                num_neighbours += 1

        if num_neighbours > 0:
            separation_vector /= num_neighbours

        return separation_vector
    
    def wall_separation_vector(self, wall):
        wall_separation_vector = np.zeros(2)
        if self.calculate_min_distance_to_wall(wall)[0] <= self.wall_sep_distance:
            wall_separation_vector = (self.pos - self.calculate_min_distance_to_wall(wall)[1])/(self.calculate_min_distance_to_wall(wall)[0])**2

        return wall_separation_vector

    def distance_to_birds(self, other_bird):
        return ((self.pos[0] - other_bird.pos[0])**2 + (self.pos[1] - other_bird.pos[1])**2)**0.5

    def calculate_min_distance_to_wall(self, wall):

        min_distance = float('inf')
        point_on_wall = np.zeros(2)
        for wall_segment in wall:
            for i in range(len(wall_segment) - 1):
                distance = ((self.pos[0] - wall_segment[i][0])**2 + (self.pos[1]- wall_segment[i][1])**2)**0.5
                if distance < min_distance:
                    min_distance = distance
                    point_on_wall = wall_segment[i]
        return min_distance, point_on_wall

    def update(self, birds):

        # Update the bird's direction based on the average direction of nearby birds
        self.dir += (self.align_vector * self.alignment_factor) + (self.cohesion_vector * self.cohesion_factor) + (self.separation_vector * self.separation_factor) + (self.wall_separation_vector)
        self.dir /= np.linalg.norm(self.dir)
        
        self.pos += self.dir * self.speed
        


class Environment:
    def __init__(self, size):

        self.left_limit = 0
        self.bottom_limit = 0
        self.right_limit = size
        self.top_limit = size 






#%%
env = Environment(100)

nwall = 100
wall = np.empty((4,nwall,2), dtype=float)
left_border = np.asarray(list(zip(np.zeros(nwall), np.linspace(0, 100, nwall))))
right_border = np.asarray(list(zip(np.full(nwall, 100), np.linspace(0, 100, nwall))))
top_border = np.asarray(list(zip(np.linspace(0, 100, nwall), np.full(nwall, 100))))
bottom_border = np.asarray(list(zip(np.linspace(0, 100, nwall), np.zeros(nwall))))

wall[0], wall[1], wall[2], wall[3] = left_border, right_border, top_border, bottom_border


#Create a list of birds
birds = []
for _ in range(20):
    x = random.uniform(5, 95)
    y = random.uniform(5, 95)
    birds.append(Bird(x, y))




#%%


#Animation using FuncAnimation method
fig1 = plt.figure(figsize=(7, 7))
ax1 = fig1.add_subplot(111)
ax1.set_xlim(-5, 105)
ax1.set_ylim(-5, 105)
scatt = ax1.scatter([bird.pos[0] for bird in birds], [bird.pos[1] for bird in birds])

#Update function for the animation
def update_frames(frame):
    for bird in birds:
        bird.update(birds)

    # Update the scatter plot data
    scatt.set_offsets([(bird.pos[0], bird.pos[1]) for bird in birds])
    return scatt

# Create the animation
anim = animation.FuncAnimation(fig1, update_frames, frames=500, interval=50, repeat=True)

ax1.plot(np.zeros(100), np.linspace(0, 100, 100), color='black')
ax1.plot(np.full(100,100), np.linspace(0, 100, 100), color='black')
ax1.plot(np.linspace(0,100,100), np.zeros(100), color='black')
ax1.plot(np.linspace(0, 100, 100),np.full(100, 100), color='black')

plt.show(block=True)

from IPython.display import HTML
HTML(anim.to_jshtml())



# %%
