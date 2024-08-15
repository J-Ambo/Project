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
        



    def update(self, birds):
        
        align_vector, cohesion_vector, separation_vector, avg_pos, wall_separation_vector = np.zeros((5, 2))

        num_neighbours = 0

        for bird in birds:

            # Calculates the average velocity of all birds within ... units of the current bird
            if self.distance_to_birds(bird) <= self.neighbourhood and bird != self: 
                
                num_neighbours += 1    

                align_vector += bird.dir
                avg_pos += bird.pos

                separation_vector += (self.pos - bird.pos)/(self.distance_to_birds(bird))


            

            if self.calculate_min_distance_to_wall(wall)[0] <= self.wall_sep_distance:

                wall_separation_vector += (self.pos - self.calculate_min_distance_to_wall(wall)[1])/(self.calculate_min_distance_to_wall(wall)[0])**2

                #if self.distance(bird) <= self.sep_distance and bird != self:
                    #separation_vector += (self.pos - bird.pos)/(self.distance(bird))**2

                #else:
                    #separation_vector += (self.pos - bird.pos)/(self.distance(bird))**2

          


        if num_neighbours > 0:        # i.e. the bird has neighbours

            align_vector /= num_neighbours
            avg_pos /= num_neighbours
            separation_vector /= num_neighbours

        else:
            pass

        cohesion_vector = avg_pos - self.pos

        # Update the bird's direction based on the average direction of nearby birds
        self.dir += (align_vector * self.alignment_factor) + (cohesion_vector * self.cohesion_factor) + (separation_vector * self.separation_factor) + (wall_separation_vector)
        self.dir /= np.linalg.norm(self.dir)
        
        self.pos += self.dir * self.speed
        #self.pos = np.clip(self.pos, 0, 100)  #boundary conditions (if a bird reaches the boundary it will turn around)
        

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



class Environment:
    def __init__(self, width, height):

        self.left_limit = 0
        self.bottom_limit = 0
        self.right_limit = width
        self.top_limit = height 






#%%
env = Environment(100,100)

nwall = 100
wall = np.empty((4,nwall,2), dtype=float)
left_border = np.asarray(list(zip(np.zeros(nwall), np.linspace(0, 100, nwall))))
right_border = np.asarray(list(zip(np.full(nwall, 100), np.linspace(0, 100, nwall))))
top_border = np.asarray(list(zip(np.linspace(0, 100, nwall), np.full(nwall, 100))))
bottom_border = np.asarray(list(zip(np.linspace(0, 100, nwall), np.zeros(nwall))))

wall[0], wall[1], wall[2], wall[3] = left_border, right_border, top_border, bottom_border


#Create a list of birds
birds = []
for _ in range(10):
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
