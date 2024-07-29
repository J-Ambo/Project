#%%

import random
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

class Bird:
    def __init__(self, x, y):
        
        self.pos = np.array([x, y])         # Position of the bird
        self.dir = np.linalg.norm(np.array([random.uniform(-1, 1), random.uniform(-1, 1)]))  # Direction of the bird
        self.speed = 0.5                     # Speed of the bird

        self.cohesion_factor = 0.05
        self.separation_factor = 0.05
        self.alignment_factor = 0.05
        self.sep_distance = 30
        self.neighbourhood = 60
        



    def update(self, birds):
        
        align_vector, cohesion_vector, separation_vector, avg_pos = np.zeros((4, 2))

        num_neighbours = 0

        for bird in birds:
            if self.distance(bird) <= self.neighbourhood and bird != self:  # Calculates the average velocity of all birds within ... units of the current bird
                
                num_neighbours += 1    

                align_vector += bird.dir
                avg_pos += bird.pos

                separation_vector += (self.pos - bird.pos)/(self.distance(bird))**2


                #if self.distance(bird) <= self.sep_distance and bird != self:
                    #separation_vector += (self.pos - bird.pos)/(self.distance(bird))**2

                #else:
                    #separation_vector += (self.pos - bird.pos)/(self.distance(bird))**2

           # elif self.distance(env) < 1:
            #    self.pos[0]


        if num_neighbours > 0:        # i.e. the bird has neighbours

            align_vector /= num_neighbours
            avg_pos /= num_neighbours
            separation_vector /= num_neighbours
    

        cohesion_vector = avg_pos - self.pos

        # Update the bird's direction based on the average direction of nearby birds
        self.dir += (align_vector * self.alignment_factor) + (cohesion_vector * self.cohesion_factor) + (separation_vector * self.separation_factor)
        self.dir /= np.linalg.norm(self.dir)



        # Update the bird's position
        #if (self.pos[0], self.pos[1]) >= (100, 100) or (self.pos[0], self.pos[1]) <= (0, 0):  #boundary conditions (if a bird reaches the boundary it will turn around)
            #self.dir = -self.dir
            #self.pos = self.dir * self.speed

        #if self.pos[0] <= env.left_limit or self.pos[0] >= env.right_limit or self.pos[1] <= env.bottom_limit or self.pos[1] >= env.top_limit:
                #self.dir = -self.dir
                #self.pos += self.dir * self.speed

        
        self.pos += self.dir * self.speed
        #self.pos = np.clip(self.pos, 0, 100)  #boundary conditions (if a bird reaches the boundary it will turn around)
        #self.pos %= 3

    def distance(self, other_bird):
        return ((self.pos[0] - other_bird.pos[0]) ** 2 + (self.pos[1] - other_bird.pos[1]) ** 2) ** 0.5




class Environment:
    def __init__(self, width, height):

        self.left_limit = 0
        self.bottom_limit = 0
        self.right_limit = width
        self.top_limit = height 

        #self.left_border = np.array([0, height])
        #self.right_border = np.array([width, 0]) + np.array([0, height])
        #self.top_border = np.array([])
        #self.bottom_border =

    
    

            




#%%
env = Environment(3,3)

#Create a list of birds
birds = []
for _ in range(20):
    x = random.uniform(5, 95)
    y = random.uniform(5, 95)
    birds.append(Bird(x, y))


'''
#Animation using plt.pause method
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)


scatter = ax.scatter([bird.x for bird in birds], [bird.y for bird in birds])


for _ in range(500):            #Update the scatter plot for each iteration
    for bird in birds:
        bird.update(birds)
    
    scatter.set_offsets([(bird.x, bird.y) for bird in birds])

    plt.pause(0.001)
#plt.show()
'''

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
anim = animation.FuncAnimation(fig1, update_frames, frames=500, interval=50)
ax1.plot(np.zeros(100), np.linspace(0, 100, 100), color='black')
ax1.plot(np.full(100,100), np.linspace(0, 100, 100), color='black')
ax1.plot(np.linspace(0,100,100), np.zeros(100), color='black')
ax1.plot(np.linspace(0, 100, 100),np.full(100, 100), color='black')

plt.show(block=True)

from IPython.display import HTML
HTML(anim.to_jshtml())


# %%



2%12
# %%
