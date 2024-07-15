#%%

import random
import matplotlib.animation as animation
import matplotlib.pyplot as plt

class Bird:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = random.uniform(-1, 1)
        self.vy = random.uniform(-1, 1)

        self.cohesion = 0.1
        self.separation = 0.5
        self.alignment = 0.1
        self.separation_distance = 10



    def update(self, birds):
        
        align_vector, cohesion_vector, separation_vector = [[0,0], [0,0], [0,0]]

        avg_vx = 0
        avg_vy = 0
        num_neighbours = 0
        avg_xpos = 0
        avg_ypos = 0

        for bird in birds:
            if self.distance(bird) < 15 and bird != self:  # Calculates the average velocity of all birds within ... units of the current bird...
                
                num_neighbours += 1    

                avg_vx += bird.vx
                avg_vy += bird.vy
                
                avg_xpos += bird.x
                avg_ypos += bird.y
                


        if num_neighbours > 0:        # i.e. the bird has neighbours
            avg_vx /= num_neighbours
            avg_vy /= num_neighbours

            avg_xpos /= num_neighbours
            avg_ypos /= num_neighbours

        # Update the bird's velocity based on the average velocity of nearby birds
        self.vx = (self.vx + avg_vx)/2  +  (avg_xpos - self.x)/2
        self.vy = (self.vy + avg_vy)/2  +  (avg_ypos - self.y)/2

        # Update the bird's position
        self.x += self.vx
        self.y += self.vy

        if (self.x, self.y) == (0 or 100, 0 or 100):  #boundary conditions, if a bird reaches the boundary it will turn around
            self.vx = -self.vx
            self.vy = -self.vy


    def distance(self, other_bird):
        return ((self.x - other_bird.x) ** 2 + (self.y - other_bird.y) ** 2) ** 0.5



#%%

#Create a list of birds
birds = []
for _ in range(50):
    x = random.uniform(10, 90)
    y = random.uniform(10, 90)
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
ax1.set_xlim(-500, 500)
ax1.set_ylim(-500, 500)
scatt = ax1.scatter([bird.x for bird in birds], [bird.y for bird in birds])

#Update function for the animation
def update_frames(frame):
    for bird in birds:
        bird.update(birds)

    # Update the scatter plot data
    scatt.set_offsets([(bird.x, bird.y) for bird in birds])
    return scatt

# Create the animation
anim = animation.FuncAnimation(fig1, update_frames, frames=50, interval=10)

plt.show(block=True)

from IPython.display import HTML
HTML(anim.to_jshtml())


# %%
