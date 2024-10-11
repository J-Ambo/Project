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

        self.cohesion_factor = 0.05
        self.separation_factor = 0.1
        self.alignment_factor = 0.02
        self.wall_sep_distance = 2
        self.neighbourhood = 10

    def calculate_align_vector(self, birds):
        align_vector = np.zeros(2)
        num_neighbours = 0

        for bird in birds:
            if self.calculate_distance_to_birds(bird) <= self.neighbourhood and bird != self: 
                align_vector += bird.dir
                num_neighbours += 1

        if num_neighbours > 0:
            align_vector /= num_neighbours

        return align_vector

    def calculate_cohesion_vector(self, birds):
        cohesion_vector, avg_pos = np.zeros((2,2))
        num_neighbours = 0

        for bird in birds:
            if self.calculate_distance_to_birds(bird) <= self.neighbourhood and bird != self: 
                avg_pos += bird.pos
                num_neighbours += 1

        if num_neighbours > 0:
            avg_pos /= num_neighbours
            cohesion_vector = avg_pos - self.pos

        return cohesion_vector

    def calculate_separation_vector(self, birds):
        separation_vector = np.zeros(2)
        num_neighbours = 0

        for bird in birds:
            if self.calculate_distance_to_birds(bird) <= self.neighbourhood and bird != self: 
                separation_vector += (self.pos - bird.pos)/(self.calculate_distance_to_birds(bird))
                num_neighbours += 1

        if num_neighbours > 0:
            separation_vector /= num_neighbours

        return separation_vector
    '''
    def calculate_wall_separation_vector(self, wall):
        wall_separation_vector = np.zeros(2)
        if self.calculate_min_distance_to_wall(env.create_walls)[0] <= self.wall_sep_distance:
            wall_separation_vector = (self.pos - self.calculate_min_distance_to_wall(env.create_walls)[1])/(self.calculate_min_distance_to_wall(env.create_walls)[0])**0.5

        return wall_separation_vector
    '''
    def calculate_distance_to_birds(self, other_bird):
        return ((self.pos[0] - other_bird.pos[0])**2 + (self.pos[1] - other_bird.pos[1])**2)**0.5

    '''
    def calculate_min_distance_to_wall(self, wall):
        min_distance = float('inf')
        point_on_wall = np.zeros(2)
        for wall_segment in env.create_walls():
            for i in range(env.nwall):
                distance = ((self.pos[0] - wall_segment[i][0])**2 + (self.pos[1]- wall_segment[i][1])**2)**0.5
                if distance < min_distance:
                    min_distance = distance
                    point_on_wall = wall_segment[i]

        return min_distance, point_on_wall
    '''

def point_is_out_of_bounds(coord, size):
    if coord >= size:
        return True
    elif coord <= 0:
        return True
    else:
        return False

def apply_boudary_condition(coord, size):
    if coord >= size:
        return size - 0.01*size
    elif coord <= 0:
        return 0 + 0.01*size
        
    


class Prey(Bird):
    def __init__(self, x, y):
        super().__init__(x, y)

    def calculate_predator_separation_vector(self, birds):
        predator_separation_vector = np.zeros(2)
        num_neighbours = 0

        for bird in birds:
            if isinstance(bird, Predator) and self.calculate_distance_to_birds(bird) <= self.neighbourhood:
                predator_separation_vector += (self.pos - bird.pos)/(self.calculate_distance_to_birds(bird))
                num_neighbours += 1
        
        if num_neighbours > 0:
            predator_separation_vector /= num_neighbours

        return predator_separation_vector
    
    def update_prey(self, birds):
        self.dir += (self.alignment_factor * self.calculate_align_vector(birds)) + (self.cohesion_factor * self.calculate_cohesion_vector(birds)) + (self.separation_factor * self.calculate_separation_vector(birds)) + (self.calculate_predator_separation_vector(birds)) # (self.calculate_wall_separation_vector(env.create_walls()))
        self.dir /= np.linalg.norm(self.dir)   
        self.pos += self.dir * self.speed


class Predator(Bird):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.speed = 0.5*self.speed

    def calculate_separation_vector(self, birds):  #Instead of steerng away the predator steers towards prey
        separation_vector = np.zeros(2)
        num_neighbours = 0

        for bird in birds:
            if self.calculate_distance_to_birds(bird) <= self.neighbourhood and bird != self: 
                separation_vector += (-self.pos + bird.pos)/(self.calculate_distance_to_birds(bird))
                num_neighbours += 1

        if num_neighbours > 0:
            separation_vector /= num_neighbours

        return separation_vector
    
    def update_predator(self, birds):
        self.dir += (self.alignment_factor * self.calculate_align_vector(birds)) + (self.cohesion_factor * self.calculate_cohesion_vector(birds)) + (self.separation_factor * self.calculate_separation_vector(birds)) #(self.calculate_wall_separation_vector(env.create_walls()))
        self.dir /= np.linalg.norm(self.dir) 

        if point_is_out_of_bounds(self.pos[0], env.size):
            apply_boudary_condition(self.pos[0], env.size)
        
        else:
            self.pos += self.dir * self.speed

        if point_is_out_of_bounds(self.pos[1], env.size):
            apply_boudary_condition(self.pos[1], env.size)

        else:
            self.pos += self.dir * self.speed
        

class Environment:
    def __init__(self, size):
        self.size = size
        self.left_limit = 0
        self.bottom_limit = 0
        self.right_limit = size
        self.top_limit = size 
        self.nwall = int(size*1.1)  #number of points on the wall

'''
    def create_walls(self): 
        wall = np.empty((4,self.nwall,2), dtype=float)
        left_border = np.asarray(list(zip(np.zeros(self.nwall), np.linspace(0, self.size, self.nwall))))
        right_border = np.asarray(list(zip(np.full(self.nwall, self.size), np.linspace(0, self.size, self.nwall))))
        top_border = np.asarray(list(zip(np.linspace(0, self.size, self.nwall), np.full(self.nwall, self.size))))
        bottom_border = np.asarray(list(zip(np.linspace(0, self.size, self.nwall), np.zeros(self.nwall))))

        wall[0], wall[1], wall[2], wall[3] = left_border, right_border, top_border, bottom_border

        return wall
'''



#%%
env = Environment(60)

#Create a list of birds
all_predators = []
for _ in range(1):
    x = random.uniform(env.size*0.05, env.size*0.95)
    y = random.uniform(env.size*0.05, env.size*0.95)
    all_predators.append(Predator(x, y))

all_prey = []
for _ in range(50):
    x = random.uniform(env.size*0.05, env.size*0.95)
    y = random.uniform(env.size*0.05, env.size*0.95)
    all_prey.append(Prey(x, y))

all_birds = all_prey + all_predators


#%%


#Animation using FuncAnimation method
fig1 = plt.figure(figsize=(7, 7))
ax1 = fig1.add_subplot(111)
ax1.set_axis_off()
ax1.set_xlim(-env.size*0.05, env.size*1.05)
ax1.set_ylim(-env.size*0.05, env.size*1.05)
scatt = ax1.scatter([bird.pos[0] for bird in all_birds], [bird.pos[1] for bird in all_birds], c=['blue' if isinstance(bird, Prey) else 'red' for bird in all_birds], s=10)

#Update function for the animation
def update_frames(frame):
    for prey in all_prey:
        prey.update_prey(all_birds)
    for predator in all_predators:
        predator.update_predator(all_birds)

    # Update the scatter plot data
    scatt.set_offsets([(bird.pos[0], bird.pos[1]) for bird in all_birds])
    return scatt

# Create the animation
anim = animation.FuncAnimation(fig1, update_frames, frames=500, interval=50, repeat=True)

ax1.plot(np.zeros(100), np.linspace(0, env.size, 100), color='black')
ax1.plot(np.full(100,env.size), np.linspace(0, env.size, 100), color='black')
ax1.plot(np.linspace(0,env.size,100), np.zeros(100), color='black')
ax1.plot(np.linspace(0, env.size, 100),np.full(100, env.size), color='black')

plt.show(block=True)

from IPython.display import HTML
HTML(anim.to_jshtml())




# %%
