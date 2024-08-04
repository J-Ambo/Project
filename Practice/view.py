from matplotlib import pyplot as plt
import random
from p1 import Bird

birds = []
for _ in range(20):
    x = random.uniform(5, 95)
    y = random.uniform(5, 95)
    birds.append(Bird(x, y))

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
