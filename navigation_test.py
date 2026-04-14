import numpy as np
import matplotlib.pyplot as plt
from astar import astar

# Create grid
grid = np.zeros((10,10))

# Add obstacles
grid[3,3] = 1
grid[3,4] = 1
grid[4,4] = 1
grid[6,7] = 1

start = (0,0)
goal = (9,9)

path = astar(grid, start, goal)

print("Path:", path)

# Plot
plt.imshow(grid, cmap='gray_r')

if path:
    x = [p[1] for p in path]
    y = [p[0] for p in path]
    plt.plot(x, y)

plt.scatter(start[1], start[0])
plt.scatter(goal[1], goal[0])

plt.title("Autonomous Navigation Path")
plt.show()  