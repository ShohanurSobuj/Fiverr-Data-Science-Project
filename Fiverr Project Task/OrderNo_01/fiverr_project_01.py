#importing libraries
import matplotlib
matplotlib.use('TkAgg')

import time
from time import sleep
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image

np.random.seed(42)

N = 10
x_min = 0
x_max = 40
y_min = -20
y_max = 20

x = np.random.uniform(x_min, x_max, N)
y = np.random.uniform(y_min, y_max, N)

size = 10
grid = np.zeros((size, size))
x_grid = np.linspace(x_min, x_max, size + 1)
y_grid = np.linspace(y_min, y_max, size + 1)

for i in range(size):
    for j in range(size):
        for x_i, y_i in zip(x, y):
            if (x_grid[i] < x_i <= x_grid[i + 1]) and (y_grid[j] < y_i <= y_grid[j + 1]):
                grid[i, j] = 1
                break

#sharex Controls sharing of properties among x (sharex) or y (sharey) axes 
#for better understand read the documentation here : https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6), sharex=True)
ax1.scatter(x, y)
ax1.set_xlim(x_min, x_max)
ax1.set_ylim(y_min, y_max)
ax1.grid()
ax1.set_xticks(x_grid)
ax1.set_yticks(y_grid)

ax2.imshow(grid.T, cmap = 'Greys', extent = (x_min, x_max, y_min, y_max))
#The Axes.invert_yaxis() function in axes module of matplotlib library is used to invert the y-axis.
ax2.invert_yaxis()
#plt.show(block=False)

while True:  
    try:
        fig.canvas.draw()
        pil_img = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
        print("Demo",pil_img)
        plt.pause(0.05)
        pil_img.save('/home/shohanursobuj/MEGAsync/Fiverr/test/IMG {0}.png'.format(time.time()))
        time.sleep(1)
        
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        break


