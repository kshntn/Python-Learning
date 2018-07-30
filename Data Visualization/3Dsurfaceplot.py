import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x, y = np.meshgrid(np.arange(-100, 100), np.arange(-100, 100))
z = x ** 2 + y ** 2
fig = plt.figure()
axis = fig.gca(projection='3d')
axis.plot_surface(x, y, z)
plt.show()
