import numpy as np
import matplotlib.pyplot as plt

x, y = np.meshgrid(np.arange(-10, 10), np.arange(-10, 10))

U = -y
V = x

plt.quiver(x, y, U, V)
plt.streamplot(x, y, U, V)
plt.show()
