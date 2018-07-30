import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


phi=np.linspace(-6*np.pi,6*np.pi,100)
z=np.linspace(-4,4,100)
x=np.sin(phi)
y=np.cos(phi)

fig=plt.figure()
axis=fig.gca(projection='3d')

axis.plot(x,y,z)
plt.show()