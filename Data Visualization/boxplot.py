import numpy as np
import matplotlib.pyplot as plt

x=np.random.rand(50)*100

plt.boxplot(x,vert=False)
print x
plt.show()