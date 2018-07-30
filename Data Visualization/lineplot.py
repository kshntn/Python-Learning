import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(0,10)
y1=np.sin(x)
y2=np.cos(x)

plt.plot(x,y1,'-',label='sine')
plt.plot(x,y2,'-',label='cosine')
plt.legend()


plt.show()