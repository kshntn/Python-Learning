import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10)
speed_plot = plt.subplot(2, 1, 1)
plt.plot(x, np.sin(x), '-', label='sin')
plt.ylabel('speed (m/s)')
plt.setp(speed_plot.get_xticklabels(), visible=False)
plt.grid(True)

plt.subplot(2, 1, 2, sharex=speed_plot)
plt.plot(x, np.cos(x), '-', label='cosine')
plt.ylabel('acceleration (m/s/s)')
plt.xlabel('time(s)')
plt.grid(True)
plt.show()
