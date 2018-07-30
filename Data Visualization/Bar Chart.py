import matplotlib.pyplot as plt
import numpy as np

bar_width = 0.25
num_bins = 5
bar1 = np.random.randint(0, 100, num_bins)
bar2 = np.random.randint(0, 100, num_bins)
indices = np.arange(num_bins)
plt.bar(indices, bar1, bar_width, label="Subject1")
plt.bar(indices + bar_width, bar2, bar_width, label="Subject2")

plt.xlabel('Final grade')
plt.ylabel("Number of students")

plt.xticks(indices+bar_width/2, ('A', 'B', 'C', 'D', 'E'))
plt.legend()
plt.show()
