import numpy as np
import matplotlib.pyplot as plt

mu = 0
sigma = 1

vals = mu + sigma * np.random.randn(1000)
plt.hist(vals, 50)

plt.xlabel('bins')
plt.ylabel('Frequncy')
plt.title('Normal distribution(sampled)')
plt.grid(True)

plt.show()
