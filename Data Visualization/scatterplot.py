import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets

iris = datasets.load_iris()
# print iris
x = iris.data[:, 0]
y = iris.data[:, 1]
colors = iris.target

plt.scatter(x, y, c=colors)
plt.show()
