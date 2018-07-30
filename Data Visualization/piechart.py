import matplotlib.pyplot as plt

labels = ['gas', 'grocery', 'books', 'rent']
values = [200, 300, 100, 900]

explode = (0, 0, 1, 0)
plt.pie(values, labels=labels, explode=explode, radius=.5,shadow=True,autopct='%f%%')

plt.show()
