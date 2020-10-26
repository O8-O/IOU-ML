import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import random

MAX_NUM = 500
MAX_CLUSTER = 4
data = []
for i in range(0, MAX_NUM):
	data.append((random.randint(0, 800), random.randint(0, 800)))
for i in range(0, MAX_NUM):
	data.append((random.randint(200, 600), random.randint(200, 600)))
for i in range(0, MAX_NUM):
	data.append((random.randint(400, 1200), random.randint(400, 1200)))

# 몇개로 나눌것인지 분류.
gmm = GaussianMixture(n_components=MAX_CLUSTER)
gmm.fit(data)
labels = gmm.predict(data)
print(labels)

color=['blue', 'green', 'cyan', 'black']
for k in range(0, 3):
	x = []
	y = []
	for i in range(0, MAX_NUM * 3):
		if labels[i] == k:
			x.append(data[i][0])
			y.append(data[i][1])
	plt.scatter(x, y, c=color[k], s=3)
plt.show()