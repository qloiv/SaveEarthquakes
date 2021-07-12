import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = np.load("output.npy")
print(data.shape)
pd = data[:, 0]
dist = data[:, 1]
mag = data[:, 2]
print(pd, dist, mag)
data[:,0] = np.abs(data[:,0]) # is the maximum is already the absolute max, we just have to also pass the absolute value
data[:, 1] = data[:, 1] / 1000  # scale from meter to km
X = np.log(data[:, 0:2])  # X data is peak displacement and distance
print(X, X.shape)
y = data[:, 2]  # the magnitude
print(y,y.shape)
reg = LinearRegression().fit(X, y)
coefs = reg.coef_
print(coefs[0])
print(coefs[1])
plt.scatter(X[:,0],y)
plt.savefig("regression.png")
