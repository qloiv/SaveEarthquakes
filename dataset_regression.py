import numpy as np
from sklearn.linear_model import LinearRegression

data = np.load("output.npy")
print(data.shape)
pd = data[:, 0]
dist = data[:, 1]
mag = data[:, 2]
print(pd, dist, mag)
data[:, 1] = data[:, 1] / 1000  # scale from meter to km
X = data[:, 0:2]  # X data is peak displacement and distance
y = data[:, 2]  # the magnitude
reg = LinearRegression().fit(X, y)
coefs = reg.coef_
print(coefs)
