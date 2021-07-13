import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

data = np.load("output.npy")
print(data.shape)
pd = data[:, 0]
dist = data[:, 1]
mag = data[:, 2]
print(pd, dist, mag)
data[:, 0] = np.abs(
    data[:, 0]
)  # is the maximum is already the absolute max, we just have to also pass the absolute value
data[:, 1] = data[:, 1] / 1000  # scale from meter to km
X = np.log(data[:, 0:2])  # X data is peak displacement and distance
print(X, X.shape)
y = data[:, 2]  # the magnitude
print(y, y.shape)
reg = LinearRegression().fit(X, y)
coefs = reg.coef_
print(np.round(coefs[0], decimals= 2))
print(np.round(coefs[1],decimals = 2))
print(np.round(reg.intercept_, decimals = 2))
#plt.scatter(X[:, 0], y)
#plt.savefig("regression.png")
