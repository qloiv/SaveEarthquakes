import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

data = np.load("output.npy")
print(data.shape)
pd = np.abs(data[:, 0])
dist = data[:, 1]/1000
ma = data[:, 2]
print(pd, dist, ma,max(ma),min(ma))
data[:, 0] = np.abs(
    data[:, 0]
)  # is the maximum is already the absolute max, we just have to also pass the absolute value
data[:, 1] = data[:, 1] / 1000  # scale from meter to km
print(data)
X = np.log10(data[:, 0:2])  # X data is peak displacement and distance
print(X, X.shape)
y = data[:, 2]  # the magnitude
print(y, y.shape)
reg = LinearRegression().fit(X, y)
coefs = reg.coef_
print(np.round(coefs[0], decimals= 2))
print(np.round(coefs[1],decimals = 2))
print(np.round(reg.intercept_, decimals = 2))
mag = 1.01 * np.log10(pd) + 0.74 * np.log10(dist) + 5.47
print("mag",mag,min(mag),max(mag))
print("magzero",np.shape(mag[mag<0]))
print("logPD ",np.log10(pd),min(np.log10(pd)),max(np.log10(pd)))
print("logDist", np.log10(dist),min(np.log10(dist)),max(np.log10(dist)))
#plt.scatter(1.01*np.log10(pd),mag, c = 0.74*np.log10(dist))
#plt.savefig("regression.png")
#plt.hist(ma, bins = 50)
#plt.hist(mag, bins = 50)
#plt.savefig("mag.png")
plt.scatter(ma,np.log10(pd),c = np.log10(dist))
a = 1.01 * np.log10(pd) + 0.74 * np.log10(50) + 5.47
b= 1.01 * np.log10(pd) + 0.74 * np.log10(100) + 5.47
c= 1.01 * np.log10(pd) + 0.74 * np.log10(200) + 5.47
print(a,b,c)
plt.plot(a,np.log10(pd))
plt.plot(b,np.log10(pd))
plt.plot(c,np.log10(pd))
plt.savefig("pdvsma")