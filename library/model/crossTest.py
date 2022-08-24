from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

_x = PolynomialFeatures(degree=zisuu)

model1 = LinearRegression()
X = _x.fit_transform(x[:,None])
model1.fit(X, train_Y)

pred = model1.predict(X)

y_close = np.allclose(pred, pridict_Y)
print(y_close)
