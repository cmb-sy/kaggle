import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def linear_regression(train_X, train_Y, X_star):
    tmp = np.linalg.inv(train_X.T @ train_X) @ (train_X.T @ train_Y)
    pridict = X_star @ tmp[:,None]
    return pridict


if __name__ == '__main__':
    learning_num = 50
    np.random.seed(0)
    x_min = 10
    x_max = 40
    x = np.linspace(x_min, x_max, learning_num) # temprature
    epsilon = np.random.normal(loc=0, scale=7e-0, size=len(x)) #ガウシアンノイズ
    y = -x + epsilon # sales

    train_X = np.concatenate((np.ones((learning_num, 1)), x[:, None]), axis=1)
    train_Y = y

    # 新規データ
    num_star = 500
    X_star = np.linspace(x_min, x_max, num_star)
    X_star = np.concatenate((np.ones((num_star, 1)), X_star[:, None]), axis=1)

    pridict_Y = linear_regression(train_X, train_Y, X_star)
#   print("予測データ", pridict_Y.shape)
    plt.scatter(train_X[:, 1], y, s=20, c="b")
    plt.plot(X_star[:, 1], pridict_Y, c="r", lw=5)
    # plt.xlabel("気温")
    # plt.legend()
    plt.show()
