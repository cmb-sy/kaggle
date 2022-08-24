from ast import Pass
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def polynomial_regression(train_X, train_Y, X_star1):
    tmp = np.linalg.pinv(train_X.T @ train_X) @ (train_X.T @ train_Y)
    pridict = X_star1 @ tmp
    return pridict

if __name__ == '__main__':
    num = 10
    zisuu = 10
    x_min = -np.pi * 2
    x_max = np.pi * 2
    np.random.seed(0)
    
    x = np.linspace(x_min, x_max, num) 
    epsilon = np.random.normal(loc=0, scale=0, size=len(x)) #ガウシアンノイズ
    y = np.sin(x) + epsilon 
    train_X = np.concatenate((np.ones((num, 1)), x[:, None]), axis=1)

    for i in range(zisuu):
        tmp_x = x ** (i+1)
        train_X = np.concatenate((train_X, tmp_x[:, None]), axis=1)

    train_Y = y
    
    
    
    star_num = 500
    X_star =  np.linspace(x_min, x_max, star_num)
    
    X_star1 = np.concatenate((np.ones((star_num, 1)), X_star[:, None]), axis=1)
    for i in range(zisuu):
        tmp_x1 = X_star ** (i+1)
        X_star1 = np.concatenate((X_star1, tmp_x1[:, None]), axis=1)

    pridict_Y = polynomial_regression(train_X, train_Y, X_star1)
#   print("予測データ", pridict_Y.shape)
    plt.scatter(train_X[:, 1], y, s=50, c="b")
    plt.plot(X_star1[:, 1], pridict_Y, c="r")
    # plt.xlabel("気温")
    # plt.legend()
    plt.show()
